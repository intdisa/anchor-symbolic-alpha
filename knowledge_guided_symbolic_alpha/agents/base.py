from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..envs import RoleActionMask, TreeStructuredLanguageMDP
from ..evaluation import FormulaEvaluator, preview_candidate_on_dataset
from ..generation import FormulaCandidate
from ..evaluation.role_profiles import normalize_role, resolve_role_profile
from ..language import ParseError
from ..language import RPNGrammar
from ..memory import ExperienceMemory
from ..models.embeddings import DatasetEmbedder, FormulaEmbedder, PoolEmbedder
from ..models.generator import TransformerGenerator, TreePolicy, TreeValue
from ..models.generator import GeneratorConditioningContext
from ..search import (
    BeamSearch,
    BeamSearchConfig,
    FormulaSampler,
    GrammarMCTS,
    GrammarMCTSConfig,
    ProposalCandidate,
    ProposalMixer,
    SamplingConfig,
    SearchEvaluation,
)


@dataclass(frozen=True)
class AgentProposal:
    role: str
    body_tokens: tuple[str, ...]
    logprob: float
    valid: bool
    terminal_error: str | None
    candidate_records: tuple[FormulaCandidate, ...] = tuple()


class BaseRoleAgent:
    def __init__(
        self,
        role: str,
        allowed_features: frozenset[str],
        allowed_operators: frozenset[str] | None = None,
        grammar: RPNGrammar | None = None,
        generator=None,
        experience_memory: ExperienceMemory | None = None,
        sampling_config: SamplingConfig | None = None,
        beam_config: BeamSearchConfig | None = None,
        mcts_config: GrammarMCTSConfig | None = None,
        token_score_biases: dict[str, float] | None = None,
        max_length_cap: int | None = None,
        seed_formulas: tuple[tuple[str, ...], ...] | None = None,
        seed_priority_enabled: bool = True,
    ) -> None:
        self.role = role
        self.allowed_features = allowed_features
        self.allowed_operators = allowed_operators
        self.grammar = grammar or RPNGrammar()
        self.generator = generator or TransformerGenerator()
        self.experience_memory = experience_memory
        self.token_score_biases = token_score_biases or {}
        self.max_length_cap = max_length_cap
        self.seed_formulas = seed_formulas or tuple()
        self.seed_priority_enabled = seed_priority_enabled
        self.current_regime = "BALANCED"
        self.mask = RoleActionMask(
            role=role,
            allowed_features=allowed_features,
            allowed_operators=allowed_operators,
        )
        self.sampler = FormulaSampler(
            self.grammar,
            self.generator,
            config=sampling_config,
            token_filter=self.mask.filter_tokens,
            score_adjuster=self._adjust_scores_with_memory,
        )
        self.beam_search = BeamSearch(
            self.grammar,
            self.generator,
            config=beam_config,
            token_filter=self.mask.filter_tokens,
            score_adjuster=self._adjust_scores_with_memory,
        )
        self.tree_policy = TreePolicy(max_length=self.grammar.max_length)
        self.tree_value = TreeValue(max_length=self.grammar.max_length)
        self.grammar_mdp = TreeStructuredLanguageMDP(self.grammar, token_filter=self.mask.filter_tokens)
        self.grammar_mcts = GrammarMCTS(
            self.grammar_mdp,
            self.generator,
            self.tree_policy,
            self.tree_value,
            config=mcts_config,
            score_adjuster=self._adjust_scores_with_memory,
        )
        self.proposal_mixer = ProposalMixer()
        self.use_mcts = False
        self.evaluator = FormulaEvaluator()
        self.dataset_embedder = DatasetEmbedder()
        self.pool_embedder = PoolEmbedder()
        self.formula_embedder = FormulaEmbedder()

    def propose(
        self,
        data: pd.DataFrame | None = None,
        target: pd.Series | None = None,
        pool=None,
        reward_shaper=None,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> AgentProposal:
        self._refresh_conditioning_context(
            data=data,
            target=target,
            pool=pool,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        candidates: list[ProposalCandidate] = []
        for tokens in self.seed_formulas:
            valid, terminal_error = self._validate_seed_formula(tokens)
            candidates.append(
                ProposalCandidate(
                    source="seed",
                    body_tokens=tokens,
                    score=0.25,
                    valid=valid,
                    terminal_error=terminal_error,
                )
            )
        sample = self.sampler.sample()
        candidates.append(
            ProposalCandidate(
                source="sample",
                body_tokens=sample.body_tokens,
                score=float(sample.logprob),
                valid=sample.valid,
                terminal_error=sample.terminal_error,
            )
        )
        candidates.extend(
            ProposalCandidate(
                source="beam",
                body_tokens=candidate.body_tokens,
                score=float(candidate.score),
                valid=candidate.valid,
                terminal_error=candidate.terminal_error,
            )
            for candidate in self.ranked_candidates(limit=5)
        )
        evaluator = None
        if data is not None and target is not None and pool is not None and reward_shaper is not None:
            evaluator = lambda tokens: self._evaluate_candidate(
                tokens,
                data,
                target,
                pool,
                reward_shaper,
                validation_data=validation_data,
                validation_target=validation_target,
            )
            if self.use_mcts:
                candidate = self.grammar_mcts.search(evaluator)
                candidates.append(
                    ProposalCandidate(
                        source="mcts",
                        body_tokens=candidate.body_tokens,
                        score=float(candidate.score),
                        valid=candidate.valid,
                        terminal_error=candidate.terminal_error,
                        accepted=candidate.accepted,
                        reason=candidate.reason,
                    )
                )
        ranked = self.proposal_mixer.rerank(candidates, evaluator=evaluator)
        candidate_records = self._build_candidate_records(ranked)
        if self.seed_priority_enabled:
            bootstrap_seed = self._select_bootstrap_seed_override(candidates, pool, evaluator)
            if bootstrap_seed is not None:
                return AgentProposal(
                    role=self.role,
                    body_tokens=bootstrap_seed.body_tokens,
                    logprob=float(bootstrap_seed.score),
                    valid=bootstrap_seed.valid,
                    terminal_error=bootstrap_seed.terminal_error,
                    candidate_records=candidate_records,
                )
            novel_seed = self._select_novel_seed_override(candidates, pool, evaluator)
            if novel_seed is not None:
                return AgentProposal(
                    role=self.role,
                    body_tokens=novel_seed.body_tokens,
                    logprob=float(novel_seed.score),
                    valid=novel_seed.valid,
                    terminal_error=novel_seed.terminal_error,
                    candidate_records=candidate_records,
                )
        best = self._prefer_novel_ranked_candidate(ranked, pool)
        if best is None:
            return AgentProposal(
                role=self.role,
                body_tokens=sample.body_tokens,
                logprob=sample.logprob,
                valid=sample.valid,
                terminal_error=sample.terminal_error,
                candidate_records=candidate_records,
            )
        return AgentProposal(
            role=self.role,
            body_tokens=best.body_tokens,
            logprob=float(best.score),
            valid=best.valid,
            terminal_error=best.terminal_error,
            candidate_records=candidate_records,
        )

    def _refresh_conditioning_context(
        self,
        data: pd.DataFrame | None,
        target: pd.Series | None,
        pool,
        validation_data: pd.DataFrame | None,
        validation_target: pd.Series | None,
    ) -> None:
        if not hasattr(self.generator, "set_conditioning_context"):
            return
        if data is None or pool is None:
            self.generator.set_conditioning_context(None)
            return
        dataset_embedding = self.dataset_embedder.embed(
            data,
            self.allowed_features,
            target=target,
            regime=self.current_regime,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        pool_embedding = self.pool_embedder.embed(pool, self.allowed_features, self.role)
        token_biases = dict(dataset_embedding.token_biases)
        for token, bias in pool_embedding.token_biases.items():
            token_biases[token] = token_biases.get(token, 0.0) + bias
        summary_vector = dataset_embedding.vector + pool_embedding.vector
        signature = (
            self.role,
            self.current_regime,
            f"pool:{len(pool.records)}",
            f"price:{sum(normalize_role(record.role) == 'target_price' for record in pool.records)}",
            f"flow:{sum(normalize_role(record.role) == 'target_flow' for record in pool.records)}",
        )
        self.generator.set_conditioning_context(
            GeneratorConditioningContext(
                summary_vector=summary_vector,
                token_biases=token_biases,
                signature=signature,
            )
        )

    def _validate_seed_formula(self, tokens: tuple[str, ...]) -> tuple[bool, str | None]:
        try:
            self.evaluator.parser.parse(tokens)
        except ParseError as exc:
            return False, str(exc)
        return True, None

    def _canonical_from_tokens(self, tokens: tuple[str, ...]) -> str | None:
        try:
            return self.evaluator.parser.parse(tokens).canonical
        except ParseError:
            return None

    def _build_candidate_records(self, ranked: list[ProposalCandidate], limit: int = 8) -> tuple[FormulaCandidate, ...]:
        records: list[FormulaCandidate] = []
        seen: set[str] = set()
        for candidate in ranked:
            if not candidate.valid:
                continue
            canonical = self._canonical_from_tokens(candidate.body_tokens)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            records.append(
                FormulaCandidate(
                    formula=" ".join(candidate.body_tokens),
                    source=candidate.source,
                    role=self.role,
                )
            )
            if len(records) >= limit:
                break
        return tuple(records)

    def _select_novel_seed_override(self, candidates, pool, evaluator):
        if pool is None or len(pool.records) == 0:
            return None
        pool_canonicals = pool.canonicals()
        cross_sectional_pool = any(bool(record.metrics.get("cross_sectional", 0.0)) for record in pool.records)
        novel_seed_candidates = [
            candidate
            for candidate in candidates
            if candidate.source == "seed"
            and candidate.valid
            and (canonical := self._canonical_from_tokens(candidate.body_tokens)) is not None
            and canonical not in pool_canonicals
        ]
        if not novel_seed_candidates:
            return None
        ranked: list[ProposalCandidate] = []
        for candidate in novel_seed_candidates:
            evaluation = evaluator(candidate.body_tokens)
            mixed_score = (
                (1.0 - self.proposal_mixer.evaluation_weight) * candidate.score
                + self.proposal_mixer.evaluation_weight * evaluation.score
                + self.proposal_mixer.source_weights.get("seed", 0.0)
                + (self.proposal_mixer.acceptance_bonus if evaluation.accepted else 0.0)
                + 0.20
                + self._cross_sectional_target_price_seed_bonus(candidate.body_tokens, pool)
            )
            ranked.append(
                ProposalCandidate(
                    source=candidate.source,
                    body_tokens=candidate.body_tokens,
                    score=float(mixed_score),
                    valid=candidate.valid,
                    terminal_error=candidate.terminal_error,
                    accepted=evaluation.accepted,
                    reason=evaluation.reason,
                )
            )
        ranked.sort(key=lambda item: (item.accepted, item.score), reverse=True)
        best = ranked[0]
        if best.accepted:
            return best
        if cross_sectional_pool and normalize_role(self.role) == "target_price":
            return best
        return None

    def _select_bootstrap_seed_override(self, candidates, pool, evaluator):
        if pool is None or len(pool.records) != 0:
            return None
        if normalize_role(self.role) != "target_price":
            return None
        seed_candidates = [
            candidate
            for candidate in candidates
            if candidate.source == "seed" and candidate.valid
        ]
        if not seed_candidates:
            return None
        ranked: list[ProposalCandidate] = []
        for candidate in seed_candidates:
            evaluation = evaluator(candidate.body_tokens)
            mixed_score = (
                (1.0 - self.proposal_mixer.evaluation_weight) * candidate.score
                + self.proposal_mixer.evaluation_weight * evaluation.score
                + self.proposal_mixer.source_weights.get("seed", 0.0)
                + (self.proposal_mixer.acceptance_bonus if evaluation.accepted else 0.0)
                + 0.30
                + self._bootstrap_target_price_seed_bonus(candidate.body_tokens)
            )
            ranked.append(
                ProposalCandidate(
                    source=candidate.source,
                    body_tokens=candidate.body_tokens,
                    score=float(mixed_score),
                    valid=candidate.valid,
                    terminal_error=candidate.terminal_error,
                    accepted=evaluation.accepted,
                    reason=evaluation.reason,
                )
            )
        ranked.sort(key=lambda item: (item.accepted, item.score), reverse=True)
        return ranked[0]

    def _cross_sectional_target_price_seed_bonus(self, tokens: tuple[str, ...], pool) -> float:
        if normalize_role(self.role) != "target_price":
            return 0.0
        if pool is None or not pool.records:
            return 0.0
        if not any(bool(record.metrics.get("cross_sectional", 0.0)) for record in pool.records):
            return 0.0
        bonus = 0.0
        if "ADD" in tokens:
            bonus += 0.30
        if len(tokens) >= 5:
            bonus += 0.08
        if len(tokens) == 2 and tokens[-1] == "RANK":
            bonus -= 0.12
        baseline = next(
            (record for record in pool.records if normalize_role(record.role) == "target_price"),
            None,
        )
        if baseline is not None and "ADD" in tokens and tokens[0] in baseline.tokens:
            bonus += 0.04
        return bonus

    def _bootstrap_target_price_seed_bonus(self, tokens: tuple[str, ...]) -> float:
        if normalize_role(self.role) != "target_price":
            return 0.0
        bonus = 0.0
        if "ADD" in tokens:
            bonus += 0.35
        if "CASH_RATIO_Q" in tokens:
            bonus += 0.08
        if "SALES_TO_ASSETS_Q" in tokens:
            bonus += 0.08
        if len(tokens) == 2 and tokens[-1] == "RANK":
            bonus -= 0.10
        return bonus

    def _prefer_novel_ranked_candidate(self, ranked_candidates, pool):
        if not ranked_candidates:
            return None
        if pool is None or len(pool.records) == 0:
            return ranked_candidates[0]
        pool_canonicals = pool.canonicals()
        best = ranked_candidates[0]
        best_canonical = self._canonical_from_tokens(best.body_tokens)
        if best_canonical is None or best_canonical not in pool_canonicals:
            return best
        for candidate in ranked_candidates[1:]:
            canonical = self._canonical_from_tokens(candidate.body_tokens)
            if canonical is not None and canonical not in pool_canonicals:
                return candidate
        return best

    def ranked_candidates(self, limit: int = 5):
        return self.beam_search.search()[:limit]

    def set_context(self, regime: str) -> None:
        self.current_regime = regime

    def apply_curriculum(self, stage) -> None:
        max_length = stage.max_length if self.max_length_cap is None else min(stage.max_length, self.max_length_cap)
        self.grammar.max_length = max_length
        self.grammar.min_length = stage.min_length
        self.sampler.config = replace(
            self.sampler.config,
            temperature=stage.temperature,
            top_k=stage.top_k,
            max_steps=max_length + 2,
        )
        self.beam_search.config = replace(
            self.beam_search.config,
            beam_width=stage.beam_width,
            per_node_top_k=stage.beam_top_k,
            max_steps=max_length + 2,
        )
        stage_for_agent = replace(stage, max_length=max_length)
        self.grammar_mcts.update_config(stage_for_agent)
        self.use_mcts = stage.use_mcts

    def score_valid_tokens(self):
        state = self.grammar.initial_state()
        valid_tokens = self.mask.filter_tokens(self.grammar.valid_next_tokens(state))
        scores = self.generator.score_tokens(state, valid_tokens)
        return self._adjust_scores_with_memory(state, scores, valid_tokens)

    def observe(self, tokens: tuple[str, ...], reward: float, accepted: bool) -> None:
        if hasattr(self.generator, "observe"):
            self.generator.observe(tokens, reward, accepted)

    def _adjust_scores_with_memory(self, state, scores: dict[str, float], valid_tokens: tuple[str, ...]) -> dict[str, float]:
        del state
        if self.experience_memory is None:
            retrieved_biases = {}
        else:
            retrieved = self.experience_memory.retrieve(self.current_regime, self.role, valid_tokens)
            retrieved_biases = retrieved.token_biases
        return {
            token: (
                float(scores.get(token, 0.0))
                + float(retrieved_biases.get(token, 0.0))
                + float(self.token_score_biases.get(token, 0.0))
            )
            for token in valid_tokens
        }

    def _evaluate_candidate(
        self,
        tokens,
        data,
        target,
        pool,
        reward_shaper,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> SearchEvaluation:
        effective_pool = pool
        outcome = reward_shaper.shape(tokens, data, target, pool, commit=False, role=self.role)
        override_pool = self._override_pool(pool)
        if override_pool is not None:
            override_outcome = reward_shaper.shape(tokens, data, target, override_pool, commit=False, role=self.role)
            if self._prefer_override_outcome(outcome, override_outcome):
                outcome = override_outcome
                effective_pool = override_pool
        if outcome.decision.candidate is None:
            return SearchEvaluation(
                score=-1.0,
                accepted=False,
                reason=outcome.decision.reason,
            )
        profile = resolve_role_profile(self.role)
        metrics = outcome.decision.candidate.metrics
        train_rank_ic = abs(float(metrics.get("rank_ic", 0.0)))
        train_rank_icir = abs(float(metrics.get("rank_icir", 0.0))) if np.isfinite(float(metrics.get("rank_icir", 0.0))) else 0.0
        max_corr = float(metrics.get("max_corr", 0.0))
        turnover = float(metrics.get("turnover", 0.0))
        complexity_penalty = profile.search_complexity_penalty_scale * len(tokens)
        operator_bonus = self._role_operator_adjustment(tokens)
        search_score = (
            3.0 * float(outcome.decision.marginal_gain)
            + 2.0 * float(outcome.decision.trade_proxy_gain)
            + 4.0 * train_rank_ic
            + 0.35 * train_rank_icir
            + (0.50 if outcome.decision.accepted else 0.0)
            - 0.10 * max_corr
            - 0.02 * turnover
            - complexity_penalty
            + operator_bonus
        )
        accepted = outcome.decision.accepted
        reason = outcome.decision.reason
        if validation_data is not None and validation_target is not None:
            preview = preview_candidate_on_dataset(
                tokens,
                effective_pool,
                validation_data,
                validation_target,
                evaluator=self.evaluator,
                role=self.role,
                min_abs_rank_ic=profile.resolved_preview_min_abs_rank_ic,
                max_correlation=profile.resolved_preview_max_correlation,
                replacement_margin=profile.replacement_margin,
                min_validation_marginal_gain=profile.preview_min_validation_marginal_gain,
            )
            if preview.record is None:
                return SearchEvaluation(
                    score=float(search_score - 0.5),
                    accepted=False,
                    reason=preview.reason,
                )
            valid_metrics = preview.record.metrics
            valid_rank_ic = abs(float(valid_metrics.get("rank_ic", 0.0)))
            valid_rank_icir = abs(float(valid_metrics.get("rank_icir", 0.0))) if np.isfinite(float(valid_metrics.get("rank_icir", 0.0))) else 0.0
            valid_turnover = float(valid_metrics.get("turnover", 0.0))
            valid_max_corr = float(valid_metrics.get("max_corr", 0.0))
            valid_drawdown = abs(min(0.0, float(valid_metrics.get("max_drawdown", 0.0))))
            train_signed_rank_ic = float(metrics.get("rank_ic", 0.0))
            valid_signed_rank_ic = float(valid_metrics.get("rank_ic", 0.0))
            sign_agreement = 0.20 if train_signed_rank_ic * valid_signed_rank_ic > 0 else -0.20
            stability_penalty = abs(train_rank_ic - valid_rank_ic)
            search_score += (
                2.5 * float(preview.marginal_gain)
                + 2.0 * float(preview.trade_proxy_gain)
                + 2.5 * valid_rank_ic
                + 0.25 * valid_rank_icir
                + sign_agreement
                - 0.05 * valid_max_corr
                - 0.015 * valid_turnover
                - 0.03 * valid_drawdown
                - stability_penalty
            )
            normalized_role = normalize_role(self.role)
            if normalized_role == "target_flow":
                search_score += (
                    1.5 * float(preview.trade_proxy_gain)
                    +
                    0.30 * max(0.0, 0.10 - stability_penalty)
                    - 0.03 * valid_turnover
                    - 0.05 * valid_drawdown
                )
            if normalized_role == "context":
                mul_div_penalty = 0.18 * sum(token in {"MUL", "DIV"} for token in tokens)
                search_score -= mul_div_penalty
                accepted = preview.accepted
                reason = preview.reason if not accepted else reason
            else:
                accepted = preview.accepted
                reason = preview.reason if not accepted else reason
        return SearchEvaluation(
            score=float(search_score),
            accepted=accepted,
            reason=reason,
        )

    def _override_pool(self, pool):
        normalized_role = normalize_role(self.role)
        if normalized_role == "target_flow":
            target_role = "target_price"
        elif normalized_role == "target_price":
            target_role = "target_flow"
        else:
            target_role = None
        if not any(normalize_role(record.role) == target_role for record in pool.records):
            if normalized_role != "target_price":
                return None
            same_role_indices = [
                index
                for index, record in enumerate(pool.records)
                if normalize_role(record.role) == normalized_role
            ]
            if not same_role_indices:
                return None
            weakest_index = min(
                same_role_indices,
                key=lambda index: float(pool.records[index].metrics.get("rank_ic", 0.0)),
            )
            override_pool = pool.copy()
            override_pool.records = [
                record
                for index, record in enumerate(override_pool.records)
                if index != weakest_index
            ]
            return override_pool
        override_pool = pool.copy()
        override_pool.records = [
            record
            for record in override_pool.records
            if normalize_role(record.role) != target_role
        ]
        if len(override_pool.records) == len(pool.records):
            return None
        return override_pool

    def _prefer_override_outcome(self, base_outcome, override_outcome) -> bool:
        if override_outcome.decision.candidate is None:
            return False
        if base_outcome.decision.candidate is None:
            return override_outcome.decision.accepted
        base_trade = float(base_outcome.decision.trade_proxy_gain)
        override_trade = float(override_outcome.decision.trade_proxy_gain)
        if override_outcome.decision.accepted and not base_outcome.decision.accepted:
            return True
        if override_outcome.decision.accepted and override_trade > base_trade + 5e-4:
            return True
        if not base_outcome.decision.accepted and override_trade > max(5e-4, base_trade):
            return True
        return False

    def _role_operator_adjustment(self, tokens: tuple[str, ...]) -> float:
        normalized_role = normalize_role(self.role)
        if normalized_role == "target_flow":
            bonus_tokens = {"TS_MEAN_5", "TS_STD_5", "DELTA_1", "CORR_5"}
            if self.role == "target_flow_gap":
                bonus_tokens = {"DELTA_1", "NEG", "RANK", "CORR_5", "SUB"}
            return 0.08 * sum(token in bonus_tokens for token in tokens)
        if normalized_role == "context":
            return (
                0.05 * sum(token in {"DELAY_1", "RANK", "CORR_5"} for token in tokens)
                - 0.10 * sum(token in {"MUL", "DIV"} for token in tokens)
            )
        return 0.0
