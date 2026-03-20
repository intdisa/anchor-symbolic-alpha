from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import sqrt
from typing import Callable

import numpy as np

from ..envs.tsl_mdp import TreeStructuredLanguageMDP


@dataclass(frozen=True)
class SearchEvaluation:
    score: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class GrammarMCTSConfig:
    simulations: int = 16
    top_k_expansion: int = 4
    rollout_depth: int = 5
    exploration_constant: float = 1.25


@dataclass(frozen=True)
class MCTSCandidate:
    body_tokens: tuple[str, ...]
    score: float
    visits: int
    valid: bool
    terminal_error: str | None
    accepted: bool
    reason: str


@dataclass
class _Node:
    state: object
    visits: int = 0
    value_sum: float = 0.0
    priors: dict[str, float] = field(default_factory=dict)
    children: dict[str, "_Node"] = field(default_factory=dict)
    ranked_actions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class GrammarMCTS:
    def __init__(
        self,
        mdp: TreeStructuredLanguageMDP,
        generator,
        tree_policy,
        tree_value,
        config: GrammarMCTSConfig | None = None,
        score_adjuster: Callable[[object, dict[str, float], tuple[str, ...]], dict[str, float]] | None = None,
    ) -> None:
        self.mdp = mdp
        self.generator = generator
        self.tree_policy = tree_policy
        self.tree_value = tree_value
        self.config = config or GrammarMCTSConfig()
        self.score_adjuster = score_adjuster

    def search(
        self,
        evaluate_formula: Callable[[tuple[str, ...]], SearchEvaluation],
    ) -> MCTSCandidate:
        root = _Node(state=self.mdp.reset())
        best_candidate: MCTSCandidate | None = None

        for _ in range(self.config.simulations):
            node = root
            path_nodes: list[_Node] = [root]

            while True:
                if self.mdp.is_terminal(node.state):
                    value, candidate = self._terminal_value(node.state, evaluate_formula)
                    best_candidate = self._better_candidate(best_candidate, candidate)
                    self._backpropagate(path_nodes, value)
                    break

                self._ensure_priors(node)
                action = self._select_action(node)
                if action not in node.children:
                    transition = self.mdp.step(node.state, action)
                    child = _Node(state=transition.next_state)
                    node.children[action] = child
                    path_nodes.append(child)
                    value, candidate = self._expand_and_evaluate(child, evaluate_formula)
                    best_candidate = self._better_candidate(best_candidate, candidate)
                    self.tree_policy.observe(node.state, action, value)
                    self._backpropagate(path_nodes, value)
                    break

                node = node.children[action]
                path_nodes.append(node)

        if best_candidate is not None:
            return best_candidate
        root_candidate = MCTSCandidate(
            body_tokens=root.state.body_tokens,
            score=-1.0,
            visits=root.visits,
            valid=False,
            terminal_error=getattr(root.state, "terminal_error", None),
            accepted=False,
            reason="no_valid_candidate",
        )
        return root_candidate

    def update_config(self, stage) -> None:
        self.config = replace(
            self.config,
            simulations=stage.mcts_simulations,
            top_k_expansion=stage.mcts_top_k,
            rollout_depth=stage.rollout_depth,
            exploration_constant=stage.exploration_constant,
        )
        self.tree_policy.set_max_length(stage.max_length)
        self.tree_value.set_max_length(stage.max_length)

    def _expand_and_evaluate(
        self,
        node: _Node,
        evaluate_formula: Callable[[tuple[str, ...]], SearchEvaluation],
    ) -> tuple[float, MCTSCandidate | None]:
        if self.mdp.is_terminal(node.state):
            return self._terminal_value(node.state, evaluate_formula)
        rollout_state = node.state
        for _ in range(self.config.rollout_depth):
            if self.mdp.is_terminal(rollout_state):
                break
            valid_actions = self.mdp.valid_actions(rollout_state)
            scores = self._action_scores(rollout_state, valid_actions)
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            rollout_state = self.mdp.step(rollout_state, ranked[0][0]).next_state
        if self.mdp.is_terminal(rollout_state):
            value, candidate = self._terminal_value(rollout_state, evaluate_formula)
            valid_action_count = len(self.mdp.valid_actions(node.state))
            self.tree_value.observe(node.state, value, valid_action_count)
            return value, candidate
        valid_action_count = len(self.mdp.valid_actions(node.state))
        value = self.tree_value.evaluate_state(node.state, valid_action_count)
        self.tree_value.observe(node.state, value, valid_action_count)
        return value, None

    def _terminal_value(
        self,
        state,
        evaluate_formula: Callable[[tuple[str, ...]], SearchEvaluation],
    ) -> tuple[float, MCTSCandidate]:
        if state.terminal_error is not None:
            return -1.0, MCTSCandidate(
                body_tokens=state.body_tokens,
                score=-1.0,
                visits=1,
                valid=False,
                terminal_error=state.terminal_error,
                accepted=False,
                reason=state.terminal_error,
            )
        evaluation = evaluate_formula(self.mdp.terminal_tokens(state))
        candidate = MCTSCandidate(
            body_tokens=state.body_tokens,
            score=float(evaluation.score),
            visits=1,
            valid=True,
            terminal_error=None,
            accepted=evaluation.accepted,
            reason=evaluation.reason,
        )
        return float(evaluation.score), candidate

    def _select_action(self, node: _Node) -> str:
        unexpanded = [action for action in node.ranked_actions if action not in node.children]
        if unexpanded:
            return unexpanded[0]
        total_visits = max(node.visits, 1)
        best_action = node.ranked_actions[0]
        best_score = float("-inf")
        for action in node.ranked_actions:
            child = node.children[action]
            q = child.mean_value
            u = self.config.exploration_constant * node.priors[action] * sqrt(total_visits) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_action = action
                best_score = score
        return best_action

    def _ensure_priors(self, node: _Node) -> None:
        if node.priors:
            return
        valid_actions = self.mdp.valid_actions(node.state)
        scores = self._action_scores(node.state, valid_actions)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.config.top_k_expansion]
        if not ranked:
            ranked = [("<EOS>", 0.0)]
        node.ranked_actions = tuple(action for action, _ in ranked)
        logits = np.array([score for _, score in ranked], dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        node.priors = {
            action: float(probability)
            for (action, _), probability in zip(ranked, probs.tolist())
        }

    def _action_scores(self, state, valid_actions: tuple[str, ...]) -> dict[str, float]:
        scores = self.generator.score_tokens(state, valid_actions)
        policy_scores = self.tree_policy.score_actions(state, valid_actions)
        blended = {
            action: float(scores.get(action, 0.0)) + float(policy_scores.get(action, 0.0))
            for action in valid_actions
        }
        if self.score_adjuster is not None:
            blended = self.score_adjuster(state, blended, valid_actions)
        return blended

    def _backpropagate(self, path_nodes: list[_Node], value: float) -> None:
        for node in path_nodes:
            node.visits += 1
            node.value_sum += value

    def _better_candidate(
        self,
        current: MCTSCandidate | None,
        candidate: MCTSCandidate | None,
    ) -> MCTSCandidate | None:
        if candidate is None:
            return current
        if current is None:
            return candidate
        return candidate if candidate.score > current.score else current
