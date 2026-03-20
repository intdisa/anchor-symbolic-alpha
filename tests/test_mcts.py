from knowledge_guided_symbolic_alpha.envs import TreeStructuredLanguageMDP
from knowledge_guided_symbolic_alpha.language import RPNGrammar
from knowledge_guided_symbolic_alpha.models.generator import TreePolicy, TreeValue
from knowledge_guided_symbolic_alpha.search import GrammarMCTS, GrammarMCTSConfig, SearchEvaluation


class DummyGenerator:
    def score_tokens(self, state, valid_tokens: tuple[str, ...]) -> dict[str, float]:
        scores = {token: 0.0 for token in valid_tokens}
        prefix = state.body_tokens
        if prefix == ():
            scores["GOLD_CLOSE"] = 3.0
        elif prefix == ("GOLD_CLOSE",):
            scores["VIX"] = 3.0
        elif prefix == ("GOLD_CLOSE", "VIX"):
            scores["MUL"] = 3.0
        elif prefix == ("GOLD_CLOSE", "VIX", "MUL"):
            scores["<EOS>"] = 3.0
        return scores

    def observe(self, tokens, reward, accepted) -> None:
        del tokens, reward, accepted


def test_grammar_mcts_finds_high_reward_formula() -> None:
    grammar = RPNGrammar(max_length=3, min_length=3)
    mdp = TreeStructuredLanguageMDP(grammar)
    mcts = GrammarMCTS(
        mdp,
        DummyGenerator(),
        TreePolicy(max_length=grammar.max_length),
        TreeValue(max_length=grammar.max_length),
        config=GrammarMCTSConfig(simulations=12, top_k_expansion=4, rollout_depth=3),
    )

    def evaluate(tokens: tuple[str, ...]) -> SearchEvaluation:
        if tokens == ("GOLD_CLOSE", "VIX", "MUL"):
            return SearchEvaluation(score=1.0, accepted=True, reason="target")
        return SearchEvaluation(score=-0.5, accepted=False, reason="other")

    candidate = mcts.search(evaluate)
    assert candidate.valid
    assert candidate.body_tokens == ("GOLD_CLOSE", "VIX", "MUL")
    assert candidate.accepted
