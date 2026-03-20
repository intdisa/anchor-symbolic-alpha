from knowledge_guided_symbolic_alpha.search import ProposalCandidate, ProposalMixer, SearchEvaluation


def test_proposal_mixer_dedupes_and_prefers_evaluated_best_candidate() -> None:
    mixer = ProposalMixer()
    candidates = [
        ProposalCandidate(source="sample", body_tokens=("GOLD_CLOSE", "VIX", "MUL"), score=0.1, valid=True),
        ProposalCandidate(source="beam", body_tokens=("VIX", "GOLD_CLOSE", "MUL"), score=0.2, valid=True),
        ProposalCandidate(source="mcts", body_tokens=("VIX", "DELTA_1", "NEG"), score=0.0, valid=True),
    ]

    def evaluate(tokens):
        if tokens == ("VIX", "DELTA_1", "NEG"):
            return SearchEvaluation(score=1.0, accepted=True, reason="best")
        return SearchEvaluation(score=0.2, accepted=False, reason="other")

    best = mixer.select_best(candidates, evaluator=evaluate)
    assert best is not None
    assert best.body_tokens == ("VIX", "DELTA_1", "NEG")
    assert best.accepted

