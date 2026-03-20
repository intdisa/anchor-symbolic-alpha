from knowledge_guided_symbolic_alpha.agents import MacroAgent
from knowledge_guided_symbolic_alpha.training import FormulaCurriculum


def test_formula_curriculum_progresses_complexity_and_search_budget() -> None:
    curriculum = FormulaCurriculum()
    early = curriculum.stage_for_episode(0, 100)
    late = curriculum.stage_for_episode(99, 100)

    assert early.max_length < late.max_length
    assert early.mcts_simulations < late.mcts_simulations
    assert not early.use_mcts
    assert late.use_mcts


def test_agent_apply_curriculum_updates_search_components() -> None:
    curriculum = FormulaCurriculum()
    agent = MacroAgent()
    stage = curriculum.stage_for_episode(99, 100)
    agent.apply_curriculum(stage)

    assert agent.grammar.max_length == stage.max_length
    assert agent.sampler.config.top_k == stage.top_k
    assert agent.beam_search.config.beam_width == stage.beam_width
    assert agent.grammar_mcts.config.simulations == stage.mcts_simulations
    assert agent.use_mcts
