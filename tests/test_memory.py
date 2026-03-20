from knowledge_guided_symbolic_alpha.agents import MacroAgent, ManagerAgent
from knowledge_guided_symbolic_alpha.memory import ExperienceMemory


def test_experience_memory_builds_positive_and_negative_token_biases() -> None:
    memory = ExperienceMemory(success_scale=0.5, failure_scale=0.5)
    memory.record("USD_STRENGTH", "macro", ("DXY", "DELAY_1", "ABS"), 0.8, True, "accepted")
    memory.record("USD_STRENGTH", "macro", ("CPI", "DELAY_1", "ADD"), -0.4, False, "fast_ic_screen")

    retrieved = memory.retrieve("USD_STRENGTH", "macro", ("DXY", "CPI", "ABS"))
    assert retrieved.token_biases["DXY"] > 0.0
    assert retrieved.token_biases["ABS"] > 0.0
    assert retrieved.token_biases["CPI"] < 0.0


def test_macro_agent_scores_shift_after_memory_feedback() -> None:
    memory = ExperienceMemory(success_scale=0.8, failure_scale=0.8)
    agent = MacroAgent(experience_memory=memory)
    agent.set_context("USD_STRENGTH")

    before = agent.score_valid_tokens()
    memory.record("USD_STRENGTH", "macro", ("DXY", "DELAY_1"), 1.0, True, "accepted")
    memory.record("USD_STRENGTH", "macro", ("CPI", "DELAY_1"), -1.0, False, "fast_ic_screen")
    after = agent.score_valid_tokens()

    assert after["DXY"] > before["DXY"]
    assert after["CPI"] < before["CPI"]


def test_manager_exposes_shared_memory() -> None:
    manager = ManagerAgent()
    assert manager.macro_agent.experience_memory is manager.experience_memory
    assert manager.micro_agent.experience_memory is manager.experience_memory
