from knowledge_guided_symbolic_alpha.language import EOS_TOKEN, RPNGrammar


def test_first_step_only_allows_body_tokens() -> None:
    grammar = RPNGrammar()
    valid = set(grammar.valid_next_tokens(grammar.initial_state()))
    assert "GOLD_CLOSE" in valid
    assert "CRUDE_OIL_CLOSE" in valid
    assert "SP500_REALIZED_VOL_20" in valid
    assert "CPI" in valid
    assert "ADD" not in valid
    assert EOS_TOKEN not in valid


def test_macro_feature_requires_delay_before_binary_use() -> None:
    grammar = RPNGrammar()
    state = grammar.step(grammar.initial_state(), "CPI")
    valid = set(grammar.valid_next_tokens(state))
    assert "DELAY_1" in valid
    assert "ADD" not in valid
    assert "SUB" not in valid


def test_complete_expression_allows_eos() -> None:
    grammar = RPNGrammar()
    state = grammar.initial_state()
    for token in ("GOLD_CLOSE", "VIX", "MUL"):
        state = grammar.step(state, token)
    valid = set(grammar.valid_next_tokens(state))
    assert EOS_TOKEN in valid


def test_empty_mask_falls_back_to_eos() -> None:
    grammar = RPNGrammar(max_length=2, min_length=2)
    state = grammar.initial_state()
    for token in ("CPI", "GOLD_CLOSE"):
        state = grammar.step(state, token)
    assert grammar.valid_next_tokens(state) == (EOS_TOKEN,)


def test_forced_eos_marks_invalid_terminal() -> None:
    grammar = RPNGrammar(max_length=2, min_length=2)
    state = grammar.initial_state()
    for token in ("CPI", "GOLD_CLOSE"):
        state = grammar.step(state, token)
    state = grammar.step(state, EOS_TOKEN)
    assert state.finished
    assert state.terminal_error == "forced_eos"
    assert not grammar.is_valid_terminal(state)
