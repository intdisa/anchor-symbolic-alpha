import pytest

from knowledge_guided_symbolic_alpha.language import ParseError, RPNParser


def test_parser_accepts_valid_formula() -> None:
    parser = RPNParser()
    parsed = parser.parse_text("GOLD_CLOSE DELAY_1 GOLD_CLOSE SUB")
    assert parsed.unit == "price"
    assert parsed.canonical_rpn == ("GOLD_CLOSE", "DELAY_1", "GOLD_CLOSE", "SUB")


def test_parser_rejects_special_token_in_body() -> None:
    parser = RPNParser()
    with pytest.raises(ParseError):
        parser.parse(["GOLD_CLOSE", "<EOS>", "VIX", "MUL"])


def test_parser_rejects_macro_without_delay() -> None:
    parser = RPNParser()
    with pytest.raises(ParseError):
        parser.parse_text("CPI TNX ADD")


def test_parser_rejects_incompatible_units() -> None:
    parser = RPNParser()
    with pytest.raises(ParseError):
        parser.parse_text("GOLD_CLOSE GOLD_VOLUME ADD")


def test_parser_canonicalizes_commutative_operands() -> None:
    parser = RPNParser()
    left = parser.parse_text("GOLD_CLOSE VIX MUL")
    right = parser.parse_text("VIX GOLD_CLOSE MUL")
    assert left.canonical == right.canonical
    assert left.canonical_rpn == right.canonical_rpn


def test_parser_rejects_incomplete_expression() -> None:
    parser = RPNParser()
    with pytest.raises(ParseError):
        parser.parse_text("GOLD_CLOSE VIX")
