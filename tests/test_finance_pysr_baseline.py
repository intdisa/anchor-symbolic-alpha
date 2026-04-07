from __future__ import annotations

import sympy as sp

from scripts.run_finance_pysr_baseline import translate_equation


def test_translate_equation_maps_simple_sympy_expression_to_rpn() -> None:
    variable_map = {
        "R_CASH_RATIO_Q": ("CASH_RATIO_Q", "RANK"),
        "R_PROFITABILITY_Q": ("PROFITABILITY_Q", "RANK"),
    }
    expr = sp.Symbol("R_CASH_RATIO_Q") + sp.Symbol("R_PROFITABILITY_Q")

    formula = translate_equation(expr, variable_map)

    assert formula == "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"
