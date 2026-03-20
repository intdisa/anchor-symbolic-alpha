import pandas as pd

from knowledge_guided_symbolic_alpha.evaluation.risk_metrics import annual_return


def test_annual_return_handles_nonpositive_gross_returns() -> None:
    returns = pd.Series([0.1, -1.2, 0.05])
    value = annual_return(returns, periods_per_year=3)
    assert value == returns.mean() * 3
