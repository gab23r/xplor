import polars as pl
import pytest

import xplor


def test_str() -> None:
    assert str((xplor.var.x + pl.col("ub")) == 1) == "((x + ub) == 1)"
    assert str(xplor.var.x.sum() + pl.col("ub").first()) == '(x.sum() + col("ub").first())'
    assert str((xplor.var.x + 1 + pl.col("ub")).sum()) == "((x + 1) + ub).sum()"


def test_invalid_dataframe_constraint_raises_exception() -> None:
    # Define the specific error message you expect
    expected_message = "Temporary constraints are not valid expression."

    # Use pytest.raises to assert that the code block raises the specified Exception
    # and also check if the error message contains the expected text.
    with pytest.raises(Exception, match=expected_message):
        # The invalid code that is expected to fail
        pl.DataFrame().with_columns(xplor.var.x == 1)
