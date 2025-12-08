from typing import Any

import polars as pl
import pytest

from xplor._utils import parse_into_expr


@pytest.mark.parametrize(
    ("value", "expected_dtype"),
    [
        (123.45, pl.Float64),
        (42, pl.Float64),
        ("a", None),
        (None, pl.Null),
        (pl.col.a.alias("test"), None),
    ],
)
def test_to_expr_conversion(value: Any, expected_dtype: pl.DataType | None):
    """
    Tests that various inputs are correctly converted into the expected
    Polars Expression type, value, and dtype.
    """
    result_expr = parse_into_expr(value)
    assert isinstance(result_expr, pl.Expr)

    if expected_dtype is not None:
        assert pl.select(result_expr).dtypes[0] == expected_dtype


def test_to_expr_conversion_raise() -> None:
    expected_message = "Impossible to convert to expression."

    with pytest.raises(Exception, match=expected_message):
        # The invalid code that is expected to fail
        parse_into_expr(object())  # ty:ignore[invalid-argument-type]
