from collections.abc import Callable, Sequence
from typing import Any

import polars as pl


def parse_into_expr(value: float | str | pl.Expr | None) -> pl.Expr:
    if isinstance(value, float | int):
        return pl.lit(value, dtype=pl.Float64)
    elif value is None:
        return pl.lit(value, dtype=pl.Null)
    elif isinstance(value, str):
        return pl.col(value)
    return value


def map_rows(df: pl.DataFrame, f: Callable[[tuple], Any]) -> pl.Series:
    """Apply a custom/user-defined function (UDF) over the rows of the DataFrame.

    This function is the equivalent of `pl.Dataframe.map_rows`, but works with `pl.Object`.
    See: https://github.com/pola-rs/polars/issues/25570
    """
    return pl.Series([f(r) for r in df.rows()], dtype=pl.Object)


def series_to_df(series: Sequence[pl.Series]) -> pl.DataFrame:
    """Broadcast a list of series to the same height and return a DataFrame."""
    max_length = next((s.len() for s in series if s.len() > 1), len(series[0]))
    return pl.DataFrame(
        [
            pl.repeat(s[0], max_length, eager=True, dtype=s.dtype).alias(s.name)
            if s.len() != max_length
            else s
            for s in series
        ]
    )
