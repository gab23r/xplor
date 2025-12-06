from enum import StrEnum

import polars as pl


class VarType(StrEnum):
    """The type of the variable.

    Attributes
    ----------
    CONTINUOUS:
        A real-valued variable that can take any value between its lower and upper bounds.
    INTEGER:
        A discrete variable that can only take whole number values between its bounds (e.g., 0, 1, 2, ...).
    BINARY:
        A specialized integer variable restricted to only two values: 0 or 1.

    """

    CONTINUOUS = "CONTINUOUS"
    """A real-valued variable that can take any value between its lower and upper bounds."""

    INTEGER = "INTEGER"
    """A discrete variable that can only take whole number values between its bounds (e.g., 1, 2, 3, ...)."""

    BINARY = "BINARY"
    """A specialized integer variable restricted to only two values: 0 or 1."""


def cast_to_dtypes(series: pl.Series, vartype: VarType) -> pl.Series:
    """Cast a series to the corresponding data type base on its vartype."""
    if vartype == VarType.CONTINUOUS:
        return series.cast(pl.Float64)
    elif vartype == VarType.BINARY:
        return series.cast(pl.Int8).cast(pl.Boolean)
    else:
        return series.cast(pl.Int32)
