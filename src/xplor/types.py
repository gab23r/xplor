import polars as pl

from xplor.typing import VariableType


def cast_to_dtypes(series: pl.Series, variable_type: VariableType) -> pl.Series:
    """Cast a series to the corresponding data type base on its variable_type."""
    if variable_type == "INTEGER":
        return series.cast(pl.Int32)
    elif variable_type == "BINARY":
        return series.cast(pl.Int8).cast(pl.Boolean)
    else:
        return series.cast(pl.Float64)
