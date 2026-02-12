import polars as pl

from xplor.typing import VariableType


def cast_to_dtypes(series: pl.Series, variable_type: VariableType | str) -> pl.Series:
    """Cast a series to the corresponding data type base on its variable_type.

    Note: Accepts str to handle internal markers like "_INTERVAL_CP" for CPLEX CP backend.
    """
    if variable_type == "INTEGER":
        return series.cast(pl.Int32)
    elif variable_type == "BINARY":
        return series.cast(pl.Int8).cast(pl.Boolean)
    elif variable_type == "_INTERVAL_CP":
        # Interval variables (CPLEX CP) are already returned as structs, no casting needed
        return series
    else:
        return series.cast(pl.Float64)
