import polars as pl

from xplor.typing import VariableType

# Extension types for optimization variables
ContinuousVarExtension = pl.Extension(
    name="var.continuous",
    storage=pl.Object,
    metadata="Continuous optimization variable",
)

IntegerVarExtension = pl.Extension(
    name="var.integer",
    storage=pl.Object,
    metadata="Integer optimization variable",
)

BinaryVarExtension = pl.Extension(
    name="var.binary",
    storage=pl.Object,
    metadata="Binary optimization variable",
)

# Extension type for expressions (LinExpr, QuadExpr, etc.)
ExprExtension = pl.Extension(
    name="xplor.expr",
    storage=pl.Object,
    metadata="Optimization expression",
)


def get_var_extension(vtype: VariableType) -> pl.Extension:
    """Return the Extension type for a given variable type."""
    match vtype:
        case "INTEGER":
            return IntegerVarExtension
        case "BINARY":
            return BinaryVarExtension
        case _:
            return ContinuousVarExtension


def cast_to_dtypes(series: pl.Series, source_dtype: pl.DataType) -> pl.Series:
    """Cast a series to the corresponding data type based on the source Extension dtype.

    Uses the Extension type name to determine the appropriate output dtype:
    - var.integer → Int32
    - var.binary → Boolean
    - var.continuous / other → Float64

    Parameters
    ----------
    series : pl.Series
        The series containing extracted values to cast.
    source_dtype : pl.DataType
        The dtype of the original variable series (used to determine target type).

    """
    if isinstance(source_dtype, pl.Extension):
        ext_name = source_dtype.ext_name()
        if ext_name == "var.integer":
            return series.cast(pl.Int32)
        if ext_name == "var.binary":
            return series.cast(pl.Int8).cast(pl.Boolean)
    return series.cast(pl.Float64)
