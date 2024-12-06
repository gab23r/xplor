import re

import gurobipy as gp
import polars as pl

from xplor import _utils


def apply_eval(df: pl.DataFrame, expr: str) -> pl.DataFrame:
    """Evaluate a string expression and add the result as a new column to the DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame
    expr : str
        Expression to evaluate. Can be in the form 'new_col = expression' or just 'expression'.
        If an alias is provided (using =), the result will be named accordingly.

    Returns
    -------
    pl.DataFrame
        DataFrame with the evaluated expression added as a new column

    Examples
    --------
    >>> df.pipe(apply_eval, "y = 2 * x - c")

    """
    *alias, expr = re.split("=", expr)

    series = _utils.evaluate_expr(df, expr.strip())
    if alias:
        series = series.alias(alias[0].strip())

    return df.with_columns(series)


def first(expr: pl.Expr | str) -> pl.Expr:
    """Return the first element of each group in a polars expression.

    Parameters
    ----------
    expr : pl.Expr | str
        Column name or polars expression to get first element from

    Returns
    -------
    pl.Expr
        Expression that will return the first element of each group

    Examples
    --------
    >>> df.group_by('group').agg(first('value'))

    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_batches(lambda d: d[0], return_dtype=pl.Object, returns_scalar=True)


def last(expr: pl.Expr | str) -> pl.Expr:
    """Return the last element of each group in a polars expression.

    Parameters
    ----------
    expr : pl.Expr | str
        Column name or polars expression to get last element from

    Returns
    -------
    pl.Expr
        Expression that will return the last element of each group

    Examples
    --------
    >>> df.group_by('group').agg(last('value'))

    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_batches(lambda d: d[-1], return_dtype=pl.Object, returns_scalar=True)


def quicksum(expr: pl.Expr | str) -> pl.Expr:
    """Apply Gurobi's quicksum to elements in each group.

    Parameters
    ----------
    expr : pl.Expr | str
        Column name or polars expression containing Gurobi variables or expressions to sum

    Returns
    -------
    pl.Expr
        Expression that will return the Gurobi quicksum of elements in each group

    Examples
    --------
    >>> df.group_by('group').agg(quicksum('x'))

    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_batches(gp.quicksum, return_dtype=pl.Object, returns_scalar=True)


def any(expr: pl.Expr | str) -> pl.Expr:
    """Create a Gurobi OR constraint from elements in each group.

    Parameters
    ----------
    expr : pl.Expr | str
        Column name or polars expression containing Gurobi variables or expressions

    Returns
    -------
    pl.Expr
        Expression that will return the Gurobi OR of elements in each group

    Examples
    --------
    >>> df.group_by('group').agg(any('binary_var'))

    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_batches(
        lambda d: gp.or_(d.to_list()), return_dtype=pl.Object, returns_scalar=True
    )


def abs(expr: pl.Expr | str) -> pl.Expr:
    """Apply Gurobi's absolute value function to elements in each group.

    Parameters
    ----------
    expr : pl.Expr | str
        Column name or polars expression containing Gurobi variables or expressions

    Returns
    -------
    pl.Expr
        Expression that will return the absolute value of elements in each group

    Examples
    --------
    >>> df.with_columns(x_abs = abs('x'))

    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_elements(lambda d: gp.abs_(d), return_dtype=pl.Object)


def read_value(expr: pl.Expr | str) -> pl.Expr:
    """Extract the optimal value from Gurobi variables or expressions after optimization.

    Parameters
    ----------
    expr : pl.Expr | str
        Column name or polars expression containing Gurobi variables or linear expressions

    Returns
    -------
    pl.Expr
        Expression that will return the optimal values after model solving.
        For variables, returns X attribute value.
        For linear expressions, returns the evaluated value.

    Examples
    --------
    >>> df.with_columns(read_value(pl.selectors.object()))

    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_batches(
        lambda s:
        # in case of a variable
        pl.Series([e.x for e in s])
        if s.len() and hasattr(s[0], "X")
        # in case of a linExpr
        else pl.Series([e.getValue() for e in s]),
        return_dtype=pl.Float64,
    )
