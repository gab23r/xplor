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
    return expr.map_elements(lambda d: d[0], return_dtype=pl.Object)


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
    return expr.map_elements(lambda d: d[-1], return_dtype=pl.Object)


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
    return expr.map_elements(gp.quicksum, return_dtype=pl.Object)


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
    return expr.map_elements(lambda d: gp.or_(d.to_list()), return_dtype=pl.Object)


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
    >>> df.group_by('group').agg(abs('value'))

    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_elements(lambda d: gp.abs_(d.to_list()), return_dtype=pl.Object)


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
    >>> df.with_columns(read_value('x'))

    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_batches(
        lambda s:
        # in case of a variable
        pl.Series([e.x for e in s])
        if s.len() and hasattr(s[0], "X")
        # in case of a linExpr
        else pl.Series([e.getValue() for e in s])
    )


def add_vars(
    df: pl.DataFrame,
    model: gp.Model,
    name: str,
    *,
    lb: float | str | pl.Expr = 0.0,
    ub: float | str | pl.Expr = gp.GRB.INFINITY,
    obj: float | str | pl.Expr = 0.0,
    indices: list[str] | None = None,
    vtype: str = gp.GRB.CONTINUOUS,
) -> pl.DataFrame:
    """Add a variable to the given model for each row in the dataframe.

    Parameters
    ----------
    df: pl.DataFrame
        The dataframe that will hold the new variables
    model : Model
        A Gurobi model to which new variables will be added
    name : str
        Used as the appended column name, as well as the base
        name for added Gurobi variables
    lb : float | pl.Expr | None
        Lower bound for created variables. May be a single value
        or the name of a column in the dataframe, defaults to 0.0
    ub : float | pl.Expr | None
        Upper bound for created variables. May be a single value
        or the name of a column in the dataframe, defaults to
        :code:`GRB.INFINITY`
    obj: float | pl.Expr | None
        Objective function coefficient for created variables.
        May be a single value, or the name of a column in the dataframe,
        defaults to 0.0
    indices: list[str] | None
        Keys of the variables
    vtype: str | None
        Gurobi variable type for created variables, defaults
        to :code:`GRB.CONTINUOUS`

    Returns
    -------
    DataFrame
        A new DataFrame with new Vars appended as a column

    """
    lb_ = df.with_columns(lb=lb)["lb"].to_numpy() if isinstance(lb, str | pl.Expr) else lb
    ub_ = df.with_columns(ub=ub)["ub"].to_numpy() if isinstance(ub, str | pl.Expr) else ub
    obj_ = df.with_columns(obj=obj)["obj"].to_numpy() if isinstance(obj, str | pl.Expr) else obj
    if indices is not None:
        name_ = (
            df.select(
                pl.format(
                    "{}[{}]",
                    pl.lit(name),
                    pl.concat_str(indices, separator=","),
                )
            )
            .to_series(0)
            .to_list()
        )
    else:
        name_ = indices
    vars = model.addMVar(
        df.height,
        vtype=vtype,
        lb=lb_,
        ub=ub_,
        obj=obj_,
        name=name_,
    )
    # model.update()
    return df.with_columns(pl.Series(name, vars.tolist(), dtype=pl.Object))


def add_constrs(
    df: pl.DataFrame,
    model: gp.Model,
    expr: str,
    name: str | None = None,
) -> pl.DataFrame:
    """Create a Gurobi constraint for each row in the dataframe using a string expression.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing the data for creating constraints
    model : gp.Model
        A Gurobi model to which new constraints will be added
    expr : str
        A string expression representing the constraint. Must include a comparison
        operator ('<=', '==', or '>='). The expression can reference column names
        and use standard mathematical operators. For example: "2*x + y <= z"
    name : str | None
        Base name for the constraints. If provided, constraints will be added as
        a new column to the DataFrame with this name. If None, constraints are
        still added to the model but not returned in the DataFrame, by default None

    Returns
    -------
    pl.DataFrame
        If name is provided, returns DataFrame with new constraints appended as a column.
        If name is None, returns the original DataFrame unchanged.

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "x": [gp.Var()],
    ...     "y": [gp.Var()],
    ...     "z": [5]
    ... })
    >>> df = df.pipe(add_constrs, model, "2*x + y <= z", name="capacity")

    Notes
    -----
    - Expression can use any column name from the DataFrame
    - Supports arithmetic operations (+, -, *, /) and Gurobi functions
    - Empty DataFrames are returned unchanged
    - The model is not automatically updated after adding constraints

    See Also
    --------
    add_vars : Function to add variables to the model

    """
    if df.height == 0:
        return df

    lhs_, sense_, rhs_ = _utils.evaluate_comp_expr(df, expr)
    constrs = _utils.add_constrs_from_dataframe_args(df, model, lhs_, sense_, rhs_, name)

    # model.update()
    if name is None:
        return df
    else:
        return df.with_columns(pl.Series(name, constrs, dtype=pl.Object))
