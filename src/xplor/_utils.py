import ast
import re

import gurobipy as gp
import polars as pl


def add_constrs_from_dataframe_args(
    df: pl.DataFrame,
    model: gp.Model,
    lhs: float | pl.Expr | pl.Series,
    sense: str,
    rhs: float | pl.Expr | pl.Series,
    name: str | None,
) -> list[gp.QConstr | gp.Constr]:
    rows = df.select(lhs=lhs, rhs=rhs).rows()
    first_lhs = rows[0][0]
    first_rhs = rows[0][1]

    if isinstance(first_rhs, gp.GenExprOr | gp.GenExprAbs):
        _add_constr = model.addConstr
    elif isinstance(first_lhs, gp.QuadExpr | gp.QuadExpr):
        _add_constr = model.addQConstr
    else:
        _add_constr = model.addLConstr

    if sense in ("<=", "<"):
        operator = "__le__"
    elif sense in (">=", ">"):
        operator = "__ge__"
    elif sense in ("==", "="):
        operator = "__eq__"
    else:
        msg = f"sense should be one of ('<=', '>=', '=='), got {sense}"
        raise Exception(msg)

    name = name or ""
    constrs = [
        _add_constr(
            getattr(lhs, operator)(rhs),
            name=name,
        )
        for lhs, rhs in rows
    ]

    return constrs


def evaluate_comp_expr(df: pl.DataFrame, expr: str) -> tuple[pl.Series, str, pl.Series]:
    # Just get the first character of sense, to match the gurobipy enums
    lhs, rhs = re.split("[<>=]+", expr)
    sense = expr.replace(lhs, "").replace(rhs, "")[0]

    lhsseries = evaluate_expr(df, lhs.strip())
    rhsseries = evaluate_expr(df, rhs.strip())
    return lhsseries, sense, rhsseries


def evaluate_expr(df: pl.DataFrame, expr: str) -> pl.Series:
    if expr in df:
        return df[expr]
    else:
        tree = ast.parse(expr, mode="eval")
        vars = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in df.columns
        }
        if vars.intersection(df.select(pl.col(pl.Object)).columns):
            return pl.Series(
                [eval(expr, None, r) for r in df.select(vars).rows(named=True)],
                dtype=pl.Object,
            )
        else:
            return df.with_columns(__xplor_tmp__=eval(expr))["__xplor_tmp__"]
