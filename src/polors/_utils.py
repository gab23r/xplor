import ast
import re

import gurobipy as gp
import polars as pl

CONSTRAINT_SENSES = frozenset((gp.GRB.LESS_EQUAL, gp.GRB.EQUAL, gp.GRB.GREATER_EQUAL))


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

    if isinstance(first_rhs, gp.GenExprOr):
        _add_constr = model.addConstr
    elif isinstance(first_lhs, gp.QuadExpr) or isinstance(first_rhs, gp.QuadExpr):
        _add_constr = model.addQConstr
    else:
        _add_constr = model.addLConstr

    if sense == gp.GRB.EQUAL:
        operator = "__eq__"
    elif sense == gp.GRB.LESS_EQUAL:
        operator = "__le__"
    elif sense == gp.GRB.GREATER_EQUAL:
        operator = "__ge__"
    else:
        raise Exception(f"operator should be one of {CONSTRAINT_SENSES}, got {sense}")

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
        if not vars:
            return df.with_columns(__polors_tmp__=eval(expr))["__polors_tmp__"]
        else:
            return pl.Series(
                [eval(expr, None, r) for r in df.select(vars).rows(named=True)],
                dtype=pl.Object,
            )
