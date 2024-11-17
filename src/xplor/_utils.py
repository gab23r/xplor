import ast
import re

import polars as pl


def format_indices(df: pl.DataFrame, name: str, indices: list[str]) -> list[str]:
    return (
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
