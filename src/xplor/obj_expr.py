from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from xplor._utils import map_rows, series_to_df

OPERATOR_MAP = {
    "__add__": "+",
    "__radd__": "+",
    "__mul__": "*",
    "__rmul__": "*",
    "__truediv__": "/",
    "__rtruediv__": "/",
    "__eq__": "==",
    "__ge__": ">=",
    "__le__": "<=",
}


@dataclass
class ObjExprNode:
    """Represents a single operation (operator) and its value (operand)."""

    operator: str  #  '__add__', '__rtruediv__'
    operand: Any  # 1, 'a', or ObjExpr('b')


class ObjExpr(pl.Expr):
    """Custom Polars Expression wrapper designed for building composite expressions
    that mix standard pl.Expr and custom variables.

    ObjExpr constructs an internal Abstract Syntax Tree (AST) of operations, which
    is then transformed into an efficient pl.map_batches call during execution.
    This allows complex, multi-variable logic to run within Polars' optimized
    batch context.

    The final expression uses integer indexing (d[0], d[1], etc.) where:
    - Indices 0 to N-1 map to the unique ObjExpr names (from `self._obj_names`).
    - Indices N onwards map to embedded Polars expressions (pl.Expr).

    Attributes:
        _expr (pl.Expr): The root expression.
        _nodes (list[ObjExprNode]): The internal list representing the operation AST.

    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr: pl.Expr = expr
        self._nodes: list[ObjExprNode] = []

    def _repr_html_(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        expr_str, exprs = self.process_expression()
        return self._get_repr(expr_str, exprs)

    @property
    def _pyexpr(self):  # type: ignore # noqa: ANN202
        if not self._nodes:
            return self._expr._pyexpr
        expr_str, exprs = self.process_expression()
        return pl.map_batches(
            exprs,
            lambda s: map_rows(series_to_df(s, rename_series=True), eval(f"lambda d: {expr_str}")),
            return_dtype=pl.Object,
        )._pyexpr

    def _append_node(self, operator: str, operand: Any) -> ObjExpr:
        """Append a node and return the current instance for chaining."""
        self._nodes.append(ObjExprNode(operator, operand))
        return self

    def __add__(self, other: Any) -> ObjExpr:
        return self._append_node("__add__", other)

    def __radd__(self, other: Any) -> ObjExpr:
        return self._append_node("__radd__", other)

    def __truediv__(self, other: Any) -> ObjExpr:
        return self._append_node("__truediv__", other)

    def __rtruediv__(self, other: Any) -> ObjExpr:
        return self._append_node("__rtruediv__", other)

    def __mul__(self, other: Any) -> ObjExpr:
        return self._append_node("__mul__", other)

    def __rmul__(self, other: Any) -> ObjExpr:
        return self._append_node("__rmul__", other)

    def __eq__(self, other: object) -> ObjExpr:  # type: ignore[override]
        return self._append_node("__eq__", other)

    def __le__(self, other: object) -> ObjExpr:
        return self._append_node("__le__", other)

    def __ge__(self, other: object) -> ObjExpr:
        return self._append_node("__ge__", other)

    def process_expression(self) -> tuple[str, list[pl.Expr]]:
        """Transform a composite object expression into a list of Polars sub-expressions
        and an equivalent lambda function, using integer indexing for all inputs.
        """
        exprs: list[pl.Expr] = [self._expr]
        expr_str = "d[0]"

        for node in self._nodes:
            if isinstance(node.operand, pl.Expr):
                exprs.append(node.operand)
                operand_str = f"d[{len(exprs) - 1}]"
            else:
                operand_str = node.operand

            # Sequential building with parentheses to maintain precedence based on chain order
            if node.operator.startswith("__r"):
                expr_str = f"({operand_str} {OPERATOR_MAP[node.operator]} {expr_str})"
            else:
                expr_str = f"({expr_str} {OPERATOR_MAP[node.operator]} {operand_str})"

        return expr_str, exprs

    @staticmethod
    def _get_repr(expr_str: str, exprs: list[pl.Expr]) -> str:
        """Return the representation of the expression."""
        expr_repr = expr_str
        for i, expr in enumerate(exprs):
            expr_repr = expr_repr.replace(f"d[{i}]", str(expr.meta.output_name()))
        return expr_repr
