from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from xplor._utils import map_rows, series_to_df

OPERATOR_MAP = {
    "__add__": "+",
    "__radd__": "+",
    "__sub__": "-",
    "__rsub__": "-",
    "__mul__": "*",
    "__rmul__": "*",
    "__truediv__": "/",
    "__rtruediv__": "/",
    "__eq__": "==",
    "__ge__": ">=",
    "__le__": "<=",
}


class ExpressionRepr(str):
    """A special string type used to represent expressions that need to be
    evaluated dynamically for each row of a Polars DataFrame.

    Example usage:
    ```python
    >>> expr_str = ExpressionRepr("row[0] * 2 + row[1]")
    >>> row = (3, 5)
    >>> expr_str.evaluate(row)
    11
    ```
    """

    def evaluate(self, row: tuple[float | int]) -> Any:
        """Evaluate the expression with `row`."""
        return eval(self, globals(), {"row": row})


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

    Attributes:
        _expr (pl.Expr): The root expression.
        _expr_name (str | None): The root expression name.
        _nodes (list[ObjExprNode]): The internal list representing the operation AST.

    """

    def __init__(self, expr: pl.Expr, name: str | None = None) -> None:
        self._expr: pl.Expr = expr
        self._name: str | None = name
        self._nodes: list[ObjExprNode] = []

    def _repr_html_(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self.process_expression()[0]

    def __str__(self) -> str:
        expr_repr, exprs = self.process_expression()
        return self._get_str(expr_repr, exprs)

    @property
    def _pyexpr(self):  # type: ignore # noqa: ANN202
        if not self._nodes:
            return self._expr._pyexpr
        expr_repr, exprs = self.process_expression()
        if self._nodes[-1].operator in ("__eq__", "__ge__", "__le__"):
            msg = (
                "Temporary constraints are not valid expression.\n"
                "Please wrap your constraint with `xplor.Model.constr()`"
            )
            raise Exception(msg)
        return pl.map_batches(
            exprs,
            lambda s: map_rows(series_to_df(s, rename_series=True), expr_repr.evaluate),
            return_dtype=pl.Object,
        )._pyexpr

    def _append_node(self, operator: str, operand: Any) -> ObjExpr:
        """Append a node and return the current instance for chaining."""
        self._nodes.append(ObjExprNode(operator, operand))
        return self

    def __add__(self, other: Any) -> ObjExpr:
        return self._append_node("__add__", other)

    def __sub__(self, other: Any) -> ObjExpr:
        return self._append_node("__sub__", other)

    def __rsub__(self, other: Any) -> ObjExpr:
        return self._append_node("__rsub__", other)

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

    def process_expression(self) -> tuple[ExpressionRepr, list[pl.Expr]]:
        """Transform a composite object expression into a list of Polars sub-expressions
        and an equivalent lambda function, using integer indexing for all inputs.
        """
        exprs: list[pl.Expr] = [self._expr]
        expr_repr = "row[0]"

        for node in self._nodes:
            if isinstance(node.operand, pl.Expr):
                exprs.append(node.operand)
                operand_str = f"row[{len(exprs) - 1}]"
            else:
                operand_str = node.operand

            # Sequential building with parentheses to maintain precedence based on chain order
            if node.operator.startswith("__r"):
                expr_repr = f"({operand_str} {OPERATOR_MAP[node.operator]} {expr_repr})"
            else:
                expr_repr = f"({expr_repr} {OPERATOR_MAP[node.operator]} {operand_str})"
        # remove full outer parenthesis
        if self._nodes:
            expr_repr = expr_repr[1:-1]
        return ExpressionRepr(expr_repr), exprs

    def _get_str(self, expr_repr: str, exprs: list[pl.Expr]) -> str:
        """Return the representation of the expression."""
        expr_str = expr_repr

        for i, expr in enumerate(exprs):
            if i == 0 and self._name is not None:
                replacement = self._name
            elif expr.meta.is_column() or isinstance(expr, ObjExpr):
                replacement = expr.meta.output_name()
            else:
                replacement = str(expr)
                for n in expr.meta.root_names():
                    replacement = replacement.replace(f'col("{n}")', n)

            # Perform the replacement
            expr_str = expr_str.replace(
                f"row[{i}]",
                replacement,
            )

        return expr_str
