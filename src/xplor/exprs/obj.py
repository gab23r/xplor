from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import polars as pl
import polars._plr as plr

from xplor._utils import expr_ends_with_alias, map_rows, series_to_df

if TYPE_CHECKING:
    from polars._typing import IntoExpr, IntoExprColumn

OPERATOR_MAP: dict[str, str] = {
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

    def evaluate(self, row: tuple[float | int, ...]) -> Any:
        """Evaluate the expression with `row`."""
        return eval(self, globals(), {"row": row})


@dataclass
class ObjExprNode:
    """Represents a single operation (operator) and its value (operand)."""

    operator: str  #  '__add__', '__rtruediv__'
    operand: pl.Expr | float  # 1, or ObjExpr('b')


class ObjExpr(pl.Expr):
    """Custom Polars Expression wrapper designed for building composite expressions
    that mix standard pl.Expr and custom variables.

    ObjExpr constructs an internal Abstract Syntax Tree (AST) of operations, which
    is then transformed into an efficient pl.map_batches call during execution.
    This allows complex, multi-variable logic to run within Polars' optimized
    batch context.

    Attributes:
        _expr (pl.Expr): The root expression.
        _name (str | None): The root expression name.
        _nodes (list[ObjExprNode]): The internal list representing the operation AST.

    """

    def __init__(self, expr: pl.Expr, name: str | None = None) -> None:
        self._expr: pl.Expr = expr
        self._name: str | None = name
        self._nodes: list[ObjExprNode] = []

    def _repr_html_(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self.parse()[0]

    def __str__(self) -> str:
        expr_repr, exprs = self.parse()
        return self._get_str(expr_repr, exprs)

    def _to_expr(self) -> pl.Expr:
        if not self._nodes:
            return self._expr
        expr_repr, exprs = self.parse()

        return pl.map_batches(
            exprs,
            lambda s: map_rows(series_to_df(s, rename=True), expr_repr.evaluate),
            return_dtype=pl.Object,
        )

    @property
    def _pyexpr(self) -> plr.PyExpr:
        return self._to_expr()._pyexpr

    def _append_node(self, operator: str, operand: pl.Expr | float) -> Self:
        """Append a node and return the current instance for chaining."""
        self._nodes.append(ObjExprNode(operator, operand))
        return self

    def __add__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__add__", other)

    def __sub__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__sub__", other)

    def __rsub__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__rsub__", other)

    def __radd__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__radd__", other)

    def __truediv__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__truediv__", other)

    def __rtruediv__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__rtruediv__", other)

    def __mul__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__mul__", other)

    def __rmul__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__rmul__", other)

    def __eq__(self, other: pl.Expr | float) -> Self:  # type: ignore[override]
        return self._append_node("__eq__", other)

    def __le__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__le__", other)

    def __ge__(self, other: pl.Expr | float) -> Self:  # ty:ignore[invalid-method-override]
        return self._append_node("__ge__", other)

    def alias(self, name: str) -> ObjExpr:
        """Rename the object expression."""
        self._name = name
        self._expr = self._expr.alias(name)
        return self

    def shift(self, n: int | IntoExprColumn = 1, *, fill_value: IntoExpr | None = None) -> ObjExpr:
        """Shift values by the given number of indices."""
        if isinstance(n, int):

            def shift_series(d: pl.Series) -> pl.Series:
                data_list = d.to_list()  # Convert Series chunk to a list for easier manipulation
                N = len(data_list)
                fill_values = [fill_value] * min(abs(n), N)
                return pl.Series(
                    fill_values + data_list[:-n] if n > 0 else data_list[-n:] + fill_values,
                    dtype=pl.Object,
                )

            return ObjExpr(self.map_batches(shift_series, return_dtype=pl.Object))
        else:
            return ObjExpr(pl.Expr.shift(self, n, fill_value=fill_value))

    def fill_null(self, value: Any | pl.Expr | None = None) -> ObjExpr:  # type: ignore
        """fill_null implementation for object."""
        if isinstance(value, pl.Expr):
            return ObjExpr(pl.Expr.fill_null(self, value))

        return ObjExpr(self.fill_null(pl.lit(value, dtype=pl.Object)))

    @property
    def name(self) -> ObjExprNameNameSpace:
        """Create an object namespace of all expressions that modify expression names."""
        return ObjExprNameNameSpace(self)

    def parse(self, exprs: list[pl.Expr] | None = None) -> tuple[ExpressionRepr, list[pl.Expr]]:
        """Transform a composite object expression into a list of Polars sub-expressions
        and an equivalent lambda function, using integer indexing for all inputs.
        """
        if exprs is None:
            exprs = [self._expr]
            expr_repr = "row[0]"
        else:
            expr_repr = self._get_expr_repr(exprs, self._expr)
        for node in self._nodes:
            if isinstance(node.operand, pl.Expr):
                # Check if this is a column expression that has already been added to `exprs`
                operand_str = self._get_expr_repr(exprs, node.operand)
            else:
                operand_str = node.operand

            # Sequential building with parentheses to maintain precedence based on chain order
            if node.operator.startswith("__r"):
                expr_repr: str = f"({operand_str} {OPERATOR_MAP[node.operator]} {expr_repr})"
            else:
                expr_repr: str = f"({expr_repr} {OPERATOR_MAP[node.operator]} {operand_str})"
        # remove full outer parenthesis
        if self._nodes:
            expr_repr = expr_repr[1:-1]
        return ExpressionRepr(expr_repr), exprs

    def _get_expr_repr(self, exprs: list[pl.Expr], expr: pl.Expr) -> str:
        """Get the index of an expr in the list of expressions.

        If the expression is not present, it is inserted at the end of the list.
        """
        expr_index = next(
            (
                i
                for i, other_expr in enumerate(exprs)
                if expr.meta.undo_aliases().meta.eq(other_expr.meta.undo_aliases())
            ),
            len(exprs),
        )
        if expr_index == len(exprs):
            exprs.append(expr)
        return f"row[{expr_index}]"

    def _get_str(self, expr_repr: str, exprs: list[pl.Expr]) -> str:
        """Return the representation of the expression."""
        expr_str = expr_repr

        for i, expr in enumerate(exprs):
            if i == 0 and self._name is not None:
                replacement = self._name
            elif isinstance(expr, ObjExpr):
                replacement = expr._get_str(*expr.parse())
            elif expr.meta.is_column() or expr_ends_with_alias(expr):
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


class ObjExprNameNameSpace:
    """Namespace for expressions that operate on expression names."""

    _accessor = "name"

    def __init__(self, expr: ObjExpr) -> None:
        self._pyexpr = expr._pyexpr

    def prefix(self, prefix: str) -> ObjExpr:
        """Add a prefix to the root column name of the object expression."""
        return ObjExpr(pl.Expr._from_pyexpr(self._pyexpr.name_prefix(prefix)))

    def suffix(self, suffix: str) -> ObjExpr:
        """Add a suffix to the root column name of the object expression."""
        return ObjExpr(pl.Expr._from_pyexpr(self._pyexpr.name_suffix(suffix)))
