from __future__ import annotations

import polars as pl

from xplor.exprs.obj import ObjExpr


class ConstrExpr(ObjExpr):
    """Represents a specific type of ObjExpr used for constraints.
    Inherits all behavior from ObjExpr.
    """

    @classmethod
    def from_obj_expr(cls, obj_expr: ObjExpr) -> ConstrExpr:
        """Create a new ConstrExpr instance from an existing ObjExpr instance
        by copying its core state.
        """
        # Create a new instance of ConstrExpr (cls) using the parent's core attributes
        # Note: We are creating a NEW object, not modifying the old one.
        new_constr = cls(expr=obj_expr._expr, name=obj_expr._name)

        new_constr._nodes = list(obj_expr._nodes)

        return new_constr

    # @property
    # def _pyexpr(self) -> plr.PyExpr:
    #     msg = (
    #         "Temporary constraints are not valid expression.\n"
    #         "Please wrap your constraint with `xplor.Model.add_constrs()`"
    #     )
    #     raise Exception(msg)

    @property
    def name(self) -> ConstrExprNameNameSpace:
        """Create an object namespace of all expressions that modify expression names."""
        return ConstrExprNameNameSpace(self)

    def alias(self, name: str) -> ConstrExpr:
        """Rename a Constraint expressiom."""
        return ConstrExpr.from_obj_expr(super().alias(name))


class ConstrExprNameNameSpace:
    """Namespace for constraint expressions that operate on expression names."""

    _accessor = "name"

    def __init__(self, expr: ConstrExpr) -> None:
        self._pyexpr = expr._pyexpr

    def prefix(self, prefix: str) -> ConstrExpr:
        """Add a prefix to the root column name of the object expression."""
        return ConstrExpr(pl.Expr._from_pyexpr(self._pyexpr.name_prefix(prefix)))

    def suffix(self, suffix: str) -> ConstrExpr:
        """Add a suffix to the root column name of the object expression."""
        return ConstrExpr(pl.Expr._from_pyexpr(self._pyexpr.name_suffix(suffix)))
