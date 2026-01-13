"""Xplor Expressions."""

from xplor.exprs.constr import ConstrExpr
from xplor.exprs.var import VarExpr, _ProxyVarExpr

var = _ProxyVarExpr()

__all__ = ["ConstrExpr", "VarExpr", "var"]
