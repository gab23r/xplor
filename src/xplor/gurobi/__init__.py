"""Xplor Gurobi backend."""

from xplor.gurobi.model import XplorGurobi
from xplor.gurobi.var import _ProxyGurobiVarExpr

var = _ProxyGurobiVarExpr()
__all__ = ["XplorGurobi", "var"]
