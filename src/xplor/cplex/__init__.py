"""CPLEX backend for xplor using docplex."""

from xplor.cplex.model import XplorCplex
from xplor.cplex.var import _ProxyCplexVarExpr

var = _ProxyCplexVarExpr()
__all__ = ["XplorCplex", "var"]
