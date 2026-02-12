"""CPLEX CP backend for constraint programming with Polars."""

from xplor.cplex_cp.model import XplorCplexCP
from xplor.cplex_cp.var import _ProxyCplexCPVarExpr

var = _ProxyCplexCPVarExpr()
__all__ = ["XplorCplexCP", "var"]
