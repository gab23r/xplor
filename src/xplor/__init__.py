"""Operation research with polars."""

import xplor._dependencies as _dependencies
from xplor.exprs import var
from xplor.types import VarType

if _dependencies.is_installed("gurobipy"):
    from xplor.gurobi import XplorGurobi

if _dependencies.is_installed("ortools"):
    from xplor.mathopt import XplorMathOpt

if _dependencies.is_installed("hexaly"):
    from xplor.hexaly import XplorHexaly


__all__ = ["VarType", "XplorGurobi", "XplorHexaly", "XplorMathOpt", "var"]
