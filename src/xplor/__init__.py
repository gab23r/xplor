"""Operation research with polars."""

import builtins
import contextlib

from xplor.types import VarType
from xplor.var_expr import _ProxyObjExpr

with contextlib.suppress(builtins.BaseException):
    from xplor.gurobi import XplorGurobi

with contextlib.suppress(builtins.BaseException):
    from xplor.mathopt import XplorMathOpt

with contextlib.suppress(builtins.BaseException):
    from xplor.hexaly import XplorHexaly
var = _ProxyObjExpr()

__all__ = ["VarType", "XplorGurobi", "XplorHexaly", "XplorMathOpt", "var"]
