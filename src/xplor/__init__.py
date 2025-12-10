"""Operation research with polars."""

import builtins
import contextlib

from xplor.exprs.var import _ProxyVarExpr
from xplor.types import VarType

with contextlib.suppress(builtins.BaseException):
    from xplor.gurobi import XplorGurobi

with contextlib.suppress(builtins.BaseException):
    from xplor.mathopt import XplorMathOpt

with contextlib.suppress(builtins.BaseException):
    from xplor.hexaly import XplorHexaly
var = _ProxyVarExpr()

__all__ = ["VarType", "XplorGurobi", "XplorHexaly", "XplorMathOpt", "var"]
