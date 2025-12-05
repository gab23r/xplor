"""Operation research with polars."""

import builtins
import contextlib

from xplor.model import XplorModel
from xplor.types import VarType
from xplor.var_expr import _ProxyObjExpr

with contextlib.suppress(builtins.BaseException):
    from xplor.gurobi import XplorGurobi

with contextlib.suppress(builtins.BaseException):
    from xplor.mathopt import XplorMathOpt

var = _ProxyObjExpr()

__all__ = ["VarType", "XplorGurobi", "XplorMathOpt", "XplorModel", "var"]
