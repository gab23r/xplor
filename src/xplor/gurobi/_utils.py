from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import gurobipy as gp
import polars as pl

if TYPE_CHECKING:
    from xplor.gurobi.model import XplorGurobi

Param = ParamSpec("Param")
T = TypeVar("T")


def update_model(
    func: Callable[Concatenate["XplorGurobi", Param], T],
) -> Callable[Concatenate["XplorGurobi", Param], T]:
    @wraps(func)
    def _wrapper(self: "XplorGurobi", *args: Param.args, **kwargs: Param.kwargs) -> Any:
        result = func(self, *args, **kwargs)
        if self.auto_update:
            self.model.update()
        return result

    return _wrapper


def add_constrs_from_dataframe_args(
    df: pl.DataFrame,
    model: gp.Model,
    lhs: float | pl.Expr | pl.Series,
    sense: str,
    rhs: float | pl.Expr | pl.Series,
    name: list[str] | str,
) -> list[gp.QConstr | gp.Constr]:
    rows = df.select(lhs=lhs, rhs=rhs).rows()

    lhs_constr_type = str(type(rows[0][0]))
    rhs_constr_type = str(type(rows[0][1]))
    if "GenExpr" in lhs_constr_type or "GenExpr" in rhs_constr_type:
        _add_constr = model.addConstr
    elif "QuadExpr" in lhs_constr_type or "QuadExpr" in rhs_constr_type:
        _add_constr = model.addQConstr
    else:
        _add_constr = model.addLConstr

    if sense in ("<=", "<"):
        operator = "__le__"
    elif sense in (">=", ">"):
        operator = "__ge__"
    elif sense in ("==", "="):
        operator = "__eq__"
    else:
        msg = f"sense should be one of ('<=', '>=', '=='), got {sense}"
        raise Exception(msg)

    if isinstance(name, str):
        constrs = [_add_constr(getattr(lhs, operator)(rhs), name=name) for lhs, rhs in rows]
    else:
        constrs = [
            _add_constr(getattr(lhs, operator)(rhs), name=name)
            for name, (lhs, rhs) in zip(name, rows, strict=True)
        ]

    return constrs
