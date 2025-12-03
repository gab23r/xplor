from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl

from xplor._utils import parse_into_expr, series_to_df
from xplor.var import VarType

if TYPE_CHECKING:
    import gurobipy as gp
    from ortools.math_opt.python import mathopt

    from xplor.obj_expr import ObjExpr


class XplorModel(ABC):
    """Abstract base class for all Xplor optimization model wrappers.

    Defines the core interface for adding variables and constraints
    to the underlying optimization model (e.g., MathOpt, Gurobi, etc.).
    """

    def __init__(self, model: gp.Model | mathopt.Model) -> None:
        """Initialize the model wrapper.

        Concrete implementations must handle model initialization and setup.
        """
        self.model = model

    def var(
        self,
        name: str,
        *,
        lb: float | str | pl.Expr = 0.0,
        ub: float | str | pl.Expr | None = None,
        obj: float | str | pl.Expr = 0.0,
        indices: pl.Expr | list[str] | None = None,
        vtype: VarType = VarType.CONTINUOUS,
    ) -> pl.Expr:
        """Define and return a Var expression for optimization variables.

        Parameters
        ----------
        name : str
            The base name for the variables.
        lb : float | str | pl.Expr
            Lower bound for created variables.
        ub : float | str | pl.Expr
            Upper bound for created variables.
        obj: float | str | pl.Expr
            Objective function coefficient for created variables.
        indices: list[str]
            Keys (column names) that uniquely identify each variable.
        vtype: VarType, default "CONTINUOUS"
            The type of the variable.

        Returns
        -------
        Var
            A Var expression that, when consumed, adds variables to the model.

        """
        indices = pl.row_index() if indices is None else indices

        return pl.map_batches(
            [
                parse_into_expr(lb).alias("lb"),
                parse_into_expr(ub).alias("ub"),
                parse_into_expr(obj).alias("obj"),
                pl.format(f"{name}[{{}}]", pl.concat_str(indices, separator=",")).alias("name"),
            ],
            lambda s: self._add_vars(series_to_df(s), vtype=vtype),
            return_dtype=pl.Object,
        ).alias(name)

    def constr(self, expr: ObjExpr, name: str | None = None) -> pl.Expr:
        """Define and return a Constr expression for model constraints.

        Parameters
        ----------
        expr : ObjExpr
            The constraint expression (e.g., a relational expression).
        name : str | None, default None
            The base name for the constraints.

        Returns
        -------
        Constr
            A Constr expression that, when consumed, adds constraints to the model.

        """
        expr_str, exprs = expr.process_expression()
        name = name or expr._get_repr(expr_str, exprs)
        return pl.map_batches(
            exprs,
            lambda s: self._add_constrs(series_to_df(s), name=name, expr_str=expr_str),
            return_dtype=pl.Object,
        ).alias(name)

    @abstractmethod
    def optimize(self) -> Any:
        """Solve the model."""

    @abstractmethod
    def get_objective_value(self) -> float:
        """Return the objective value."""

    @abstractmethod
    def _add_constrs(self, df: pl.DataFrame, name: str, expr_str: str) -> pl.Series:
        """Return a series of variables.

        `df` should contains columns: ["lb", "ub", "name"].
        """

    @abstractmethod
    def _add_vars(
        self,
        df: pl.DataFrame,
        vtype: VarType = VarType.CONTINUOUS,
    ) -> pl.Series:
        """Return a series of variables.

        `df` should contains columns: ["lb", "ub", "obj, "name"].
        """
