from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import gurobipy as gp
import polars as pl

from xplor.gurobi.var import _ProxyGurobiVarExpr, to_mvar_or_mlinexpr
from xplor.model import XplorModel
from xplor.types import cast_to_dtypes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gurobipy import TempGenConstr, TempLConstr

    from xplor.exprs.obj import ExpressionRepr


class XplorGurobi(XplorModel[gp.Model, gp.Var, gp.LinExpr]):
    """Xplor wrapper for the Gurobi solver.

    This class provides a specialized wrapper for Gurobi, translating XplorModel's
    abstract operations into Gurobi-specific API calls for defining variables,
    constraints, optimizing, and extracting solutions.

    Type Parameters
    ----------------
    ModelType : gp.Model
        The Gurobi model type.
    ExpressionType : gp.LinExpr
        Stores objective terms as Gurobi LinExpr objects.

    Attributes
    ----------
    model : gp.Model
        The instantiated Gurobi model object.

    """

    model: gp.Model

    def __init__(self, model: gp.Model | None = None) -> None:
        """Initialize the XplorGurobi model wrapper.

        If no Gurobi model is provided, a new one is instantiated.

        Parameters
        ----------
        model : gurobipy.Model | None, default None
            An optional, pre-existing Gurobi model instance.

        """
        model = gp.Model() if model is None else model
        super().__init__(model=model)

    def _add_continuous_vars(
        self,
        names: list[str],
        lb: list[float] | None,
        ub: list[float] | None,
    ) -> list[gp.Var]:
        mvar = self.model.addMVar(
            len(names),
            lb=lb,
            ub=ub,
            vtype=gp.GRB.CONTINUOUS,
            name=names,
        )  # ty:ignore[no-matching-overload]
        self.model.update()
        return mvar.tolist()

    def _add_integer_vars(
        self,
        names: list[str],
        lb: list[float] | None,
        ub: list[float] | None,
    ) -> list[gp.Var]:
        mvar = self.model.addMVar(
            len(names),
            vtype=gp.GRB.INTEGER,
            lb=lb,
            ub=ub,
            name=names,
        )  # ty:ignore[no-matching-overload]
        self.model.update()
        return mvar.tolist()

    def _add_binary_vars(
        self,
        names: list[str],
    ) -> list[gp.Var]:
        mvar = self.model.addMVar(
            len(names),
            vtype=gp.GRB.BINARY,
            name=names,
        )
        self.model.update()
        return mvar.tolist()

    def _add_constr(self, tmp_constr: TempLConstr | TempGenConstr, name: str) -> None:
        self.model.addConstr(tmp_constr, name=name)

    def _add_constrs(
        self,
        df: pl.DataFrame,
        /,
        indices: pl.Series,
        **constrs_repr: ExpressionRepr,
    ) -> None:
        """Add constraints.

        Overrides base class to use MVar for full vectorization.
        """
        arrays: tuple = tuple(to_mvar_or_mlinexpr(s) for s in df.iter_columns())
        # Evaluate each constraint expression vectorized - NO Python loops!
        for name, constr_repr in constrs_repr.items():
            # Handle gp.GenExpr case, for loop is needed
            rhs_idx = constr_repr.extract_indices(side="rhs")
            if rhs_idx and isinstance(df[:, rhs_idx[0]].first(ignore_nulls=True), gp.GenExpr):
                for row, idx in zip(df.rows(), indices, strict=True):
                    self._add_constr(constr_repr.evaluate(row), name=f"{name}[{idx}]")
            else:
                result = constr_repr.evaluate(df.row(0) if df.height == 1 else arrays)
                self._add_constr(result, name=name)

    def optimize(self, **kwargs: Any) -> None:
        """Solve the Gurobi model.

        Before optimization, sets up multi-objective functions using setObjectiveN
        if multiple priority levels are defined. Higher priority values are optimized
        first (consistent with Gurobi's convention).

        """
        # Build multi-objective functions from accumulated terms
        if self._priority_obj_terms:
            # Sort priorities descending (highest user priority first)
            user_priorities = sorted(self._priority_obj_terms.keys(), reverse=True)

            for obj_index, priority in enumerate(user_priorities):
                # Set multi-objective
                self.model.setObjectiveN(
                    self._priority_obj_terms[priority],
                    index=obj_index,
                    priority=priority,
                    weight=1.0,
                    name=f"priority_{priority}",
                )

            self.model.update()

        self.model.optimize(**kwargs)

    def get_objective_value(self) -> float:
        """Return the objective value from the solved Gurobi model.

        Returns
        -------
        float
            The value of the objective function.

        Raises
        ------
        ValueError
            If the model has multiple objectives. Use get_multi_objective_values() instead.

        """
        if self.model.NumObj > 1:
            msg = (
                f"Model has {self.model.NumObj} objectives. "
                "Use get_multi_objective_values() to retrieve all objective values."
            )
            raise ValueError(msg)
        return self.model.getObjective().getValue()

    def get_multi_objective_values(self) -> dict[int, float]:
        """Return all objective values from a multi-objective Gurobi model.

        Returns a dictionary mapping user priority levels to their objective values.

        Returns
        -------
        dict[int, float]
            Dictionary mapping priority level to objective value.
            Keys are user priority levels (higher priority = higher number).
            Values are the objective values for each priority.

        Examples
        --------
        >>> xmodel.optimize()
        >>> obj_values = xmodel.get_multi_objective_values()
        >>> print(obj_values)
        {2: -150.0, 1: 50.0, 0: 10.0}  # priority -> objective value

        """
        if self.model.NumObj == 0:
            return {}

        # Build mapping from Gurobi objective index to user priority
        # We stored objectives with name "priority_{user_priority}"
        result = {}

        for obj_idx in range(self.model.NumObj):
            self.model.setParam("ObjNumber", obj_idx)
            obj_name = self.model.ObjNName

            # Extract user priority from name "priority_{user_priority}"
            if obj_name.startswith("priority_"):
                user_priority = int(obj_name.split("_")[1])
                result[user_priority] = self.model.ObjNVal

        return result

    def read_values(self, name: pl.Expr) -> pl.Expr:
        """Read the value of an optimization variable.

        Parameters
        ----------
        name : pl.Expr
            Expression to evaluate.

        Returns
        -------
        pl.Expr
            Values of the variable expression.

        Examples
        --------
        >>> xmodel: XplorModel
        >>> df_with_solution = df.with_columns(xmodel.read_values(pl.selectors.object()))

        """

        def _extract(v: Any) -> float | None:
            if hasattr(v, "x"):
                return v.x
            if v is None:
                return None
            if hasattr(v, "getValue"):
                return v.getValue()
            return float(v)

        return name.map_batches(
            lambda d: cast_to_dtypes(
                pl.Series([_extract(v) for v in d]), self.var_types.get(d.name, "CONTINUOUS")
            )
        )

    def _linear_expr(self, arg1: Sequence[float], arg2: Sequence[gp.Var]) -> gp.LinExpr:
        return gp.LinExpr(arg1, arg2)

    @cached_property
    def var(self) -> _ProxyGurobiVarExpr:
        """The entry point for creating custom expression objects (VarExpr) that represent
        variables or columns used within a composite Polars expression chain.

        This proxy acts similarly to `polars.col()`, allowing you to reference
        optimization variables (created via `xmodel.add_vars()`) or standard DataFrame columns
        in a solver-compatible expression.

        The resulting expression object can be combined with standard Polars expressions
        to form constraints or objective function components.

        Examples
        --------
        >>> xmodel = XplorMathOpt()
        >>> df = df.with_columns(xmodel.add_vars("production"))
        >>> df.select(total_cost = xmodel.var("production") * pl.col("cost"))

        """
        return _ProxyGurobiVarExpr()
