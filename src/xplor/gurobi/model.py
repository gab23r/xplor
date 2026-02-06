from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gurobipy as gp
import polars as pl

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes

if TYPE_CHECKING:
    from gurobipy import TempLConstr


class XplorGurobi(XplorModel[gp.Model, gp.LinExpr]):
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
    vars : dict[str, pl.Series]
        A dictionary storing Polars Series of optimization variables,
        indexed by name.
    var_types : dict[str, VarType]
        A dictionary storing the `VarType` (CONTINUOUS, INTEGER, BINARY)
        for each variable series, indexed by its base name.

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

    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
        priority: int = 0,
    ) -> pl.Series:
        """Create Gurobi variables and accumulate objective coefficients by priority.

        For multi-objective optimization, objective coefficients are not passed to
        addMVar. Instead, they are accumulated by priority level and built into
        objective expressions in optimize() using setObjectiveN().

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame containing the columns ["lb", "ub", "obj", "name"].
        name : str
            The base name for the variables.
        vtype : VarType, default VarType.CONTINUOUS
            The type of the variable.
        priority : int, default 0
            Multi-objective optimization priority (higher values optimized first).

        Returns
        -------
        pl.Series
            A Polars Object Series containing the created Gurobi variable objects.

        """
        self.var_types[name] = vtype

        # Create variables WITHOUT obj parameter (required for multi-objective)
        mvar = self.model.addMVar(
            df.height,
            vtype=getattr(gp.GRB, vtype),
            lb=df["lb"].to_numpy() if df["lb"].dtype != pl.Null else None,
            ub=df["ub"].to_numpy() if df["ub"].dtype != pl.Null else None,
            name=df["name"].to_list(),
        )

        # Store variables as Series
        self.vars[name] = pl.Series(mvar.tolist(), dtype=pl.Object())

        # Accumulate objective coefficients for this priority level
        # Only store non-zero coefficients to avoid unnecessary terms
        if df.filter(pl.col("obj") != 0).height:
            obj_expr = gp.LinExpr(df["obj"].to_list(), mvar.tolist())
            if priority not in self._priority_obj_terms:
                self._priority_obj_terms[priority] = obj_expr
            else:
                self._priority_obj_terms[priority] += obj_expr

        self.model.update()
        return self.vars[name]

    def _add_constr(self, tmp_constr: TempLConstr, name: str) -> None:
        self.model.addLConstr(tmp_constr, name=name)

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
                pl.Series([_extract(v) for v in d]), self.var_types.get(d.name, VarType.CONTINUOUS)
            )
        )
