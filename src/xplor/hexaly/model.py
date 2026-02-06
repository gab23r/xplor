from __future__ import annotations

from typing import Any

import polars as pl
from hexaly.modeler import HxExpression
from hexaly.optimizer import HexalyOptimizer, HxModel, HxSolution, HxSolutionStatus

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes


class XplorHexaly(XplorModel[HxModel, HxExpression]):
    """Xplor wrapper for the Hexaly solver.

    This class extends `XplorModel` to provide an interface for building
    and solving optimization problems using Hexaly.

    Type Parameters
    ----------------
    ModelType : HxModel
        The Hexaly model type.
    ExpressionType : HxExpression
        Stores objective terms as Hexaly HxExpression objects.

    Attributes
    ----------
    optimizer: HexalyOptimizer
    model: HxModel
        The model definition within the Hexaly solver.

    """

    optimizer: HexalyOptimizer
    model: HxModel

    def __init__(self, optimizer: HexalyOptimizer | None = None) -> None:  # Updated type hint
        """Initialize the XplorHexaly model wrapper.
        If no Hexaly solver instance is provided, a new one is instantiated.

        Parameters
        ----------
        optimizer : hexaly.HexalyOptimizer | None, default None
            An optional, pre-existing Hexaly instance.

        """
        self.optimizer = HexalyOptimizer() if optimizer is None else optimizer
        super().__init__(model=self.optimizer.model)

    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
        priority: int = 0,
    ) -> pl.Series:
        """Return a series of Hexaly variables.

        Handles the conversion of Xplor's VarType to Hexaly's variable types.

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
            A Polars Object Series containing the created Hexaly variable objects.

        """
        hexaly_vars: list[HxExpression] = []
        current_objective_terms: list[HxExpression] = []

        match vtype:
            case VarType.CONTINUOUS:
                var_f = self.model.float
            case VarType.INTEGER:
                var_f = self.model.int
            case VarType.BINARY:
                var_f = lambda *_: self.model.bool()  # noqa: E731

        for lb_, ub_, obj_, name_ in df.rows():
            (var := var_f(lb_, ub_)).set_name(name_)
            hexaly_vars.append(var)

            if obj_ != 0:
                current_objective_terms.append(obj_ * var)

        self.var_types[name] = vtype
        self.vars[name] = pl.Series(hexaly_vars, dtype=pl.Object)

        # Accumulate objective coefficients for this priority level
        if current_objective_terms:
            obj_expr = self.model.sum(current_objective_terms)
            if priority not in self._priority_obj_terms:
                self._priority_obj_terms[priority] = obj_expr
            else:
                self._priority_obj_terms[priority] = self.model.sum(
                    self._priority_obj_terms[priority], obj_expr
                )

        return self.vars[name]

    def _add_constr(self, tmp_constr: HxExpression, name: str) -> None:
        tmp_constr.name = name
        self.model.add_constraint(tmp_constr)

    def optimize(self, time_limit: float | None = None) -> None:  # ty:ignore[invalid-method-override]
        """Solve the Hexaly model.

        Uses `hexaly.solve()` to solve the model.

        Parameters
        ----------
        time_limit : float | None, default None
            An optional time limit in seconds for the solver. If None,
            Hexaly's default time limit is used.

        """
        # Build multi-objective functions from accumulated terms
        # NOTE: Hexaly supports hierarchical optimization via multiple minimize/maximize calls
        if self._priority_obj_terms:
            # Sort priorities descending (highest user priority first)
            user_priorities = sorted(self._priority_obj_terms.keys(), reverse=True)

            for user_priority in user_priorities:
                # Hexaly supports hierarchical objectives via multiple minimize calls
                # The first call has the highest priority
                self.model.minimize(self._priority_obj_terms[user_priority])
        else:
            msg = "No objective function defined for the Hexaly model."
            raise Exception(msg)

        if time_limit is not None:
            self.optimizer.param.set_time_limit(time_limit)

        self.model.close()
        self.optimizer.solve()

    def get_objective_value(self) -> float:
        """Return the objective value from the solved Hexaly model.

        The value is read from the model's objective expression.

        Returns
        -------
        float
            The value of the objective function.

        Raises
        ------
        Exception
            If the model has not been optimized successfully or if no objective
            is defined.
        ValueError
            If the model has multiple objectives. Use get_multi_objective_values() instead.

        """
        sol: HxSolution = self.optimizer.get_solution()
        status: type[HxSolutionStatus] = sol.get_status()
        if status in (HxSolutionStatus.INCONSISTENT, HxSolutionStatus.INFEASIBLE):
            msg = f"The Hexaly model status is {status}."
            raise Exception(msg)

        if not self._priority_obj_terms:
            msg = "At least one objective is required in the model."
            raise Exception(msg)

        # Check if model has multiple objectives
        if len(self._priority_obj_terms) > 1:
            msg = (
                f"Model has {len(self._priority_obj_terms)} objectives. "
                "Use get_multi_objective_values() to retrieve all objective values."
            )
            raise ValueError(msg)

        # Return the single objective value
        priority = next(iter(self._priority_obj_terms.keys()))
        return self._priority_obj_terms[priority].value

    def get_multi_objective_values(self) -> dict[int, float]:
        """Return all objective values from a multi-objective Hexaly model.

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
        sol: HxSolution = self.optimizer.get_solution()
        status: type[HxSolutionStatus] = sol.get_status()
        if status in (HxSolutionStatus.INCONSISTENT, HxSolutionStatus.INFEASIBLE):
            msg = f"The Hexaly model status is {status}."
            raise Exception(msg)

        if not self._priority_obj_terms:
            return {}

        # Return all objective values by priority
        result = {}
        for priority, obj_expr in self._priority_obj_terms.items():
            result[priority] = obj_expr.value

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
            if hasattr(v, "value"):
                return v.value
            if v is None:
                return None
            return float(v)

        return name.map_batches(
            lambda d: cast_to_dtypes(
                pl.Series([_extract(v) for v in d]), self.var_types.get(d.name, VarType.CONTINUOUS)
            )
        )
