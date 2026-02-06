from __future__ import annotations

from typing import Any

import polars as pl
from ortools.math_opt.python import mathopt, parameters, result
from ortools.math_opt.python.variables import LinearSum

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes


class XplorMathOpt(XplorModel[mathopt.Model, LinearSum]):
    """Xplor wrapper for the OR-Tools MathOpt solver.

    This class extends `XplorModel` to provide an interface for building
    and solving optimization problems using OR-Tools MathOpt.

    Type Parameters
    ----------------
    ModelType : mathopt.Model
        The MathOpt model type.
    ExpressionType : LinearSum
        Stores objective terms as MathOpt LinearSum expression objects.

    Attributes
    ----------
    model : mathopt.Model
        The underlying OR-Tools MathOpt model instance.
    vars : dict[str, pl.Series]
        A dictionary storing Polars Series of optimization variables,
        indexed by name.
    var_types : dict[str, VarType]
        A dictionary storing the `VarType` (CONTINUOUS, INTEGER, BINARY)
        for each variable series, indexed by its base name.
    result : result.SolveResult
        The result object returned by MathOpt after optimization.
        It contains solution status, objective value, and variable values.

    """

    model: mathopt.Model
    result: result.SolveResult

    def __init__(self, model: mathopt.Model | None = None) -> None:
        """Initialize the XplorMathOpt model wrapper.

        If no MathOpt model is provided, a new one is instantiated.

        Parameters
        ----------
        model : mathopt.Model | None, default None
            An optional, pre-existing MathOpt model instance.

        """
        model = mathopt.Model() if model is None else model
        super().__init__(model=model)

    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
        priority: int = 0,
    ) -> pl.Series:
        """Return a series of MathOpt variables.

        Handles the conversion of Xplor's VarType to MathOpt's boolean `is_integer` flag.
        For "BINARY" types, bounds are explicitly clipped to [0, 1] as a prerequisite for MathOpt.

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
            A Polars Object Series containing the created MathOpt variable objects.

        """
        # mathopt.Model don't super binary variable directly
        if vtype == "BINARY":
            df = df.with_columns(
                pl.col("lb").fill_null(0).clip(lower_bound=0).fill_null(0),
                pl.col("ub").fill_null(1).clip(upper_bound=1).fill_null(1),
            )
        self.var_types[name] = vtype
        self.vars[name] = pl.Series(
            [
                self.model.add_variable(
                    lb=lb_, ub=ub_, name=name_, is_integer=vtype != VarType.CONTINUOUS
                )
                for lb_, ub_, name_ in df.drop("obj").rows()
            ],
            dtype=pl.Object,
        )
        # Accumulate objective coefficients for this priority level
        if df.select("obj").filter(pl.col("obj") != 0).height:
            obj_expr = sum(w * v for w, v in zip(df["obj"], self.vars[name], strict=True))
            if priority not in self._priority_obj_terms:
                self._priority_obj_terms[priority] = obj_expr
            else:
                self._priority_obj_terms[priority] += obj_expr

        return self.vars[name]

    def _add_constr(self, tmp_constr: Any, name: str) -> None:
        self.model.add_linear_constraint(tmp_constr, name=name)

    def optimize(self, solver_type: parameters.SolverType | None = None) -> None:  # ty:ignore[invalid-method-override]
        """Solve the MathOpt model.

        Uses `mathopt.solve()` to solve the model and stores the result internally.

        Parameters
        ----------
        solver_type : parameters.SolverType | None, default SolverType.GLOP
            The specific OR-Tools solver to use (e.g., GLOP, GSCIP).
            Defaults to MathOpt's native GLOP solver if none is provided.

        Examples
        --------
        1. Using the default solver (GLOP):
           >>> xmodel.optimize()

        2. Specifying a different solver (requires setup/licensing for commercial solvers):
           >>> from ortools.math_opt.python.parameters import SolverType
           >>> xmodel.optimize(solver_type=SolverType.GUROBI)

        """
        solver_type = mathopt.SolverType.GLOP if solver_type is None else solver_type

        # Build multi-objective functions from accumulated terms
        if self._priority_obj_terms:
            # Sort priorities descending (highest user priority first)
            user_priorities = sorted(self._priority_obj_terms.keys(), reverse=True)
            max_user_priority = max(user_priorities)

            for user_priority in user_priorities:
                # Get the accumulated expression for this priority level
                obj_expr = self._priority_obj_terms[user_priority]

                # Invert priority: higher user priority → lower MathOpt priority
                mathopt_priority = max_user_priority - user_priority

                if mathopt_priority == 0:
                    # Set as primary objective
                    self.model.objective.set_to_linear_expression(obj_expr)
                    self.model.objective.is_maximize = False
                else:
                    # Add as auxiliary objective
                    self.model.add_auxiliary_objective(
                        priority=mathopt_priority,
                        name=f"priority_{user_priority}",
                        expr=obj_expr,
                        is_maximize=False,
                    )

        self.result = mathopt.solve(self.model, solver_type)

    def get_objective_value(self) -> float:
        """Return the objective value from the solved MathOpt model.

        The value is read from the stored `result` object.

        Returns
        -------
        float
            The value of the objective function.

        Raises
        ------
        Exception
            If the model has not been optimized successfully (i.e., `self.result` is None).
        ValueError
            If the model has multiple objectives. Use get_multi_objective_values() instead.

        """
        if self.result is None:
            msg = "The model is not optimized."
            raise Exception(msg)

        # Check if model has multiple objectives
        num_aux_objs = len(list(self.model.auxiliary_objectives()))
        if num_aux_objs > 0:
            total_objs = 1 + num_aux_objs
            msg = (
                f"Model has {total_objs} objectives. "
                "Use get_multi_objective_values() to retrieve all objective values."
            )
            raise ValueError(msg)

        return self.result.objective_value()

    def get_multi_objective_values(self) -> dict[int, float]:
        """Return all objective values from a multi-objective MathOpt model.

        Returns a dictionary mapping user priority levels to their objective values.

        Returns
        -------
        dict[int, float]
            Dictionary mapping priority level to objective value.
            Keys are user priority levels (higher priority = higher number).
            Values are the objective values for each priority.

        Examples
        --------
        >>> xmodel.optimize(solver_type=mathopt.SolverType.GSCIP)
        >>> obj_values = xmodel.get_multi_objective_values()
        >>> print(obj_values)
        {2: -150.0, 1: 50.0, 0: 10.0}  # priority -> objective value

        """
        if self.result is None or not self._priority_obj_terms:
            return {}

        result_dict = {}
        user_priorities = sorted(self._priority_obj_terms.keys(), reverse=True)
        max_user_priority = max(user_priorities)

        # Primary objective (mathopt_priority=0) maps to highest user priority
        result_dict[max_user_priority] = self.result.objective_value()

        # Compute auxiliary objective values by evaluating expressions
        # MathOpt doesn't provide direct access to auxiliary objective values in results,
        # so we compute them from the stored expressions and variable values
        var_values = self.result.variable_values()

        for aux_obj in self.model.auxiliary_objectives():
            # Extract user priority from name "priority_{user_priority}"
            obj_name = aux_obj.name
            if obj_name.startswith("priority_"):
                user_priority = int(obj_name.split("_")[1])

                # Evaluate the auxiliary objective using variable values
                obj_value = aux_obj.offset
                for term in aux_obj.linear_terms():
                    obj_value += term.coefficient * var_values.get(term.variable, 0.0)

                result_dict[user_priority] = obj_value

        return result_dict

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
        result_values = self.result.variable_values()
        return name.map_batches(
            lambda d: cast_to_dtypes(
                pl.Series([result_values.get(v) for v in d]),
                self.var_types.get(d.name, VarType.CONTINUOUS),
            )
        )
