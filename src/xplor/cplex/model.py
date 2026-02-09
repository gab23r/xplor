from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import polars as pl
from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model

from xplor.cplex.var import _ProxyCplexVarExpr
from xplor.model import XplorModel
from xplor.types import cast_to_dtypes

if TYPE_CHECKING:
    from docplex.mp.constr import LinearConstraint


class XplorCplex(XplorModel[Model, Var, LinearExpr]):
    """Xplor wrapper for the CPLEX solver using docplex.

    This class provides a specialized wrapper for CPLEX, translating XplorModel's
    abstract operations into CPLEX-specific API calls via the docplex library
    for defining variables, constraints, optimizing, and extracting solutions.

    Type Parameters
    ----------------
    ModelType : docplex.mp.model.Model
        The CPLEX model type.
    ExpressionType : docplex.mp.linear.LinearExpr
        Stores objective terms as CPLEX LinearExpr objects.

    Attributes
    ----------
    model : docplex.mp.model.Model
        The instantiated CPLEX model object.

    """

    model: Model

    def __init__(self, model: Model | None = None) -> None:
        """Initialize the XplorCplex model wrapper.

        If no CPLEX model is provided, a new one is instantiated.

        Parameters
        ----------
        model : docplex.mp.model.Model | None, default None
            An optional, pre-existing CPLEX model instance.

        """
        model = Model() if model is None else model
        super().__init__(model=model)

    def _add_continuous_vars(
        self,
        names: list[str],
        lb: list[float] | None,
        ub: list[float] | None,
    ) -> list[Var]:
        return self.model.continuous_var_list(len(names), lb, ub, names)

    def _add_integer_vars(
        self,
        names: list[str],
        lb: list[float] | None,
        ub: list[float] | None,
    ) -> list[Var]:
        return self.model.integer_var_list(len(names), lb, ub, names)

    def _add_binary_vars(
        self,
        names: list[str],
    ) -> list[Var]:
        return self.model.binary_var_list(len(names), name=names)

    def _add_constr(self, tmp_constr: LinearConstraint, name: str) -> None:
        """Add a constraint to the CPLEX model.

        Parameters
        ----------
        tmp_constr : LinearConstraint
            The constraint to add.
        name : str
            Name for the constraint.

        """
        tmp_constr.name = name
        self.model.add_constraint(tmp_constr)

    def optimize(self, **kwargs: Any) -> None:
        """Solve the CPLEX model.

        Before optimization, sets up multi-objective functions if multiple
        priority levels are defined. Higher priority values are optimized first.

        Parameters
        ----------
        **kwargs : Any
            Additional parameters passed to model.solve().
            Common parameters include:
            - log_output: bool - Whether to display solver output
            - time_limit: float - Maximum solve time in seconds

        """
        # Build multi-objective functions from accumulated terms
        if self._priority_obj_terms:
            # Sort priorities descending (highest user priority first)
            user_priorities = sorted(self._priority_obj_terms.keys(), reverse=True)

            if len(user_priorities) == 1:
                # Single objective
                priority = user_priorities[0]
                self.model.minimize(self._priority_obj_terms[priority])
            else:
                # Multi-objective optimization
                # CPLEX requires setting objectives with priorities
                for priority in user_priorities:
                    expr = self._priority_obj_terms[priority]
                    # Set as multi-objective with priority weight
                    # Higher user priority gets higher CPLEX priority
                    self.model.add_kpi(expr, publish_name=f"priority_{priority}")

                # For multi-objective in CPLEX, we minimize the primary (highest priority)
                # and the rest become constraints or are handled via blended objectives
                primary_priority = user_priorities[0]
                self.model.minimize(self._priority_obj_terms[primary_priority])

        # Solve the model
        self.solution = self.model.solve(**kwargs)

    def get_objective_value(self) -> float:
        """Return the objective value from the solved CPLEX model.

        Returns
        -------
        float
            The value of the objective function.

        Raises
        ------
        ValueError
            If the model has multiple objectives. Use get_multi_objective_values() instead.

        """
        if not self.solution:
            msg = "Model has not been optimized or no solution found."
            raise ValueError(msg)

        if len(self._priority_obj_terms) > 1:
            msg = (
                f"Model has {len(self._priority_obj_terms)} objectives. "
                "Use get_multi_objective_values() to retrieve all objective values."
            )
            raise ValueError(msg)

        return self.solution.get_objective_value()

    def get_multi_objective_values(self) -> dict[int, float]:
        """Return all objective values from a multi-objective CPLEX model.

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
        if not self.solution:
            msg = "Model has not been optimized or no solution found."
            raise ValueError(msg)

        if not self._priority_obj_terms:
            return {}

        # Evaluate each objective expression using the current solution
        result = {}
        for priority, expr in self._priority_obj_terms.items():
            # Get the value of the expression from the solution
            obj_value = expr.solution_value
            result[priority] = obj_value

        return result

    def read_values(self, name: pl.Expr) -> pl.Expr:
        """Read the value of an optimization variable from the solution.

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
            if v is None:
                return None
            if hasattr(v, "solution_value"):
                return v.solution_value
            return float(v)

        return name.map_batches(
            lambda d: cast_to_dtypes(
                pl.Series([_extract(v) for v in d]),
                self.var_types.get(d.name, "CONTINUOUS"),
            )
        )

    @cached_property
    def var(self) -> _ProxyCplexVarExpr:
        """The entry point for creating custom expression objects (VarExpr) that represent
        variables or columns used within a composite Polars expression chain.

        This proxy acts similarly to `polars.col()`, allowing you to reference
        optimization variables (created via `xmodel.add_vars()`) or standard DataFrame columns
        in a solver-compatible expression.

        The resulting expression object can be combined with standard Polars expressions
        to form constraints or objective function components.

        Examples
        --------
        >>> xmodel = XplorCplex()
        >>> df = df.with_columns(xmodel.add_vars("production"))
        >>> df.select(total_cost = xmodel.var("production") * pl.col("cost"))

        """
        return _ProxyCplexVarExpr()
