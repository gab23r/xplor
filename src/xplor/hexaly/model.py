from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from hexaly.optimizer import HexalyOptimizer, HxModel, HxSolution, HxSolutionStatus

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes

if TYPE_CHECKING:
    from hexaly.modeler import HxExpression

    from xplor.obj_expr import ExpressionRepr


class XplorHexaly(XplorModel):
    """Xplor wrapper for the Hexaly solver.

    This class extends `XplorModel` to provide an interface for building
    and solving optimization problems using Hexaly.

    Attributes
    ----------
    optimizer: HexalyOptimizer
    model: HxModel
        The model definition within the Hexaly solver.

    """

    optimizer: HexalyOptimizer
    model: HxModel
    _objective_expr: HxExpression | None = None  # To accumulate objective terms

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
        self._objective_expr = None

    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
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

        Returns
        -------
        pl.Series
            A Polars Object Series containing the created Hexaly variable objects.

        """
        hexaly_vars: list[HxExpression] = []  # Updated type hint
        current_objective_terms: list[HxExpression] = []  # Updated type hint

        match vtype:
            case VarType.CONTINUOUS:
                var_f = self.model.float
            case VarType.INTEGER:
                var_f = self.model.int
            case VarType.BINARY:
                var_f = lambda *_: self.model.bool  # noqa: E731

        for lb_, ub_, obj_, name_ in df.rows():
            (var := var_f(lb_, ub_)).set_name(name_)
            hexaly_vars.append(var)

            if obj_ != 0:
                current_objective_terms.append(obj_ * var)

        self.var_types[name] = vtype
        self.vars[name] = pl.Series(hexaly_vars, dtype=pl.Object)

        if current_objective_terms:
            # If there's an existing objective, add to it, otherwise start new
            new_objective_part = self.model.sum(current_objective_terms)
            if self._objective_expr is None:
                self._objective_expr = new_objective_part
            else:
                self._objective_expr = self.model.sum(self._objective_expr, new_objective_part)

            # Re-set the objective if it already exists, or set for the first time
            # Hexaly's model.minimize() and maximize() automatically handle re-setting
            self.model.minimize(self._objective_expr)  # Assuming minimization by default for Xplor

        return self.vars[name]

    def _add_constrs(self, df: pl.DataFrame, name: str, expr_repr: ExpressionRepr) -> pl.Series:
        """Return a series of Hexaly constraints.

        This method is called by `XplorModel.add_constrs` after the expression
        has been processed into rows of data and a constraint string.

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame containing the necessary components for the constraint expression.
        name : str
            The base name for the constraint.
        expr_repr : ExpressionRepr
            The evaluated string representation of the constraint expression.

        Returns
        -------
        pl.Series
            A Polars Object Series containing the namd of Hexaly constraint objects.

        """
        [self.model.add_constraint(expr_repr.evaluate(row)) for row in df.rows()]
        return pl.Series(
            name,
            [f"{name}[{i}]" for i in range(df.height)],
            dtype=pl.Object,
        )

    def optimize(self, time_limit: float | None = None) -> None:  # ty:ignore[invalid-method-override]
        """Solve the Hexaly model.

        Uses `hexaly.solve()` to solve the model.

        Parameters
        ----------
        time_limit : float | None, default None
            An optional time limit in seconds for the solver. If None,
            Hexaly's default time limit is used.

        """
        if self._objective_expr is None:
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

        """
        sol: HxSolution = self.optimizer.get_solution()
        status: HxSolutionStatus = sol.get_status()
        if status in (HxSolutionStatus.INCONSISTENT, HxSolutionStatus.INFEASIBLE):
            msg = f"The Hexaly model status is {status}."
            raise Exception(msg)

        if self._objective_expr is not None:
            return self._objective_expr.value
        else:
            msg = "At least one objective is required in the model."
            raise Exception(msg)

    def get_variable_values(self, name: str) -> pl.Series:
        """Read the optimal values of a variable series from the Hexaly solution.

        The method ensures the returned Polars Series has the correct data type
        (Float64 for continuous, Int64 for integer/binary).

        Parameters
        ----------
        name : str
            The base name used when the variable series was created with `xmodel.add_vars()`.

        Returns
        -------
        pl.Series
            A Polars Series containing the optimal variable values.

        """
        return cast_to_dtypes(
            pl.Series(name, [v.value for v in self.vars[name]]), self.var_types[name]
        )
