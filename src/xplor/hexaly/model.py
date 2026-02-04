from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from hexaly.optimizer import HexalyOptimizer, HxModel, HxSolution, HxSolutionStatus

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes

if TYPE_CHECKING:
    from hexaly.modeler import HxExpression


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
                var_f = lambda *_: self.model.bool()  # noqa: E731

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
        status: type[HxSolutionStatus] = sol.get_status()
        if status in (HxSolutionStatus.INCONSISTENT, HxSolutionStatus.INFEASIBLE):
            msg = f"The Hexaly model status is {status}."
            raise Exception(msg)

        if self._objective_expr is not None:
            return self._objective_expr.value
        else:
            msg = "At least one objective is required in the model."
            raise Exception(msg)

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
