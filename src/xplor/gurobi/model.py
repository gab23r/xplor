from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import gurobipy as gp
import polars as pl

from xplor.gurobi.var import (
    GurobiVarExpr,
    _ProxyGurobiVarExpr,
    first_gurobi_expr,
    to_mvar_or_mlinexpr,
)
from xplor.model import XplorModel
from xplor.types import cast_to_dtypes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gurobipy import TempGenConstr, TempLConstr

    from xplor.exprs import ConstrExpr
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

    def add_constrs(
        self,
        df: pl.DataFrame,
        *constr_exprs: ConstrExpr,
        indices: pl.Expr | list[str] | None = None,
        **named_constr_exprs: ConstrExpr,
    ) -> pl.DataFrame:
        """Add constraints using optimized Gurobi MVar evaluation.

        Overrides base class to use direct Gurobi MVar objects when possible, avoiding
        the overhead of the VarExpr materialization system. This provides ~20x speedup
        for simple constraint patterns.

        The optimization is used when:
        - No custom indices are specified
        - Constraints don't have multiple outputs (e.g., no var("x", "y"))
        - Constraints don't use name modifiers (e.g., .name.suffix())

        For complex cases, falls back to the base class implementation to ensure
        correct behavior.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing variable columns and data
        *constr_exprs
            Constraint expressions (positional arguments)
        indices : pl.Expr | list[str] | None
            Optional indices for constraint naming
        **named_constr_exprs
            Named constraint expressions (keyword arguments)

        Returns
        -------
        pl.DataFrame
            The input DataFrame (unchanged)

        Examples
        --------
        >>> xmodel = XplorGurobi()
        >>> df = pl.DataFrame(height=1000000).with_columns(
        ...     xmodel.add_vars("x"),
        ...     xmodel.add_vars("y"),
        ...     w=10
        ... )
        >>> xmodel.add_constrs(df, var.x == var.y + pl.col("w") * 2)

        Notes
        -----
        The optimized path works with VarExpr (var.x) and Polars expressions.
        Complex cases automatically fall back to the base class implementation.

        """
        from xplor.exprs import VarExpr

        # Check if we can use the optimized path
        # Fall back to base class if:
        # - Custom indices are specified (need proper naming)
        # - Any constraint has multiple outputs
        # - Any constraint contains non-vectorizable expressions (GenExpr, QuadExpr, NLExpr)
        all_exprs = list(constr_exprs) + list(named_constr_exprs.values())

        # Check for multiple outputs
        if any(expr.meta.has_multiple_outputs() for expr in all_exprs):  # indices is not None or
            return super().add_constrs(df, *constr_exprs, indices=indices, **named_constr_exprs)

        # Check for non-vectorizable expressions by doing a quick evaluation
        for constr_expr in all_exprs:
            expr_repr, exprs = constr_expr.parse()
            # Check if any column contains GenExpr, QuadExpr, or NLExpr
            for expr in exprs:
                if isinstance(expr, VarExpr):
                    col_name = expr.meta.output_name()
                    first_val = first_gurobi_expr(df[col_name])
                    if isinstance(first_val, (gp.GenExpr, gp.QuadExpr, gp.NLExpr)):
                        # Contains non-vectorizable expressions, use base class
                        return super().add_constrs(
                            df, *constr_exprs, indices=indices, **named_constr_exprs
                        )

        # Helper function to recursively evaluate expressions
        def evaluate_expr(expr: Any) -> Any:
            """Recursively evaluate VarExpr, pl.Expr, or literal values to Gurobi objects."""
            if isinstance(expr, VarExpr):
                if expr._nodes:
                    # VarExpr contains operations - recursively evaluate sub-expressions
                    sub_expr_repr, sub_exprs = expr.parse()
                    sub_gp_arrays = [evaluate_expr(sub_expr) for sub_expr in sub_exprs]
                    return sub_expr_repr.evaluate(tuple(sub_gp_arrays))
                # Simple variable reference without operations
                col_name = expr.meta.output_name()
                return to_mvar_or_mlinexpr(df[col_name])
            if isinstance(expr, pl.Expr):
                # Polars expression: evaluate and convert
                result = df.select(expr).to_series()
                return to_mvar_or_mlinexpr(result)
            # Literal value
            return expr

        # Optimized path: process constraints directly
        # Combine positional and named constraints
        all_constrs = {str(expr): expr for expr in constr_exprs}
        all_constrs.update(named_constr_exprs)

        for constr_name, constr_expr in all_constrs.items():
            # Parse the constraint to get operator and operands
            expr_repr, exprs = constr_expr.parse()
            # Evaluate each sub-expression to Gurobi objects (with recursion)
            gp_arrays = [evaluate_expr(expr) for expr in exprs]

            # Evaluate the constraint using the expression representation
            gp_constr = expr_repr.evaluate(tuple(gp_arrays))

            # Add to model
            self._add_constr(gp_constr, name=constr_name)

        return df

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
            # Handle gp.GenExpr, gp.QuadExpr, and gp.NLExpr - these require row-by-row processing
            rhs_idx = constr_repr.extract_indices(side="rhs")
            if rhs_idx:
                first_val = first_gurobi_expr(df[:, rhs_idx[0]])
                # Check if RHS contains non-vectorizable expressions
                if isinstance(first_val, (gp.GenExpr, gp.QuadExpr, gp.NLExpr)):
                    for row, idx in zip(df.rows(), indices, strict=True):
                        self._add_constr(constr_repr.evaluate(row), name=f"{name}[{idx}]")
                    continue

            # Vectorized path for linear constraints
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


def sum_by(
    df: pl.DataFrame,
    *args: GurobiVarExpr,
    by: str | list[str],
    **kwargs: GurobiVarExpr,
) -> pl.DataFrame:
    """Group and aggregate with Gurobi matrix API, including grouping columns.

    Automatically applies `.sum_by(by)` to GurobiVarExpr expressions,
    eliminating the need to repeat grouping columns in each aggregation.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with variables
    by : str | list[str]
        Column(s) to group by
    *args
        Positional aggregations
    **kwargs
        Named aggregations

    Returns
    -------
    pl.DataFrame
        DataFrame with grouping columns + aggregated columns

    Examples
    --------
    >>> from xplor.gurobi import sum_by, var
    >>> # Positional arguments with auto-naming (cleanest!)
    >>> df.pipe(sum_by, var.x, by=["w"])
    >>> # Result columns: ["w", "x"]
    >>>
    >>> # Multiple positional arguments
    >>> df.pipe(
    ...     sum_by,
    ...     var.x,
    ...     var.y,
    ...     var.x * pl.col("coeff"),
    ...     by=["w"],
    ... )
    >>> # Result columns: ["w", "x", "y", "x * coeff"]
    >>>
    >>> # Named arguments for custom column names
    >>> df_grouped = sum_by(
    ...     df,
    ...     x_sum=var.x,  # Automatically sum_by(["ata_group", "week"])
    ...     weighted=var.x * pl.col("w"),  # Also automatic
    ...     by=["ata_group", "week"],
    ... )
    >>>
    >>> # Can also use explicit .sum_by() if needed
    >>> df_grouped = sum_by(
    ...     df,
    ...     x_sum=var.x.sum_by(["w"]),
    ...     by=["w"],
    ... )
    >>>
    >>> # Can join on grouping columns
    >>> df_grouped.join(other_df, on=["ata_group", "week"])

    """
    # Get unique group combinations
    by_cols = [by] if isinstance(by, str) else by
    df_groups = df.select(*[c if isinstance(c, pl.Expr) else pl.col(c) for c in by_cols]).unique(
        maintain_order=True
    )

    sum_by_exprs = [expr.sum_by(by) for expr in args] + [
        arg.sum_by(by).alias(name) for name, arg in kwargs.items()
    ]

    return pl.concat([df_groups, df.select(sum_by_exprs)], how="horizontal")
