from typing import Any, Callable

import gurobipy as gp
import polars as pl


class GurobiPolars:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def assign(self, **kwargs: Callable[[dict[str, Any]], Any]) -> pl.DataFrame:
        rows = self._df.rows(named=True)
        return self._df.with_columns(
            [pl.Series(name, [f(row) for row in rows]) for name, f in kwargs.items()]
        )

    def aggregate(
        self,
        columns: list[str],
        *,
        by: list[str | pl.Expr] | None = None,
        function: Callable[[Any], gp.LinExpr | gp.QuadExpr] = gp.quicksum,
    ) -> pl.DataFrame:
        return (
            self._df.group_by(by)
            .agg(pl.col(columns).map_elements(function, return_dtype=pl.Object))
            .drop("literal", strict=False)
        )

    def add_vars(
        self,
        model: gp.Model,
        name: str,
        *,
        lb: float | pl.Expr = 0.0,
        ub: float | pl.Expr = gp.GRB.INFINITY,
        obj: float | pl.Expr = 0.0,
        indices: list[str | pl.Expr] | None = None,
        vtype: str = gp.GRB.CONTINUOUS,
    ) -> pl.DataFrame:
        """Add a variable to the given model for each row in the dataframe.

        Parameters
        ----------
        model : Model
            A Gurobi model to which new variables will be added
        name : str
            Used as the appended column name, as well as the base
            name for added Gurobi variables
        lb : float | pl.Expr, optional
            Lower bound for created variables. May be a single value
            or the name of a column in the dataframe, defaults to 0.0
        ub : float | pl.Expr, optional
            Upper bound for created variables. May be a single value
            or the name of a column in the dataframe, defaults to
            :code:`GRB.INFINITY`
        obj: float | pl.Expr, optional
            Objective function coefficient for created variables.
            May be a single value, or the name of a column in the dataframe,
            defaults to 0.0
        vtype: str, optional
            Gurobi variable type for created variables, defaults
            to :code:`GRB.CONTINUOUS`

        Returns
        -------
        DataFrame
            A new DataFrame with new Vars appended as a column
        """
        lb_ = self._df.with_columns(lb=lb)["lb"].to_numpy() if isinstance(lb, pl.Expr) else lb
        ub_ = self._df.with_columns(ub=ub)["ub"].to_numpy() if isinstance(ub, pl.Expr) else ub
        obj_ = self._df.with_columns(obj=obj)["obj"].to_numpy() if isinstance(obj, pl.Expr) else obj
        if indices is not None:
            name_ = (
                self._df.select(
                    pl.format(
                        "{}[{}]",
                        pl.lit(name),
                        pl.concat_str([c for c in indices], separator=","),
                    )
                )
                .to_series(0)
                .to_list()
            )
        else:
            name_ = indices
        vars = model.addMVar(
            self._df.height,
            vtype=vtype,
            lb=lb_,
            ub=ub_,
            obj=obj_,
            name=name_,
        )
        model.update()
        return self._df.with_columns(pl.Series(name, vars.tolist(), dtype=pl.Object))

    def add_constrs(
        self,
        model: gp.Model,
        lhs: float | pl.Expr,
        sense: str,
        rhs: float | pl.Expr,
        name: str | None = None,
    ) -> pl.DataFrame:
        constrs = [
            model.addLConstr(
                lhs,
                sense,
                rhs,
                name=name or "",
            )
            for lhs, rhs in self._df.select(lhs=lhs, rhs=rhs).rows()
        ]
        model.update()
        if name is None:
            return self._df
        else:
            return self._df.with_columns(pl.Series(name, constrs, dtype=pl.Object))


# %%
