import gurobipy as gp
import polars as pl

from xplor.model import XplorModel
from xplor.var import VarType, cast_to_dtypes


class XplorGurobi(XplorModel):
    """Xplor base class to wrap your Gurobi model."""

    model: gp.Model

    def __init__(self, model: gp.Model | None = None) -> None:
        """Initialize the model wrapper.

        Concrete implementations must handle model initialization and setup.
        """
        model = gp.Model() if model is None else model
        super().__init__(model=model)

    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
    ) -> pl.Series:
        """Return a series of variables.

        `df` should contains columns: ["lb", "ub", "obj", "name"].
        """
        self.var_types[name] = vtype
        self.vars[name] = pl.Series(
            self.model.addMVar(
                df.height,
                vtype=getattr(gp.GRB, vtype),
                lb=df["lb"].to_numpy(),
                ub=df["ub"].to_numpy(),
                obj=df["obj"].to_numpy(),
                name=df["name"].to_list(),
            ).tolist(),
            dtype=pl.Object(),
        )
        self.model.update()
        return self.vars[name]

    def _add_constrs(self, df: pl.DataFrame, name: str, expr_str: str) -> pl.Series:
        """Return a series of variables."""
        # TODO: manage non linear constraint
        # https://github.com/gab23r/xplor/issues/1

        if df.height == 0:
            return pl.Series(name, dtype=pl.Object)

        # row = df.row(0)
        # lhs_constr_type = str(type(row[0]))
        # rhs_constr_type = str(type(row[1]))
        # if "GenExpr" in lhs_constr_type or "GenExpr" in rhs_constr_type:
        #     _add_constr = self.model.addConstr
        # elif "QuadExpr" in lhs_constr_type or "QuadExpr" in rhs_constr_type:
        #     _add_constr = self.model.addQConstr
        # else:
        #     _add_constr = self.model.addLConstr

        _add_constr = self.model.addLConstr
        series = pl.Series(
            [_add_constr(eval(expr_str), name=name) for d in df.rows()], dtype=pl.Object
        )
        self.model.update()
        return series

    def optimize(self, solver_type: None = None) -> None:
        """Solve the model.

        solver_type is ignored for Gurobi models.
        """
        self.model.optimize()

    def get_objective_value(self) -> float:
        """Return the objective value."""
        return self.model.getObjective().getValue()

    def get_variable_values(self, name: str) -> pl.Series:
        """Read the value of a variables."""
        return cast_to_dtypes(pl.Series([e.x for e in self.vars[name]]), self.var_types[name])
