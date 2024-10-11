import gurobipy as gp
import polars as pl
import xplor.gurobi as pg


def test_gurobi_model(model: gp.Model):
    df = pl.DataFrame(
        {
            "i": [0, 0, 1, 2, 2],
            "j": [1, 2, 0, 0, 1],
            "u": [0.3, 1.2, 0.7, 0.9, 1.2],
            "c": [1.3, 1.7, 1.4, 1.1, 0.9],
            "obj": [2.5, 2.7, 1.2, 1.7, 3.9],
        }
    )

    df = (
        df.pipe(
            pg.add_vars,
            model,
            name="x",
            ub="u",
            obj="obj",
            indices=["i", "j"],
            vtype=gp.GRB.CONTINUOUS,
        )
        .pipe(pg.apply_eval, "y = 2 * x - c")
        .group_by("i")
        .agg(pg.quicksum("y"), pl.col("c").min())
        .pipe(pg.add_constrs, model, "y <= c", name="constr")
    )
