import gurobipy as gp
import polars as pl

pytest_plugins = ["utils"]


def test_gurobi_model(model: gp.Model):
    df = pl.DataFrame(
        data=["aaa", "bbb", "ccc", "aaa", "bbb", "ccc"],
        schema=[("txt", pl.String)],
    ).with_columns(a=pl.int_range(pl.len()).cast(pl.Float64))
    df.gp.add_vars(model, "t", ub=pl.col.a)
