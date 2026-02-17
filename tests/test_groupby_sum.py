"""Test GurobiVarExpr.sum() with by parameter using matrix API."""

import gurobipy as gp
import polars as pl
import pytest

import xplor
from xplor.gurobi import XplorGurobi


def test_simple_groupby_sum():
    """Test var.x.sum_by("w")."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=100).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        w=pl.int_range(0, 100) // 10,  # 10 groups of 10
    )

    # Compute grouped sum - returns one row per group (not broadcast)
    result = df.select(group_sum=xmodel.var.x.sum_by("w"))

    # Should return one row per group
    assert len(result) == 10  # 10 groups, not 100 rows
    assert result["group_sum"].dtype == pl.Object


def test_weighted_groupby_sum():
    """Test (var.x * coeff).sum_by("w")."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=100).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        w=pl.int_range(0, 100) // 10,
        coeff=pl.int_range(0, 100) % 5 + 1.0,  # Coefficients 1-5
    )

    # Weighted grouped sum - returns one row per group
    result = df.select(weighted_sum=(xmodel.var.x * pl.col("coeff")).sum_by("w"))

    assert len(result) == 10  # 10 groups
    assert result["weighted_sum"].dtype == pl.Object


def test_groupby_sum_in_constraints():
    """Test using grouped sum in optimization constraints."""
    xmodel = XplorGurobi()
    xmodel.model.setParam("OutputFlag", 0)

    df = pl.DataFrame(height=100).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        w=pl.int_range(0, 100) // 10,
    )

    # Get group-level sums (one per group) - select returns aggregated results
    df_grouped = df.select(group_sum=xmodel.var.x.sum_by("w"))

    # Add constraint: each group sum == 50
    xmodel.add_constrs(df_grouped, group_constraint=xplor.var.group_sum == 50)

    # Set objective: minimize sum of x
    xmodel.model.setObjective(gp.MVar(df["x"]).sum(), sense=gp.GRB.MINIMIZE)  # ty:ignore[too-many-positional-arguments]

    xmodel.optimize()

    # Check solution
    x_values = df.select(xmodel.read_values(pl.col("x"))).to_series()

    # Each group of 10 should sum to 50, so average should be 5.0
    for group_id in range(10):
        group_vals = x_values[group_id * 10 : (group_id + 1) * 10]
        assert group_vals.sum() == pytest.approx(50.0, rel=1e-4)


def test_groupby_sum_complex_expression():
    """Test (var.x + var.y).sum_by("w")."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=50).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        xmodel.add_vars("y", lb=0, ub=10),
        w=pl.int_range(0, 50) // 5,  # 10 groups of 5
    )

    # Complex expression with groupby - returns one row per group
    result = df.select(sum_xy=(xmodel.var.x + xmodel.var.y).sum_by("w"))

    assert len(result) == 10  # 10 groups
    assert result["sum_xy"].dtype == pl.Object


def test_groupby_sum_multi_column():
    """Test groupby with multiple columns using list."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=100).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        w=pl.int_range(0, 100) // 10,
        region=pl.int_range(0, 100) % 5,
    )

    # Group by multiple columns using list - returns one row per group
    result = df.select(group_sum=xmodel.var.x.sum_by(["w", "region"]))

    # Should have 10 * 5 = 50 unique groups
    assert len(result) == 50
    assert result["group_sum"].dtype == pl.Object


def test_groupby_sum_comparison_with_polars():
    """Compare results with actual Polars groupby for verification."""
    xmodel = XplorGurobi()
    xmodel.model.setParam("OutputFlag", 0)

    df = pl.DataFrame(height=30).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        w=pl.int_range(0, 30) // 10,
    )

    # Set fixed values for x
    for i, x_val in enumerate([1.0, 2.0, 3.0] * 10):
        xmodel.add_constrs(
            df.filter(pl.int_range(0, 30) == i),
            constr=xplor.var.x == x_val,
        )

    xmodel.optimize()

    # Get solution values
    df = df.with_columns(x_val=xmodel.read_values(pl.col("x")))

    # Compute using Polars groupby
    polars_sums = (
        df.group_by("w").agg(pl.col("x_val").sum().alias("sum")).sort("w")["sum"].to_list()
    )

    # Pattern is [1,2,3] * 10, so:
    # Group 0 (rows 0-9): [1,2,3,1,2,3,1,2,3,1] = 19
    # Group 1 (rows 10-19): [2,3,1,2,3,1,2,3,1,2] = 20
    # Group 2 (rows 20-29): [3,1,2,3,1,2,3,1,2,3] = 21
    expected_sums = [19.0, 20.0, 21.0]
    assert polars_sums == pytest.approx(expected_sums, rel=1e-4)
