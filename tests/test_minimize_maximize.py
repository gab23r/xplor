"""Tests for minimize and maximize methods."""

import polars as pl
import pytest

import xplor
from xplor.gurobi import XplorGurobi


def test_minimize_single_objective():
    """Test minimize with a single objective expression."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1, 2], "cost": [1.0, 2.0, 3.0]})

    # Create variables
    df = df.with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))

    # Add objective using minimize
    df.pipe(
        xmodel.minimize,
        total_cost=(xplor.var("x") * pl.col("cost")).sum(),
    )

    # Add constraint: sum of x >= 5
    df.pipe(xmodel.add_constrs, xplor.var("x").sum() >= 5.0)

    xmodel.optimize()

    # Should minimize cost while meeting constraint
    # Optimal: x[0]=5, x[1]=0, x[2]=0 (cost = 5)
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result[0] == pytest.approx(5.0, rel=1e-4)
    assert result[1] == pytest.approx(0.0, rel=1e-4)
    assert result[2] == pytest.approx(0.0, rel=1e-4)


def test_maximize_single_objective():
    """Test maximize with a single objective expression."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1, 2], "revenue": [1.0, 2.0, 3.0]})

    # Create variables
    df = df.with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))

    # Add objective using maximize
    df.pipe(
        xmodel.maximize,
        total_revenue=(xplor.var("x") * pl.col("revenue")).sum(),
    )

    # Add constraint: sum of x <= 5
    df.pipe(xmodel.add_constrs, xplor.var("x").sum() <= 5.0)

    xmodel.optimize()

    # Should maximize revenue while meeting constraint
    # Optimal: x[0]=0, x[1]=0, x[2]=5 (revenue = 15)
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result[0] == pytest.approx(0.0, rel=1e-4)
    assert result[1] == pytest.approx(0.0, rel=1e-4)
    assert result[2] == pytest.approx(5.0, rel=1e-4)


def test_minimize_aggregated_expression():
    """Test minimize with an aggregated expression like sum()."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1, 2]})

    # Create variables
    df = df.with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))

    # Add objective using minimize with sum
    df.pipe(
        xmodel.minimize,
        sum_x=xplor.var("x").sum(),
    )

    # Add constraint: sum of x >= 3
    df.pipe(xmodel.add_constrs, xplor.var("x").sum() >= 3.0)

    xmodel.optimize()

    # Should minimize sum while meeting constraint
    # Optimal: sum = 3 (distributed among variables)
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert sum(result) == pytest.approx(3.0, rel=1e-4)


def test_maximize_aggregated_expression():
    """Test maximize with an aggregated expression like sum()."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1, 2]})

    # Create variables
    df = df.with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))

    # Add objective using maximize with sum
    df.pipe(
        xmodel.maximize,
        sum_x=xplor.var("x").sum(),
    )

    # Add constraint: sum of x <= 15
    df.pipe(xmodel.add_constrs, xplor.var("x").sum() <= 15.0)

    xmodel.optimize()

    # Should maximize sum while meeting constraint
    # Optimal: sum = 15
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert sum(result) == pytest.approx(15.0, rel=1e-4)


def test_multi_objective_minimize_with_priority():
    """Test minimize with multiple priorities."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1], "cost1": [10.0, 20.0], "cost2": [5.0, 15.0]})

    # Create variables
    df = df.with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))

    # Priority 1: Minimize cost1
    # Priority 0: Minimize cost2
    df.pipe(
        xmodel.minimize,
        total_cost1=(xplor.var("x") * pl.col("cost1")).sum(),
        priority=1,
    )

    df.pipe(
        xmodel.minimize,
        total_cost2=(xplor.var("x") * pl.col("cost2")).sum(),
        priority=0,
    )

    # Add constraint: sum of x >= 5
    df.pipe(xmodel.add_constrs, xplor.var("x").sum() >= 5.0)

    xmodel.optimize()

    # Should minimize cost1 first (priority 1), then cost2 (priority 0)
    # Optimal for priority 1: x[0]=5, x[1]=0 (cost1 = 50)
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result[0] == pytest.approx(5.0, rel=1e-4)
    assert result[1] == pytest.approx(0.0, rel=1e-4)


def test_multi_objective_maximize_with_priority():
    """Test maximize with multiple priorities."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1], "revenue1": [10.0, 20.0], "revenue2": [5.0, 15.0]})

    # Create variables
    df = df.with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))

    # Priority 1: Maximize revenue1
    # Priority 0: Maximize revenue2
    df.pipe(
        xmodel.maximize,
        total_revenue1=(xplor.var("x") * pl.col("revenue1")).sum(),
        priority=1,
    )

    df.pipe(
        xmodel.maximize,
        total_revenue2=(xplor.var("x") * pl.col("revenue2")).sum(),
        priority=0,
    )

    # Add constraint: sum of x <= 5
    df.pipe(xmodel.add_constrs, xplor.var("x").sum() <= 5.0)

    xmodel.optimize()

    # Should maximize revenue1 first (priority 1)
    # Optimal for priority 1: x[0]=0, x[1]=5 (revenue1 = 100)
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result[0] == pytest.approx(0.0, rel=1e-4)
    assert result[1] == pytest.approx(5.0, rel=1e-4)


def test_mixed_minimize_maximize():
    """Test mixing minimize and maximize at different priorities."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1], "revenue": [10.0, 20.0], "cost": [5.0, 15.0]})

    # Create variables
    df = df.with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))

    # Priority 1: Maximize revenue
    # Priority 0: Minimize cost
    df.pipe(
        xmodel.maximize,
        total_revenue=(xplor.var("x") * pl.col("revenue")).sum(),
        priority=1,
    )

    df.pipe(
        xmodel.minimize,
        total_cost=(xplor.var("x") * pl.col("cost")).sum(),
        priority=0,
    )

    # Add constraint: sum of x <= 5
    df.pipe(xmodel.add_constrs, xplor.var("x").sum() <= 5.0)

    xmodel.optimize()

    # Should maximize revenue first (priority 1)
    # Optimal for priority 1: x[0]=0, x[1]=5 (revenue = 100)
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result[0] == pytest.approx(0.0, rel=1e-4)
    assert result[1] == pytest.approx(5.0, rel=1e-4)


def test_multiple_objectives_same_priority():
    """Test multiple objectives at the same priority level (should be combined)."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1, 2], "cost1": [1.0, 2.0, 3.0], "cost2": [3.0, 2.0, 1.0]})

    # Create variables
    df = df.with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))

    # Add two objectives at same priority (should be combined)
    df.pipe(
        xmodel.minimize,
        total_cost1=(xplor.var("x") * pl.col("cost1")).sum(),
        total_cost2=(xplor.var("x") * pl.col("cost2")).sum(),
        priority=1,
    )

    # Add constraint: sum of x >= 5
    df.pipe(xmodel.add_constrs, xplor.var("x").sum() >= 5.0)

    xmodel.optimize()

    # Should minimize the combined cost (cost1 + cost2) = [4, 4, 4] for each x
    # Since all have equal combined cost, any distribution with sum=5 is optimal
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert sum(result) == pytest.approx(5.0, rel=1e-4)


def test_method_chaining():
    """Test that minimize/maximize can be chained with other operations."""
    xmodel = XplorGurobi()

    df = (
        pl.DataFrame({"id": [0, 1, 2], "cost": [1.0, 2.0, 3.0]})
        .with_columns(xmodel.add_vars("x", lb=0.0, ub=10.0))
        .pipe(
            xmodel.minimize,
            total_cost=(xplor.var("x") * pl.col("cost")).sum(),
        )
        .pipe(xmodel.add_constrs, xplor.var("x").sum() >= 5.0)
    )

    xmodel.optimize()

    # Should work with method chaining
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result[0] == pytest.approx(5.0, rel=1e-4)
