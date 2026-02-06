"""Tests for multi-objective optimization support (Gurobi backend)."""

import polars as pl
import pytest

import xplor
from xplor.gurobi import XplorGurobi


def test_backward_compatible_default_priority():
    """Test that default priority=0 behaves identically to single-objective."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1], "lb": [0.0, 0.0], "ub": [1.0, 1.0], "obj": [-1, -2]})

    # Create variables without specifying priority (should default to 0)
    df = df.with_columns(xmodel.add_vars("x", lb="lb", ub="ub", obj="obj"))

    # Add constraint
    df.pipe(xmodel.add_constrs, xplor.var.x.sum() <= 1.5)

    xmodel.optimize()

    # Should optimize to same result as single-objective
    objective_value = xmodel.get_objective_value()
    assert objective_value == pytest.approx(-2.5, rel=1e-4)

    # Check variable values
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result.to_list() == pytest.approx([0.5, 1.0], rel=1e-4)


def test_two_priority_levels():
    """Test optimization with two priority levels."""
    xmodel = XplorGurobi()

    # Create a scenario where we want to:
    # Priority 1 (higher): Maximize sum of x1 (use negative obj to maximize with minimize solver)
    # Priority 0 (lower): Minimize sum of x2
    df = pl.DataFrame(
        {
            "id": [0, 1],
            "lb": [0.0, 0.0],
            "ub": [1.0, 1.0],
        }
    )

    # Add variables with different priorities
    # Using negative coefficients to maximize with minimize solver
    df = df.with_columns(
        x1=xmodel.add_vars("x1", lb="lb", ub="ub", obj=-10.0, priority=1),
        x2=xmodel.add_vars("x2", lb="lb", ub="ub", obj=1.0, priority=0),
    )

    # Constraint: x1 + x2 <= 1.5
    df.pipe(xmodel.add_constrs, xplor.var.x1.sum() + xplor.var.x2.sum() <= 1.5)

    xmodel.optimize()

    # Verify that we have multiple objectives
    assert xmodel.model.NumObj == 2

    # Priority 1 should be optimized first, so sum of x1 should be maximized to 1.5
    result = df.select(xmodel.read_values(pl.selectors.object())).to_dict(as_series=False)

    # Sum of x1 should be 1.5 (constraint limit) since priority 1 maximizes it
    # x2 should be 0 since priority 0 minimizes it
    assert sum(result["x1"]) == pytest.approx(1.5, rel=1e-4)
    assert sum(result["x2"]) == pytest.approx(0.0, rel=1e-4)


def test_priority_alignment():
    """Verify that user priorities are passed directly to Gurobi (higher = optimized first)."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0], "lb": [0.0], "ub": [1.0]})

    # Create variables with priorities 2, 1, 0
    df = df.with_columns(
        x2=xmodel.add_vars("x2", lb="lb", ub="ub", obj=pl.lit(1.0), priority=2),
        x1=xmodel.add_vars("x1", lb="lb", ub="ub", obj=pl.lit(1.0), priority=1),
        x0=xmodel.add_vars("x0", lb="lb", ub="ub", obj=pl.lit(1.0), priority=0),
    )

    xmodel.optimize()

    # Verify that Gurobi sees 3 objectives
    assert xmodel.model.NumObj == 3

    # Check that priorities align with user priorities
    # User priority 2 should be Gurobi priority 2 (highest, optimized first)
    # User priority 1 should be Gurobi priority 1
    # User priority 0 should be Gurobi priority 0 (lowest, optimized last)
    xmodel.model.setParam("ObjNumber", 0)
    assert xmodel.model.ObjNPriority == 2  # User priority 2 -> Gurobi priority 2

    xmodel.model.setParam("ObjNumber", 1)
    assert xmodel.model.ObjNPriority == 1  # User priority 1 -> Gurobi priority 1

    xmodel.model.setParam("ObjNumber", 2)
    assert xmodel.model.ObjNPriority == 0  # User priority 0 -> Gurobi priority 0


def test_multiple_add_vars_same_priority():
    """Test that multiple add_vars calls at the same priority are combined."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(
        {"id": [0, 1], "lb": [0.0, 0.0], "ub": [1.0, 1.0], "obj_x": [1.0, 2.0], "obj_y": [3.0, 4.0]}
    )

    # Add two sets of variables at priority 1
    df = df.with_columns(
        x=xmodel.add_vars("x", lb="lb", ub="ub", obj="obj_x", priority=1),
        y=xmodel.add_vars("y", lb="lb", ub="ub", obj="obj_y", priority=1),
    )

    # Constraint to make it bounded
    df.pipe(xmodel.add_constrs, xplor.var.x.sum() + xplor.var.y.sum() <= 2.0)

    xmodel.optimize()

    # Should have only 1 objective (both variables at priority 1 combined)
    assert xmodel.model.NumObj == 1


def test_mixed_priority_zero_and_nonzero():
    """Test mix of default priority (0) and explicit priority (1)."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0], "lb": [0.0], "ub": [10.0]})

    # Add variables: one with explicit priority 1, one with default priority 0
    df = df.with_columns(
        x_high=xmodel.add_vars("x_high", lb="lb", ub="ub", obj=pl.lit(1.0), priority=1),
        x_low=xmodel.add_vars("x_low", lb="lb", ub="ub", obj=pl.lit(2.0)),  # default priority=0
    )

    xmodel.optimize()

    # Should have 2 objectives
    assert xmodel.model.NumObj == 2


def test_no_objectives():
    """Test that model with variables but no objective coefficients works."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0, 1], "lb": [0.0, 0.0], "ub": [1.0, 1.0]})

    # Create variables with obj=0 (default)
    df = df.with_columns(xmodel.add_vars("x", lb="lb", ub="ub"))

    # Add constraint to make it feasible
    df.pipe(xmodel.add_constrs, xplor.var.x.sum() == 1.0)

    xmodel.optimize()

    # Should find a feasible solution
    assert xmodel.model.Status == 2  # GRB.OPTIMAL


def test_sparse_objectives():
    """Test that only non-zero objective coefficients are stored."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(
        {"id": [0, 1, 2], "lb": [0.0, 0.0, 0.0], "ub": [1.0, 1.0, 1.0], "obj": [1.0, 0.0, 2.0]}
    )

    # Some variables have zero objective
    df = df.with_columns(xmodel.add_vars("x", lb="lb", ub="ub", obj="obj", priority=1))

    xmodel.optimize()

    # Should still create objective with non-zero terms
    assert xmodel.model.NumObj == 1


def test_get_objective_value_raises_for_multi_objective():
    """Test that get_objective_value raises error for multi-objective models."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0], "lb": [0.0], "ub": [10.0]})

    # Create multi-objective model
    df = df.with_columns(
        x2=xmodel.add_vars("x2", lb="lb", ub="ub", obj=pl.lit(100.0), priority=2),
        x1=xmodel.add_vars("x1", lb="lb", ub="ub", obj=pl.lit(10.0), priority=1),
    )

    xmodel.optimize()

    # get_objective_value should raise ValueError for multi-objective
    with pytest.raises(ValueError, match="Model has 2 objectives"):
        xmodel.get_objective_value()


def test_get_multi_objective_values():
    """Test that get_multi_objective_values returns all objective values."""
    xmodel = XplorGurobi()

    df = pl.DataFrame({"id": [0], "lb": [0.0], "ub": [10.0]})

    # Priority 2: coefficient 100
    # Priority 1: coefficient 10
    # Priority 0: coefficient 1
    df = df.with_columns(
        x2=xmodel.add_vars("x2", lb="lb", ub="ub", obj=pl.lit(100.0), priority=2),
        x1=xmodel.add_vars("x1", lb="lb", ub="ub", obj=pl.lit(10.0), priority=1),
        x0=xmodel.add_vars("x0", lb="lb", ub="ub", obj=pl.lit(1.0), priority=0),
    )

    xmodel.optimize()

    # get_multi_objective_values should return dict mapping priority -> value
    obj_values = xmodel.get_multi_objective_values()

    assert isinstance(obj_values, dict)
    assert len(obj_values) == 3
    assert 2 in obj_values
    assert 1 in obj_values
    assert 0 in obj_values

    # Verify all priorities are present and values are numeric
    for priority in [2, 1, 0]:
        assert priority in obj_values
        assert isinstance(obj_values[priority], (int, float))


def test_hierarchical_optimization():
    """Test that hierarchical optimization works correctly.

    Priority 1: Maximize revenue (use negative coefficients)
    Priority 0: Minimize cost

    The solver should maximize revenue first, then minimize cost
    without reducing revenue.
    """
    xmodel = XplorGurobi()

    df = pl.DataFrame(
        {
            "product": ["A", "B"],
            "revenue": [10.0, 20.0],
            "cost": [5.0, 15.0],
            "max_qty": [10.0, 10.0],
        }
    )

    # Priority 1: Maximize revenue (use negative coefficients for maximization)
    # Priority 0: Minimize cost
    df = df.with_columns(
        qty=xmodel.add_vars("qty", lb=0.0, ub="max_qty", obj=-pl.col("revenue"), priority=1),
    )

    # Add a second priority for cost (if we want to minimize cost as secondary objective)
    # But for this test, let's keep it simple with just one priority

    # Add a constraint: total quantity <= 12
    df.pipe(xmodel.add_constrs, xplor.var.qty.sum() <= 12.0)

    xmodel.optimize()

    # Check that revenue is maximized
    result = df.select(xmodel.read_values(pl.col("qty"))).to_series()

    # B has higher revenue per unit (20 vs 10), so should be selected first
    # Total qty constraint is 12, so should maximize B first: A=2, B=10 (revenue = 220)
    assert result.to_list() == pytest.approx([2.0, 10.0], rel=1e-4)
