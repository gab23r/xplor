"""Tests for multi-objective optimization support (OR-Tools MathOpt backend).

NOTE: Multi-objective optimization in OR-Tools MathOpt is only supported by
commercial solvers (GUROBI, CPLEX). Tests requiring GUROBI are skipped when
TOKENSERVER environment variable is not set.
"""

import os

import polars as pl
import pytest
from ortools.math_opt.python import mathopt

import xplor
from xplor.mathopt import XplorMathOpt

# Skip GUROBI tests if TOKENSERVER is not set (no Gurobi license available)
skip_if_no_gurobi = pytest.mark.skipif(
    "TOKENSERVER" not in os.environ,
    reason="MathOpt GUROBI tests require TOKENSERVER environment variable",
)


def test_backward_compatible_default_priority():
    """Test that default priority=0 behaves identically to single-objective."""
    xmodel = XplorMathOpt()

    df = pl.DataFrame({"id": [0, 1], "lb": [0.0, 0.0], "ub": [1.0, 1.0], "obj": [-1, -2]})

    # Create variables without specifying priority (should default to 0)
    df = df.with_columns(xmodel.add_vars("x", lb="lb", ub="ub", obj="obj"))

    # Add constraint
    df.pipe(xmodel.add_constrs, xplor.var.x.sum() <= 1.5)

    xmodel.optimize(solver_type=mathopt.SolverType.GLOP)

    # Should optimize to same result as single-objective
    objective_value = xmodel.get_objective_value()
    assert objective_value == pytest.approx(-2.5, rel=1e-4)

    # Check variable values
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result.to_list() == pytest.approx([0.5, 1.0], rel=1e-4)


@skip_if_no_gurobi
def test_two_priority_levels():
    """Test MathOpt optimization with two priority levels."""
    xmodel = XplorMathOpt()

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

    xmodel.optimize(solver_type=mathopt.SolverType.GUROBI)

    # Verify that we have multiple objectives (1 primary + 1 auxiliary)
    assert len(list(xmodel.model.auxiliary_objectives())) == 1

    # Priority 1 should be optimized first, so sum of x1 should be maximized to 1.5
    result = df.select(xmodel.read_values(pl.selectors.object())).to_dict(as_series=False)

    # Sum of x1 should be 1.5 (constraint limit) since priority 1 maximizes it
    # x2 should be 0 since priority 0 minimizes it
    assert sum(result["x1"]) == pytest.approx(1.5, rel=1e-4)
    assert sum(result["x2"]) == pytest.approx(0.0, rel=1e-4)


@skip_if_no_gurobi
def test_priority_alignment():
    """Verify that user priorities are correctly inverted for MathOpt (higher = optimized first)."""
    xmodel = XplorMathOpt()

    df = pl.DataFrame({"id": [0], "lb": [0.0], "ub": [1.0]})

    # Create variables with priorities 2, 1, 0
    df = df.with_columns(
        x2=xmodel.add_vars("x2", lb="lb", ub="ub", obj=pl.lit(1.0), priority=2),
        x1=xmodel.add_vars("x1", lb="lb", ub="ub", obj=pl.lit(1.0), priority=1),
        x0=xmodel.add_vars("x0", lb="lb", ub="ub", obj=pl.lit(1.0), priority=0),
    )

    xmodel.optimize(solver_type=mathopt.SolverType.GUROBI)

    # Verify that MathOpt sees 3 objectives (1 primary + 2 auxiliary)
    aux_objectives = list(xmodel.model.auxiliary_objectives())
    assert len(aux_objectives) == 2

    # Check that auxiliary objectives have correct MathOpt priorities
    # User priority 2 -> MathOpt priority 0 (primary objective, optimized first)
    # User priority 1 -> MathOpt priority 1 (auxiliary, optimized second)
    # User priority 0 -> MathOpt priority 2 (auxiliary, optimized third)
    aux_priorities = sorted([aux.priority for aux in aux_objectives])
    assert aux_priorities == [1, 2]


def test_multiple_add_vars_same_priority():
    """Test that multiple add_vars calls at the same priority are combined."""
    xmodel = XplorMathOpt()

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

    xmodel.optimize(solver_type=mathopt.SolverType.GLOP)

    # Should have only 1 objective (both variables at priority 1 combined into primary)
    assert len(list(xmodel.model.auxiliary_objectives())) == 0


@skip_if_no_gurobi
def test_mixed_priority_zero_and_nonzero():
    """Test mix of default priority (0) and explicit priority (1)."""
    xmodel = XplorMathOpt()

    df = pl.DataFrame({"id": [0], "lb": [0.0], "ub": [10.0]})

    # Add variables: one with explicit priority 1, one with default priority 0
    df = df.with_columns(
        x_high=xmodel.add_vars("x_high", lb="lb", ub="ub", obj=pl.lit(1.0), priority=1),
        x_low=xmodel.add_vars("x_low", lb="lb", ub="ub", obj=pl.lit(2.0)),  # default priority=0
    )

    xmodel.optimize(solver_type=mathopt.SolverType.GUROBI)

    # Should have 2 objectives (1 primary + 1 auxiliary)
    assert len(list(xmodel.model.auxiliary_objectives())) == 1


@skip_if_no_gurobi
def test_get_objective_value_raises_for_multi_objective():
    """Test that get_objective_value raises error for multi-objective models."""
    xmodel = XplorMathOpt()

    df = pl.DataFrame({"id": [0], "lb": [0.0], "ub": [10.0]})

    # Create multi-objective model
    df = df.with_columns(
        x2=xmodel.add_vars("x2", lb="lb", ub="ub", obj=pl.lit(100.0), priority=2),
        x1=xmodel.add_vars("x1", lb="lb", ub="ub", obj=pl.lit(10.0), priority=1),
    )

    xmodel.optimize(solver_type=mathopt.SolverType.GUROBI)

    # get_objective_value should raise ValueError for multi-objective
    with pytest.raises(ValueError, match="Model has 2 objectives"):
        xmodel.get_objective_value()


@skip_if_no_gurobi
def test_get_multi_objective_values():
    """Test that get_multi_objective_values returns all objective values."""
    xmodel = XplorMathOpt()

    df = pl.DataFrame({"id": [0], "lb": [0.0], "ub": [10.0]})

    # Priority 2: coefficient 100
    # Priority 1: coefficient 10
    # Priority 0: coefficient 1
    df = df.with_columns(
        x2=xmodel.add_vars("x2", lb="lb", ub="ub", obj=pl.lit(100.0), priority=2),
        x1=xmodel.add_vars("x1", lb="lb", ub="ub", obj=pl.lit(10.0), priority=1),
        x0=xmodel.add_vars("x0", lb="lb", ub="ub", obj=pl.lit(1.0), priority=0),
    )

    xmodel.optimize(solver_type=mathopt.SolverType.GUROBI)

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
