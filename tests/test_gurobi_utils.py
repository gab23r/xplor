"""Tests for Gurobi utility functions."""

import gurobipy as gp
import polars as pl

from xplor.gurobi import XplorGurobi
from xplor.gurobi.utils import mlinexpr_to_linexpr_list


class TestMLinExprToLinExprList:
    """Test mlinexpr_to_linexpr_list utility function."""

    def test_expanded_format_learr_path(self):
        """Test conversion with _learr populated (expanded format)."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(10)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Create expanded MLinExpr using _from_linexprs
        linexprs_input = [(df["x"][i] + df["y"][i]) for i in range(len(df))]
        mlinexpr = gp.MLinExpr._from_linexprs(linexprs_input)  # ty:ignore[unresolved-attribute]

        # Convert back
        result = mlinexpr_to_linexpr_list(mlinexpr)

        # Verify
        assert len(result) == 10
        assert all(isinstance(expr, gp.LinExpr) for expr in result)
        assert all(expr.size() == 2 for expr in result)

    def test_compact_format_csr_path(self):
        """Test conversion with compact CSR format (MVar arithmetic)."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(10)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Create compact MLinExpr using MVar arithmetic
        mvar_x = gp.MVar(df["x"])  # ty:ignore[too-many-positional-arguments]
        mvar_y = gp.MVar(df["y"])  # ty:ignore[too-many-positional-arguments]
        mlinexpr = mvar_x + mvar_y * 2

        # Convert
        result = mlinexpr_to_linexpr_list(mlinexpr)

        # Verify
        assert len(result) == 10
        assert all(isinstance(expr, gp.LinExpr) for expr in result)
        assert all(expr.size() == 2 for expr in result)

    def test_with_constants(self):
        """Test conversion handles constant terms correctly."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(5)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # MLinExpr with constant
        mvar_x = gp.MVar(df["x"])  # ty:ignore[too-many-positional-arguments]
        mvar_y = gp.MVar(df["y"])  # ty:ignore[too-many-positional-arguments]
        mlinexpr = mvar_x + mvar_y * 3 + 7

        result = mlinexpr_to_linexpr_list(mlinexpr)

        # Verify constants are preserved
        for expr in result:
            assert expr.getConstant() == 7.0

    def test_zero_constants_optimized(self):
        """Test that zero constants are handled correctly."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(5)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # MLinExpr without constant
        mvar_x = gp.MVar(df["x"])  # ty:ignore[too-many-positional-arguments]
        mvar_y = gp.MVar(df["y"])  # ty:ignore[too-many-positional-arguments]
        mlinexpr = mvar_x + mvar_y

        result = mlinexpr_to_linexpr_list(mlinexpr)

        # Verify no constant term
        for expr in result:
            assert expr.getConstant() == 0.0

    def test_large_dataset(self):
        """Test conversion works efficiently on large datasets."""
        xmodel = XplorGurobi()
        n = 10_000
        df = pl.DataFrame({"id": range(n)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Create large MLinExpr
        mvar_x = gp.MVar(df["x"])  # ty:ignore[too-many-positional-arguments]
        mvar_y = gp.MVar(df["y"])  # ty:ignore[too-many-positional-arguments]
        mlinexpr = mvar_x * 2 + mvar_y * 3 + 5

        # Should complete quickly
        result = mlinexpr_to_linexpr_list(mlinexpr)

        # Verify
        assert len(result) == n
        assert result[0].size() == 2
        assert result[0].getConstant() == 5.0

    def test_correctness_matches_baseline(self):
        """Test that optimized paths produce same results as baseline."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(100)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Create MLinExpr
        mvar_x = gp.MVar(df["x"])  # ty:ignore[too-many-positional-arguments]
        mvar_y = gp.MVar(df["y"])  # ty:ignore[too-many-positional-arguments]
        mlinexpr = mvar_x + mvar_y * 2 + 3

        # Optimized path
        result_optimized = mlinexpr_to_linexpr_list(mlinexpr)

        # Baseline path (direct iteration)
        result_baseline = list(map(lambda r: r.item(), mlinexpr))  # noqa: C417  # ty:ignore[invalid-argument-type]

        # Compare string representations
        for i in range(len(result_optimized)):
            assert str(result_optimized[i]) == str(result_baseline[i])

    def test_single_variable(self):
        """Test with single variable per expression."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(5)}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        mvar_x = gp.MVar(df["x"])  # ty:ignore[too-many-positional-arguments]
        mlinexpr = mvar_x * 5

        result = mlinexpr_to_linexpr_list(mlinexpr)

        assert len(result) == 5
        for expr in result:
            assert expr.size() == 1

    def test_many_variables(self):
        """Test with many variables per expression."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(10)}).with_columns(
            xmodel.add_vars("x1", lb=0, ub=10),
            xmodel.add_vars("x2", lb=0, ub=10),
            xmodel.add_vars("x3", lb=0, ub=10),
            xmodel.add_vars("x4", lb=0, ub=10),
        )

        mlinexpr = gp.MVar(df["x1"]) + gp.MVar(df["x2"]) + gp.MVar(df["x3"]) + gp.MVar(df["x4"])  # ty:ignore[too-many-positional-arguments]

        result = mlinexpr_to_linexpr_list(mlinexpr)

        assert len(result) == 10
        for expr in result:
            assert expr.size() == 4
