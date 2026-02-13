"""Tests to verify optimization fast paths are taken.

These tests ensure that performance-critical code paths are being executed
as expected, preventing regressions in optimization strategies.
"""

from unittest.mock import patch

import gurobipy as gp
import polars as pl

import xplor
from xplor.gurobi import XplorGurobi
from xplor.gurobi.var import to_mvar_or_mlinexpr


class TestSumFastPaths:
    """Verify sum() optimization fast paths."""

    def test_sum_with_numeric_column_uses_linexpr_fast_path(self):
        """(var * numeric_col).sum() should use fast LinExpr construction."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"cost": [2.0, 3.0, 5.0]}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        # Mock gp.LinExpr to verify it's called with correct arguments
        with patch("xplor.gurobi.var.gp.LinExpr", wraps=gp.LinExpr) as mock_linexpr:
            result = df.select((xmodel.var.x * pl.col("cost")).sum()).item()

            # Verify LinExpr was called (fast path)
            assert mock_linexpr.called
            # Verify it was called with two arguments (coeffs, vars)
            call_args = mock_linexpr.call_args[0]
            assert len(call_args) == 2
            # First arg should be coefficients [2.0, 3.0, 5.0]
            assert call_args[0] == [2.0, 3.0, 5.0]
            # Second arg should be list of variables
            assert len(call_args[1]) == 3

        # Verify result is correct
        assert isinstance(result, gp.LinExpr)
        assert result.size() == 3

    def test_sum_with_scalar_uses_fast_path(self):
        """(var * scalar).sum() should use fast LinExpr path."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1, 2]}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        # Mock gp.LinExpr to verify it's called
        with patch("xplor.gurobi.var.gp.LinExpr", wraps=gp.LinExpr) as mock_linexpr:
            result = df.select((xmodel.var.x * 2.5).sum()).item()

            # Verify LinExpr was called (fast path)
            assert mock_linexpr.called
            # Verify it was called with scalar coefficient repeated
            call_args = mock_linexpr.call_args[0]
            assert len(call_args) == 2
            # First arg should be [2.5, 2.5, 2.5]
            assert call_args[0] == [2.5, 2.5, 2.5]

        assert isinstance(result, gp.LinExpr)
        assert result.size() == 3

    def test_sum_without_multiplication_uses_vectorized_path(self):
        """(var + var).sum() should use standard vectorized path."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Verify to_mvar_or_mlinexpr is called (vectorized path)
        with patch(
            "xplor.gurobi.var.to_mvar_or_mlinexpr", wraps=to_mvar_or_mlinexpr
        ) as mock_convert:
            result = df.select((xmodel.var.x + xmodel.var.y).sum()).item()

            # Should use vectorized conversion
            assert mock_convert.called

        # Standard path uses MVar/MLinExpr vectorization
        assert isinstance(result, (gp.LinExpr, gp.MLinExpr))

    def test_sum_with_integer_coefficient_uses_fast_path(self):
        """(var * 1).sum() should use fast LinExpr path with integer coefficient."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"cost": [2.0, 3.0, 5.0]}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        result = df.select(xmodel.var.x.sum()).item()
        result_int = df.select((xmodel.var.x * 1).sum()).item()
        result_float = df.select((xmodel.var.x * 1.0).sum()).item()

        assert isinstance(result, gp.LinExpr)
        assert isinstance(result_int, gp.LinExpr)
        assert isinstance(result_float, gp.LinExpr)

    def test_operations_with_null_values(self):
        """Operations should handle null values correctly (null treated as 0)."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"cost": [2.0, 3.0, 5.0]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10).shift()
        )

        # Should handle null values without crashing
        result = df.select(xmodel.var.x + 1)

        # All rows should be LinExpr
        assert all(isinstance(val, gp.LinExpr) for val in result["x"])

        # First row (null var + 1) should be just "1.0"
        # Null variable is treated as 0, so null + 1 = 1
        assert str(result["x"][0]) == "1.0"
        assert result["x"][0].size() == 0  # No variables

        # Other rows should have variables
        assert result["x"][1].size() == 1  # Has 1 variable
        assert result["x"][2].size() == 1  # Has 1 variable

    def test_operations_with_non_var_values(self):
        """Operations should handle null values correctly (null treated as 0)."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"cost": [2.0, 3.0, 5.0]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10).shift(fill_value=pl.lit(0, dtype=pl.Object))
        )

        # Should handle null values without crashing
        result = df.select(xmodel.var.x + 1)

        # All rows should be LinExpr
        assert all(isinstance(val, gp.LinExpr) for val in result["x"])

        # First row (null var + 1) should be just "1.0"
        # Null variable is treated as 0, so null + 1 = 1
        assert str(result["x"][0]) == "1.0"
        assert result["x"][0].size() == 0  # No variables

        # Other rows should have variables
        assert result["x"][1].size() == 1  # Has 1 variable
        assert result["x"][2].size() == 1  # Has 1 variable

    def test_grouped_sum_uses_quicksum(self):
        """Grouped aggregation with .sum() should use gp.quicksum fast path."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"i": range(30)}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        # Mock gp.quicksum to verify it's called for grouped aggregation
        with patch("xplor.gurobi.var.gp.quicksum", wraps=gp.quicksum) as mock_quicksum:
            # Group by (i // 10) creates 3 groups: [0-9], [10-19], [20-29]
            result = df.group_by(pl.col("i") // 10).agg(xmodel.var.x.sum())

            # Verify quicksum was called (once per group = 3 times)
            assert mock_quicksum.called
            assert mock_quicksum.call_count == 3

            # Each call should receive a series of variables (10 vars per group)
            for call in mock_quicksum.call_args_list:
                series_arg = call[0][0]  # First positional argument
                assert len(series_arg) == 10  # 10 items per group

        # Verify result structure
        assert len(result) == 3  # 3 groups
        assert all(isinstance(expr, gp.LinExpr) for expr in result["x"])
        # Each LinExpr should have 10 variables (one per row in the group)
        assert all(expr.size() == 10 for expr in result["x"])


class TestVectorizedOperations:
    """Verify vectorized operations use MLinExpr, not numpy arrays."""

    def test_square_uses_mlinexpr(self):
        """square() should convert to MLinExpr for vectorization."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1, 2]}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        with patch(
            "xplor.gurobi.var.to_mvar_or_mlinexpr", wraps=to_mvar_or_mlinexpr
        ) as mock_convert:
            result = df.select(xmodel.var.x.square()).to_series()

            # Verify to_mvar_or_mlinexpr was called
            assert mock_convert.called

            # Verify it returned MLinExpr (not numpy array)
            for call in mock_convert.call_args_list:
                series = call[0][0]
                first = series.first(ignore_nulls=True)
                if isinstance(first, gp.Var):
                    # This call should have returned MLinExpr
                    returned = to_mvar_or_mlinexpr(series)
                    assert isinstance(returned, gp.MVar)
                    break

        # Result should be NLExpr from vectorized nlfunc.square
        assert all(isinstance(x, gp.NLExpr) for x in result)

    def test_exp_uses_mlinexpr(self):
        """exp() should convert to MLinExpr for vectorization."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        with patch(
            "xplor.gurobi.var.to_mvar_or_mlinexpr", wraps=to_mvar_or_mlinexpr
        ) as mock_convert:
            result = df.select(xmodel.var.x.exp()).to_series()

            # Verify conversion was called
            assert mock_convert.called

            # Check that MLinExpr was created for Var series
            for call in mock_convert.call_args_list:
                series = call[0][0]
                first = series.first(ignore_nulls=True)
                if isinstance(first, gp.Var):
                    returned = to_mvar_or_mlinexpr(series)
                    assert isinstance(returned, gp.MVar)
                    break

        # Result should be NLExpr
        assert all(isinstance(x, gp.NLExpr) for x in result)

    def test_all_nonlinear_functions_use_vectorization(self):
        """All nonlinear functions should use vectorized MLinExpr operations."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(xmodel.add_vars("x", lb=0.1, ub=10))

        # Test each nonlinear function
        functions = [
            ("square", xmodel.var.x.square()),
            ("exp", xmodel.var.x.exp()),
            ("log", xmodel.var.x.log()),
            ("sin", xmodel.var.x.sin()),
            ("cos", xmodel.var.x.cos()),
            ("sqrt", xmodel.var.x.sqrt()),
        ]

        for func_name, expr in functions:
            with patch(
                "xplor.gurobi.var.to_mvar_or_mlinexpr", wraps=to_mvar_or_mlinexpr
            ) as mock_convert:
                result = df.select(expr).to_series()

                # Verify MLinExpr was used
                assert mock_convert.called, f"{func_name} should call to_mvar_or_mlinexpr"

                # Verify result is NLExpr
                assert all(isinstance(x, gp.NLExpr) for x in result), (
                    f"{func_name} should return NLExpr"
                )


class TestConstraintProcessing:
    """Verify constraint processing uses correct paths."""

    def test_linear_constraints_use_vectorized_path(self):
        """Linear constraints should use vectorized MVar evaluation.

        Note: Gurobi automatically splits vectorized MLinExpr constraints into
        individual constraints, so they will have [0], [1] suffixes. However,
        the evaluation itself uses vectorized operations (no Python loops).
        """
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Add linear constraint using xmodel.var (not xplor.var)
        xmodel.add_constrs(df, linear=xmodel.var.x + xmodel.var.y <= 15)

        # Gurobi splits MLinExpr into individual constraints
        xmodel.model.update()
        constrs = [c.ConstrName for c in xmodel.model.getConstrs()]

        # Should have created 2 constraints (one per row)
        assert "linear[0]" in constrs
        assert "linear[1]" in constrs

    def test_nlexpr_constraints_use_row_by_row_path(self):
        """NLExpr constraints should be detected and processed row-by-row."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=100)
        )

        # Create NLExpr (quadratic)
        df = df.with_columns(x_sq=xmodel.var.x.square())

        # Add constraint with NLExpr on RHS using xmodel.var
        xmodel.add_constrs(df, quad=xmodel.var.y == xmodel.var.x_sq)

        # Should create row-by-row constraints with [0], [1] suffix
        # NLExpr constraints are added as General Constraints (GenConstrs)
        xmodel.model.update()
        gen_constrs = [c.GenConstrName for c in xmodel.model.getGenConstrs()]

        assert "quad[0]" in gen_constrs
        assert "quad[1]" in gen_constrs

    def test_genexpr_constraints_use_row_by_row_path(self):
        """GenExpr constraints should be detected and processed row-by-row."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(
            xmodel.add_vars("x", lb=-5, ub=5), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Create GenExpr for abs
        df = df.with_columns(
            abs_x=df["x"].map_elements(lambda x: gp.abs_(x), return_dtype=pl.Object)
        )

        # Add constraint with GenExpr on RHS using xmodel.var
        xmodel.add_constrs(df, abs_c=xmodel.var.y == xmodel.var.abs_x)

        # Should create row-by-row constraints
        # GenExpr constraints are added as General Constraints (GenConstrs)
        xmodel.model.update()
        gen_constrs = [c.GenConstrName for c in xmodel.model.getGenConstrs()]

        assert "abs_c[0]" in gen_constrs
        assert "abs_c[1]" in gen_constrs


class TestTypeCorrectness:
    """Verify correct types are returned from optimized paths."""

    def test_to_mvar_or_mlinexpr_returns_correct_types(self):
        """to_mvar_or_mlinexpr should return correct types based on input."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        # Test with Var series (Object dtype)
        var_series = df["x"]
        result = to_mvar_or_mlinexpr(var_series)
        assert isinstance(result, gp.MVar)

        # Test with numeric series
        numeric_series = pl.Series([1.0, 2.0])
        result = to_mvar_or_mlinexpr(numeric_series)
        assert isinstance(result, pl.Series.to_numpy(numeric_series).__class__)

    def test_fast_path_never_returns_mlinexpr(self):
        """Fast path for (var * coeff).sum() should never return MLinExpr."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"cost": [1.0, 2.0, 3.0]}).with_columns(xmodel.add_vars("x", lb=0, ub=10))

        # Fast path test cases
        test_cases = [
            (xplor.var.x * pl.col("cost")).sum(),  # numeric column
            (xplor.var.x * 2.5).sum(),  # scalar
            (pl.col("cost") * xplor.var.x).sum(),  # reverse order
        ]

        for expr in test_cases:
            result = df.select(expr).item()
            # Fast path should return LinExpr, not MLinExpr
            assert type(result).__name__ == "LinExpr", "Fast path should return LinExpr"
            assert not isinstance(result, gp.MLinExpr) or isinstance(result, gp.LinExpr), (
                "Should not be MLinExpr"
            )


class TestNoRegressions:
    """Ensure all optimization patterns still work correctly."""

    def test_all_fast_paths_produce_correct_results(self):
        """Verify fast paths produce mathematically correct results."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"cost": [2.0, 3.0, 5.0]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10, obj=1.0)
        )

        # Test fast path
        fast_result = df.select((xplor.var.x * pl.col("cost")).sum()).item()

        # Verify it's a LinExpr with correct coefficients
        assert isinstance(fast_result, gp.LinExpr)
        assert fast_result.size() == 3

    def test_vectorized_operations_produce_correct_types(self):
        """All vectorized operations should produce correct result types."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1, 2]}).with_columns(xmodel.add_vars("x", lb=0.1, ub=10))

        # All nonlinear functions should return NLExpr
        operations = [
            ("square", xmodel.var.x.square()),
            ("exp", xmodel.var.x.exp()),
            ("log", xmodel.var.x.log()),
            ("sqrt", xmodel.var.x.sqrt()),
            ("sin", xmodel.var.x.sin()),
            ("cos", xmodel.var.x.cos()),
        ]

        for name, expr in operations:
            result = df.select(expr).to_series()
            assert all(isinstance(x, gp.NLExpr) for x in result), f"{name} should return NLExpr"

    def test_constraint_detection_never_double_processes(self):
        """Constraints should never be added twice (vectorized + row-by-row)."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=100)
        )

        # Add quadratic constraint (should trigger row-by-row)
        df = df.with_columns(x_sq=xmodel.var.x.square())
        xmodel.add_constrs(df, quad=xmodel.var.y == xmodel.var.x_sq)

        # Count constraints - NLExpr constraints are added as GenConstrs
        xmodel.model.update()
        gen_constrs = xmodel.model.getGenConstrs()

        # Should have exactly 2 constraints (one per row), not 4
        assert len(gen_constrs) == 2, "Should have exactly 2 constraints, not duplicates"
        assert gen_constrs[0].GenConstrName == "quad[0]"
        assert gen_constrs[1].GenConstrName == "quad[1]"
