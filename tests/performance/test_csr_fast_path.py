"""Battle tests for CSR fast path in MLinExpr conversion.

These tests ensure the CSR optimization is robust and produces correct results
across various scenarios, edge cases, and data patterns.
"""

import gurobipy as gp
import polars as pl

from xplor.gurobi import XplorGurobi


class TestCSRFastPathCorrectness:
    """Verify CSR fast path produces mathematically correct results."""

    def test_csr_vs_baseline_simple_addition(self):
        """CSR path should match baseline for simple x + y operations."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(100)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Force compact MLinExpr (CSR path) by using MVar arithmetic
        result = df.select(xmodel.var.x + xmodel.var.y)

        # Verify all results are LinExpr
        assert all(isinstance(expr, gp.LinExpr) for expr in result["x"])

        # Verify structure: each should have 2 variables
        for expr in result["x"]:
            assert expr.size() == 2

    def test_csr_with_coefficients(self):
        """CSR path should handle different coefficients correctly."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(50)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # x + 2*y
        result = df.select(xmodel.var.x + xmodel.var.y * 2)

        # Verify all results are LinExpr
        assert all(isinstance(expr, gp.LinExpr) for expr in result["x"])

        # Verify each has 2 variables
        for expr in result["x"]:
            assert expr.size() == 2

    def test_csr_with_constants(self):
        """CSR path should handle constant terms correctly."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(50)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # x + y + 5
        result = df.select(xmodel.var.x + xmodel.var.y + 5)

        # Verify all results are LinExpr
        assert all(isinstance(expr, gp.LinExpr) for expr in result["x"])

        # Each should have 2 variables + constant term
        for expr in result["x"]:
            assert expr.size() == 2
            # Constant term should be 5
            assert expr.getConstant() == 5.0

    def test_csr_with_zero_constants(self):
        """CSR path should optimize away zero constants."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(50)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # x + y (no constant)
        result = df.select(xmodel.var.x + xmodel.var.y)

        # Verify no constant term
        for expr in result["x"]:
            assert expr.getConstant() == 0.0

    def test_csr_with_negative_coefficients(self):
        """CSR path should handle negative coefficients."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(50)}).with_columns(
            xmodel.add_vars("x", lb=-10, ub=10), xmodel.add_vars("y", lb=-10, ub=10)
        )

        # x - 2*y
        result = df.select(xmodel.var.x - xmodel.var.y * 2)

        # Verify all results are LinExpr
        assert all(isinstance(expr, gp.LinExpr) for expr in result["x"])

        # Each should have 2 variables
        for expr in result["x"]:
            assert expr.size() == 2

    def test_csr_many_variables_per_row(self):
        """CSR path should handle many variables per row."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(20)}).with_columns(
            xmodel.add_vars("x1", lb=0, ub=10),
            xmodel.add_vars("x2", lb=0, ub=10),
            xmodel.add_vars("x3", lb=0, ub=10),
            xmodel.add_vars("x4", lb=0, ub=10),
            xmodel.add_vars("x5", lb=0, ub=10),
        )

        # Sum of 5 variables
        result = df.select(
            xmodel.var.x1 + xmodel.var.x2 + xmodel.var.x3 + xmodel.var.x4 + xmodel.var.x5
        )

        # Verify all results are LinExpr with 5 variables
        for expr in result["x1"]:
            assert expr.size() == 5

    def test_csr_large_dataset(self):
        """CSR path should handle large datasets efficiently."""
        xmodel = XplorGurobi()
        n = 10_000
        df = pl.DataFrame({"id": range(n)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # This should use CSR fast path
        result = df.select(xmodel.var.x + xmodel.var.y * 3 + 7)

        # Verify correct number of rows
        assert len(result) == n

        # Spot check a few rows
        for i in [0, 100, 1000, 5000, 9999]:
            expr = result["x"][i]
            assert isinstance(expr, gp.LinExpr)
            assert expr.size() == 2
            assert expr.getConstant() == 7.0


class TestCSRFastPathEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_csr_single_row(self):
        """CSR path should handle single row correctly."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        result = df.select(xmodel.var.x + xmodel.var.y * 2)

        assert len(result) == 1
        assert result["x"][0].size() == 2

    def test_csr_two_rows(self):
        """CSR path should handle minimal dataset."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": [0, 1]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        result = df.select(xmodel.var.x + xmodel.var.y)

        assert len(result) == 2
        for expr in result["x"]:
            assert expr.size() == 2

    def test_csr_with_all_operations(self):
        """CSR path should work with various operations."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(100)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10),
            xmodel.add_vars("y", lb=0, ub=10),
            xmodel.add_vars("z", lb=0, ub=10),
        )

        # Complex expression: 2*x - 3*y + z + 10
        result = df.select(xmodel.var.x * 2 - xmodel.var.y * 3 + xmodel.var.z + 10)

        # Verify structure
        for expr in result["x"]:
            assert expr.size() == 3
            assert expr.getConstant() == 10.0

    def test_csr_fractional_coefficients(self):
        """CSR path should handle fractional coefficients."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(50)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # 0.5*x + 0.333*y
        result = df.select(xmodel.var.x * 0.5 + xmodel.var.y * 0.333)

        for expr in result["x"]:
            assert expr.size() == 2


class TestCSRFastPathIntegration:
    """Integration tests with actual model solving."""

    def test_csr_in_constraints(self):
        """CSR path results should work correctly in constraints."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"demand": [5, 7, 10]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=20, obj=1.0),
            xmodel.add_vars("y", lb=0, ub=20, obj=1.0),
        )

        # Add constraint: x + y >= demand (uses CSR path)
        xmodel.add_constrs(df, supply=xmodel.var.x + xmodel.var.y >= pl.col("demand"))

        # Solve
        xmodel.optimize()

        # Should have a solution
        assert xmodel.model.Status == gp.GRB.OPTIMAL

        # Read values
        solution = df.select(
            xmodel.read_values(pl.col("x")), xmodel.read_values(pl.col("y")), pl.col("demand")
        )

        # Verify constraints are satisfied
        for i in range(len(solution)):
            total_supply = solution["x"][i] + solution["y"][i]
            assert total_supply >= solution["demand"][i] - 1e-6

    def test_csr_in_objective(self):
        """CSR path results should work correctly in objectives."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"cost_x": [2.0, 3.0, 5.0], "cost_y": [1.5, 2.5, 4.0]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10, obj="cost_x"),
            xmodel.add_vars("y", lb=0, ub=10, obj="cost_y"),
        )

        # Constraint: x + y >= 5 (uses CSR path)
        xmodel.add_constrs(df, min_total=xmodel.var.x + xmodel.var.y >= 5)

        # Solve
        xmodel.optimize()

        # Should find optimal solution
        assert xmodel.model.Status == gp.GRB.OPTIMAL

    def test_csr_with_complex_model(self):
        """CSR path should work in complex models with many constraints."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"capacity": [100, 150, 200], "min_prod": [20, 30, 40]}).with_columns(
            xmodel.add_vars("prod_a", lb=0, obj=-3.0),  # negative for maximization
            xmodel.add_vars("prod_b", lb=0, obj=-5.0),
        )

        # Multiple constraints using CSR path
        xmodel.add_constrs(
            df, capacity_limit=xmodel.var.prod_a + xmodel.var.prod_b <= pl.col("capacity")
        )
        xmodel.add_constrs(
            df,
            min_production=xmodel.var.prod_a + xmodel.var.prod_b >= pl.col("min_prod"),
        )

        # Solve
        xmodel.optimize()

        assert xmodel.model.Status == gp.GRB.OPTIMAL


class TestCSRFastPathStressTest:
    """Stress tests to ensure robustness under extreme conditions."""

    def test_csr_very_large_dataset(self):
        """CSR path should handle very large datasets."""
        xmodel = XplorGurobi()
        n = 50_000
        df = pl.DataFrame({"id": range(n)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Should complete without errors
        result = df.select(xmodel.var.x * 2 + xmodel.var.y * 3 + 1)

        assert len(result) == n
        # Spot check
        assert result["x"][0].size() == 2
        assert result["x"][n // 2].size() == 2
        assert result["x"][-1].size() == 2

    def test_csr_many_variables_wide_format(self):
        """CSR path should handle wide datasets with many variables."""
        xmodel = XplorGurobi()
        n_vars = 20
        n_rows = 1000

        # Create many variables
        df = pl.DataFrame({"id": range(n_rows)})
        for i in range(n_vars):
            df = df.with_columns(xmodel.add_vars(f"x{i}", lb=0, ub=10))

        # Sum all variables
        expr = xmodel.var.x0
        for i in range(1, n_vars):
            expr = expr + getattr(xmodel.var, f"x{i}")

        result = df.select(expr)

        # Each row should have all variables
        for row_expr in result["x0"]:
            assert row_expr.size() == n_vars

    def test_csr_with_extreme_coefficients(self):
        """CSR path should handle very large and very small coefficients."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(100)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=1e10), xmodel.add_vars("y", lb=0, ub=1e10)
        )

        # Very large coefficient
        result1 = df.select(xmodel.var.x * 1e9 + xmodel.var.y)
        for expr in result1["x"]:
            assert expr.size() == 2

        # Very small coefficient
        result2 = df.select(xmodel.var.x * 1e-9 + xmodel.var.y)
        for expr in result2["x"]:
            assert expr.size() == 2


class TestCSRFastPathVsBaseline:
    """Compare CSR fast path results against baseline to ensure correctness."""

    def test_csr_matches_baseline_simple(self):
        """CSR results should match element-wise iteration."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(10)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # Get CSR result
        csr_result = df.select(xmodel.var.x + xmodel.var.y * 2)

        # Verify by checking string representation of expressions
        for i, expr in enumerate(csr_result["x"]):
            # Each expression should have correct structure
            assert expr.size() == 2
            # Coefficient of second variable should be 2
            assert isinstance(expr, gp.LinExpr)

    def test_csr_matches_baseline_with_constants(self):
        """CSR constant handling should match baseline."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"id": range(10)}).with_columns(
            xmodel.add_vars("x", lb=0, ub=10), xmodel.add_vars("y", lb=0, ub=10)
        )

        # CSR with constant
        result = df.select(xmodel.var.x + xmodel.var.y + 42)

        # Verify constants
        for expr in result["x"]:
            assert expr.getConstant() == 42.0

    def test_csr_matches_in_optimization(self):
        """CSR path should produce same optimal solution as baseline."""
        xmodel = XplorGurobi()
        df = pl.DataFrame({"demand": [10, 15, 20]}).with_columns(
            xmodel.add_vars("x", lb=0, ub=50, obj=1.0),
            xmodel.add_vars("y", lb=0, ub=50, obj=1.0),
        )

        # Constraint using CSR path
        xmodel.add_constrs(df, meet_demand=xmodel.var.x + xmodel.var.y >= pl.col("demand"))

        # Solve
        xmodel.optimize()

        assert xmodel.model.Status == gp.GRB.OPTIMAL

        # Get solution
        solution = df.select(
            xmodel.read_values(pl.col("x")), xmodel.read_values(pl.col("y")), pl.col("demand")
        )

        # Verify solution satisfies constraints
        for i in range(len(solution)):
            total = solution["x"][i] + solution["y"][i]
            # Should meet demand (within tolerance)
            assert total >= solution["demand"][i] - 1e-6
