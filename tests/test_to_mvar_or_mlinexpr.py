"""Tests for to_mvar_or_mlinexpr function."""

import gurobipy as gp
import polars as pl
import pytest

import xplor
from xplor.gurobi import XplorGurobi
from xplor.gurobi.var import to_mvar_or_mlinexpr


class TestToMVarOrMLinExpr:
    """Test to_mvar_or_mlinexpr type detection and conversion."""

    def test_pure_var_series_returns_mvar(self):
        """Pure Var series should return MVar."""
        xmodel = XplorGurobi()
        df = pl.DataFrame(height=3).with_columns(xmodel.add_vars("x"))

        result = to_mvar_or_mlinexpr(df["x"])

        assert isinstance(result, gp.MVar)

    def test_pure_linexpr_series_returns_mlinexpr(self):
        """Pure LinExpr series should return MLinExpr."""
        xmodel = XplorGurobi()
        df = pl.DataFrame(height=3).with_columns(xmodel.add_vars("x"))

        # Create series of LinExpr
        x_vars = df["x"].to_list()
        linexprs = [x + 1 for x in x_vars]
        s_linexpr = pl.Series(linexprs, dtype=pl.Object)

        result = to_mvar_or_mlinexpr(s_linexpr)

        assert isinstance(result, gp.MLinExpr)

    @pytest.mark.skip("Don't want to support that anymore.")
    def test_mixed_var_linexpr_returns_mlinexpr(self):
        """Mixed Var/LinExpr series should return MLinExpr."""
        xmodel = XplorGurobi()
        df = pl.DataFrame(height=3).with_columns(xmodel.add_vars("x"))

        x_vars = df["x"].to_list()
        mixed = [x_vars[0], x_vars[1] + 1, x_vars[2]]  # Var, LinExpr, Var
        s_mixed = pl.Series(mixed, dtype=pl.Object)

        result = to_mvar_or_mlinexpr(s_mixed)

        assert isinstance(result, gp.MLinExpr)

    def test_concat_with_varexpr_transformation(self):
        """Test concat scenario with VarExpr transformation.

        This tests the user's specific case:
        df = pl.DataFrame(...).with_columns(xmodel.add_vars("x"))
        pl.concat([df, df.select(var.x + 1)]).with_columns(x=var.x + 1)
        """
        xmodel = XplorGurobi()

        # Create initial dataframe with Var
        df = pl.DataFrame(height=3).with_columns(xmodel.add_vars("x", vtype="BINARY", obj=-1.0))

        # Concat with LinExpr column and transform
        df_concat = pl.concat([df, df.select(xplor.var.x + 1)]).with_columns(x=xplor.var.x + 1)

        # The resulting column should have mixed Var/LinExpr
        # First 3 rows: LinExpr (from var.x + 1 on original Var)
        # Last 3 rows: LinExpr (from var.x + 1 on existing LinExpr)

        result = to_mvar_or_mlinexpr(df_concat["x"])

        # Should detect LinExpr and return MLinExpr
        assert isinstance(result, gp.MLinExpr)
        assert result.shape[0] == 6

    def test_nlexpr_returns_numpy_array(self):
        """NLExpr series should return numpy array."""
        xmodel = XplorGurobi()
        df = pl.DataFrame(height=3).with_columns(xmodel.add_vars("x", lb=0.1, ub=10))

        # Create NLExpr using square
        df = df.with_columns(x_sq=xmodel.var.x.square())

        result = to_mvar_or_mlinexpr(df["x_sq"])

        # NLExpr is not vectorizable, should return numpy array
        assert isinstance(result, type(df["x_sq"].to_numpy()))
        assert len(result) == 3  # ty:ignore[invalid-argument-type]

    def test_numeric_series_returns_numpy_array(self):
        """Numeric series should return numpy array."""
        s_numeric = pl.Series([1.0, 2.0, 3.0])

        result = to_mvar_or_mlinexpr(s_numeric)

        assert isinstance(result, type(s_numeric.to_numpy()))

    def test_to_linexpr_converts_vars_to_linexprs(self):
        """Test that to_linexpr converts gp.Var to gp.LinExpr."""
        xmodel = XplorGurobi()
        df = pl.DataFrame(height=3).with_columns(xmodel.add_vars("x"))

        # Convert Var column to LinExpr using to_linexpr method
        df_with_linexpr = df.with_columns(x_linexpr=xmodel.var.x.to_linexpr())

        # Check that the result is LinExpr, not Var
        x_linexprs = df_with_linexpr["x_linexpr"].to_list()
        assert all(isinstance(expr, gp.LinExpr) for expr in x_linexprs)
        assert len(x_linexprs) == 3

        # Verify that the LinExprs are equivalent to the original Vars
        # (each LinExpr should represent 1.0 * var)
        for linexpr in x_linexprs:
            assert linexpr.size() == 1  # Should contain exactly one variable
            assert linexpr.getConstant() == 0.0  # No constant term
