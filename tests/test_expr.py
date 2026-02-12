from typing import Any

import polars as pl
import pytest

import xplor
from xplor.exprs.obj import ExpressionRepr, ObjExpr, ObjExprNode


# Create a base ObjExpr instance
@pytest.fixture
def obj_expr():
    return ObjExpr(pl.col("a"))


# --- Initialization and Representation Tests ---


def test_obj_expr_init_and_attributes(obj_expr: ObjExpr):
    """Tests basic initialization and attribute setup."""

    assert isinstance(obj_expr, ObjExpr)
    assert isinstance(obj_expr, pl.Expr)
    assert obj_expr._name is None
    assert obj_expr._expr.meta.is_column()
    assert obj_expr._nodes == []


def test_obj_expr_init_with_name():
    """Tests initialization when an explicit name is provided."""
    expr_name = "test_col"
    obj_expr = ObjExpr(pl.col("a"), name=expr_name)
    assert obj_expr._name == expr_name


# --- Operator Overloading (AST Construction) Tests ---


# Parametrize to cover all overloaded arithmetic and comparison operators
@pytest.mark.parametrize(
    ("op_method", "operator_str", "operand"),
    [
        ("__add__", "__add__", 5),
        ("__radd__", "__radd__", 5),
        ("__sub__", "__sub__", pl.col("b")),
        ("__rsub__", "__rsub__", 10),
        ("__mul__", "__mul__", 2.0),
        ("__rmul__", "__rmul__", 2.0),
        ("__truediv__", "__truediv__", pl.lit(2)),
        ("__rtruediv__", "__rtruediv__", 100),
        ("__eq__", "__eq__", 5),
        ("__le__", "__le__", pl.col("b")),
        ("__ge__", "__ge__", 10),
    ],
)
def test_operator_overloading_appends_node(
    obj_expr: ObjExpr, op_method: str, operator_str: str, operand: Any
):
    """Tests that all operator methods correctly call _append_node."""

    result = getattr(obj_expr, op_method)(operand)
    assert isinstance(result, ObjExpr)

    # Check that a single node was appended correctly
    assert len(obj_expr._nodes) == 1
    node = obj_expr._nodes[0]
    assert isinstance(node, ObjExprNode)
    assert node.operator == operator_str
    assert str(node.operand) == str(operand)


# --- Expression Processing and Execution Tests ---


def test_process_expression_simple_arithmetic(obj_expr: ObjExpr):
    """Tests parse for simple operations with literal operand."""
    obj_expr_2 = (obj_expr + 10) * 2

    expr_repr, exprs = obj_expr_2.parse()

    # Check ExpressionRepr structure
    assert isinstance(expr_repr, ExpressionRepr)
    assert str(expr_repr) == "(row[0] + 10) * 2"
    assert repr(obj_expr_2) == expr_repr
    assert obj_expr_2._repr_html_() == expr_repr

    # Check that only the root expression is in the list
    assert len(exprs) == 1
    assert exprs[0].meta.root_names() == ["a"]


def test_process_expression_with_expr_operand():
    """Tests parse for operations with another pl.Expr operand."""
    obj_expr_2 = pl.col("b") + ObjExpr(pl.col("a")) + ObjExpr(pl.col("a"))

    expr_repr, exprs = obj_expr_2.parse()  # type: ignore

    # Check ExpressionRepr structure (operand is now row[1])
    assert str(expr_repr) == "(row[1] + row[0]) + row[0]"

    # Check that both expressions are in the list
    assert len(exprs) == 2
    assert exprs[0].meta.root_names() == ["a"]
    assert exprs[1].meta.root_names() == ["b"]


# --- Execution Test (_pyexpr) and Constraint Exception ---


def test_pyexpr_for_simple_expr_delegation(obj_expr: ObjExpr):
    """Tests that _pyexpr returns the root expression's pyexpr when no nodes exist."""
    # In this case, it should just return the underlying pl.col("a") pyexpr
    assert obj_expr._pyexpr == obj_expr._expr._pyexpr


def test_str() -> None:
    assert str((xplor.var.x + pl.col("ub")) == 1) == "(x + ub) == 1"
    assert str(xplor.var.x.sum() + pl.col("ub").first()) == "x.sum() + ub.first()"
    assert str(xplor.var.x.sum().alias("a") + pl.col("ub").alias("b")) == "a + b"
    assert str((xplor.var("x") + 1 + pl.col("ub")).sum()) == "((x + 1) + ub).sum()"

    obj_expr = xplor.var.has_started == xplor.var.start + xplor.var.has_started.shift(1).alias(
        "prev_has_started"
    )
    assert str(obj_expr) == "has_started == start + prev_has_started"


def test_repr() -> None:
    assert (
        repr(xplor.var("x") + xplor.var("x").alias("b") + pl.col("x"))
        == "(row[0] + row[0]) + row[0]"
    )
    assert repr(xplor.var("x") + pl.col("^x$")) == "row[0] + row[1]"


def test_evaluate():
    expr_str = ExpressionRepr("row[0] * 2 + row[1]")
    assert expr_str.evaluate((3, 5)) == 11  # ty:ignore[invalid-argument-type]


def test_multi_expression_parsing():
    obj_expr = xplor.var.a + xplor.var.a + 2 + pl.col("b") + pl.col("a")
    exprs = obj_expr.parse()[1]

    obj_expr2 = 1 + pl.col("c") + xplor.var.a + xplor.var.b
    assert obj_expr2.parse(exprs)[0] == "(row[2] + row[0]) + row[1]"  # type: ignore


def test_gurobi_linexpr_optimization():
    """Test that (var * coeff).sum() uses optimized gp.LinExpr construction."""
    import gurobipy as gp

    from xplor.gurobi import XplorGurobi

    xmodel = XplorGurobi()

    # Create a DataFrame with variables and coefficients
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "cost": [2.0, 3.0, 5.0],
        }
    ).with_columns(xmodel.add_vars("x", lb=0, ub=10))

    # Test var * coeff pattern
    result1 = df.select((xplor.var.x * pl.col("cost")).sum()).item()
    assert isinstance(result1, gp.LinExpr)
    # Verify coefficients: should be 2.0, 3.0, 5.0
    assert result1.size() == 3

    # Test coeff * var pattern (reverse multiplication)
    result2 = df.select((pl.col("cost") * xplor.var.x).sum()).item()
    assert isinstance(result2, gp.LinExpr)
    assert result2.size() == 3

    # Both should produce equivalent expressions
    # (we can't easily compare LinExpr objects directly, but they should have same size)
    assert result1.size() == result2.size()


def test_gurobi_linexpr_optimization_with_complex_expr():
    """Test that non-(var * coeff) patterns still work correctly."""
    import gurobipy as gp

    from xplor.gurobi import XplorGurobi

    xmodel = XplorGurobi()

    df = pl.DataFrame(
        {
            "id": [0, 1],
            "cost": [2.0, 3.0],
        }
    ).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        xmodel.add_vars("y", lb=0, ub=5),
    )

    # Test more complex expression: (x + y).sum()
    result = df.select((xplor.var.x + xplor.var.y).sum()).item()
    assert isinstance(result, (gp.LinExpr, gp.MLinExpr))

    # Test scalar multiplication (should use fallback)
    result_scalar = df.select((xplor.var.x * 2).sum()).item()
    assert isinstance(result_scalar, (gp.LinExpr, gp.MLinExpr))
