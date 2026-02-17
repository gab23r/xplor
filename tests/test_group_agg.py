"""Test sum_by helper method."""

import polars as pl

from xplor.gurobi import XplorGurobi


def test_sum_by_with_grouping_columns():
    """Test that sum_by includes grouping columns in result."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=100).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        ata_group=pl.int_range(0, 100) // 10,
        week=pl.int_range(0, 100) % 5,
    )

    # Use sum_by to get grouped results with grouping columns
    # Explicit .sum_by(...) required
    df_grouped = xmodel.sum_by(df, by=["ata_group", "week"], sum=xmodel.var.x)

    # Should have grouping columns + aggregated column
    assert "ata_group" in df_grouped.columns
    assert "week" in df_grouped.columns
    assert "sum" in df_grouped.columns

    # Should have one row per unique (ata_group, week) combination
    expected_groups = df.select(["ata_group", "week"]).unique()
    assert len(df_grouped) == len(expected_groups)


def test_sum_by_can_join():
    """Test that result from sum_by can be joined."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=100).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        ata_group=pl.int_range(0, 100) // 10,
        week=pl.int_range(0, 100) % 5,
    )

    # Create capacity data
    capacity_df = pl.DataFrame(
        {
            "ata_group": [0, 1, 2],
            "week": [0, 0, 0],
            "capacity": [100.0, 200.0, 150.0],
        }
    )

    # Group and aggregate - explicit .sum_by(...)
    df_grouped = xmodel.sum_by(df, by=["ata_group", "week"], sum=xmodel.var.x)

    # Should be able to join on grouping columns
    result = df_grouped.join(capacity_df, on=["ata_group", "week"], how="inner")

    # Join should work
    assert "capacity" in result.columns
    assert len(result) == 3


def test_sum_by_single_column():
    """Test sum_by with single grouping column."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=50).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        group=pl.int_range(0, 50) // 10,
    )

    # Single column grouping - explicit .sum_by(...)
    df_grouped = xmodel.sum_by(df, by="group", total=xmodel.var.x)

    assert "group" in df_grouped.columns
    assert "total" in df_grouped.columns
    assert len(df_grouped) == 5


def test_sum_by_auto_aggregation():
    """Test that sum_by automatically applies .sum_by() to VarExpr."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=100).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        xmodel.add_vars("y", lb=0, ub=10),
        w=pl.int_range(0, 100) // 10,
        coeff=pl.int_range(0, 100) % 5 + 1.0,  # Numeric coefficients
    )

    # Automatic aggregation - no need to call .sum_by() on each expression!
    df_grouped = xmodel.sum_by(
        df,
        by="w",
        x_sum=xmodel.var.x,  # Automatically applies .sum_by("w")
        y_sum=xmodel.var.y,  # Automatically applies .sum_by("w")
        weighted=xmodel.var.x * pl.col("coeff"),  # Also automatic
    )

    assert "w" in df_grouped.columns
    assert "x_sum" in df_grouped.columns
    assert "y_sum" in df_grouped.columns
    assert "weighted" in df_grouped.columns
    assert len(df_grouped) == 10


def test_sum_by_positional_args():
    """Test that sum_by accepts positional arguments with auto-naming."""
    xmodel = XplorGurobi()

    df = pl.DataFrame(height=100).with_columns(
        xmodel.add_vars("x", lb=0, ub=10),
        xmodel.add_vars("y", lb=0, ub=10),
        w=pl.int_range(0, 100) // 10,
    )

    # Positional arguments - auto-named based on variable
    df_grouped = df.pipe(
        xmodel.sum_by,
        xmodel.var.x,  # Will be named "x"
        xmodel.var.y,  # Will be named "y"
        by="w",
    )

    assert "w" in df_grouped.columns
    assert "x" in df_grouped.columns
    assert "y" in df_grouped.columns
    assert len(df_grouped) == 10
