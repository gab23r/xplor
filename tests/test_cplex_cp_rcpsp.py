"""Tests for CPLEX CP backend with RCPSP (Resource-Constrained Project Scheduling)."""

import polars as pl


def test_simple_precedence():
    """Test basic precedence constraint between two tasks."""
    from xplor.cplex_cp import XplorCplexCP, var

    xmodel = XplorCplexCP()

    # Create simple task DataFrame with two tasks per row
    df = pl.DataFrame(
        {
            "task1": ["T1"],
            "task2": ["T2"],
            "dur1": [3],
            "dur2": [5],
        }
    )

    # Add interval variables for both tasks
    df = df.with_columns(
        xmodel.add_interval_vars("iv1", duration=pl.col("dur1")),
        xmodel.add_interval_vars("iv2", duration=pl.col("dur2")),
    )

    # T1 must finish before T2 starts
    xmodel.add_constrs(df, precedence=var.iv1.end_before_start(var.iv2))

    # Minimize makespan
    xmodel.minimize_makespan("iv2")

    # Solve
    xmodel.optimize(LogVerbosity="Quiet")

    # Extract solution
    df = df.with_columns(
        iv1_sol=xmodel.read_values(pl.col("iv1")),
        iv2_sol=xmodel.read_values(pl.col("iv2")),
    ).with_columns(
        start1=pl.col("iv1_sol").struct.field("start"),
        end1=pl.col("iv1_sol").struct.field("end"),
        start2=pl.col("iv2_sol").struct.field("start"),
        end2=pl.col("iv2_sol").struct.field("end"),
    )

    # Verify T1 ends before T2 starts
    t1_end = df.select("end1").to_series()[0]
    t2_start = df.select("start2").to_series()[0]

    assert t1_end <= t2_start, f"T1 should end ({t1_end}) before T2 starts ({t2_start})"
    assert df.select("start1").to_series()[0] == 0


def test_synchronization():
    """Test start-at-start synchronization constraint."""
    from xplor.cplex_cp import XplorCplexCP, var

    xmodel = XplorCplexCP()

    # Two tasks that should start together
    df = pl.DataFrame(
        {
            "dur1": [5],
            "dur2": [5],
        }
    )

    # Add interval variables
    df = df.with_columns(
        xmodel.add_interval_vars("iv1", duration=pl.col("dur1")),
        xmodel.add_interval_vars("iv2", duration=pl.col("dur2")),
    )

    # Synchronize starts
    xmodel.add_constrs(df, sync=var.iv1.start_at_start(var.iv2))

    # Minimize makespan
    xmodel.minimize_makespan("iv1")

    # Solve
    xmodel.optimize(LogVerbosity="Quiet")

    # Extract solution
    df = df.with_columns(
        iv1_sol=xmodel.read_values(pl.col("iv1")),
        iv2_sol=xmodel.read_values(pl.col("iv2")),
    ).with_columns(
        start1=pl.col("iv1_sol").struct.field("start"),
        start2=pl.col("iv2_sol").struct.field("start"),
    )

    # Verify both tasks start at the same time
    t1_start = df.select("start1").to_series()[0]
    t2_start = df.select("start2").to_series()[0]

    assert t1_start == t2_start, "Tasks should start at the same time"


def test_no_overlap():
    """Test no-overlap constraint to ensure tasks don't overlap in time."""
    from xplor.cplex_cp import XplorCplexCP, var

    xmodel = XplorCplexCP()

    # Create multiple tasks that need to be scheduled without overlapping
    df = pl.DataFrame(
        {
            "task": ["T1", "T2", "T3", "T4"],
            "duration": [3, 5, 2, 4],
        }
    )

    # Add interval variables
    df = df.with_columns(xmodel.add_interval_vars("iv", duration=pl.col("duration")))

    # Apply no-overlap constraint - ensures no two tasks overlap in time
    xmodel.add_constrs(df, no_overlap=var.iv.no_overlap())

    # Minimize makespan (total completion time)
    xmodel.minimize_makespan("iv")

    # Solve
    xmodel.optimize(LogVerbosity="Quiet")

    # Extract solution
    df = (
        df.with_columns(
            iv_sol=xmodel.read_values(pl.col("iv")),
        )
        .with_columns(
            start=pl.col("iv_sol").struct.field("start"),
            end=pl.col("iv_sol").struct.field("end"),
        )
        .sort("start")
    )

    # Verify no overlap: each task should end before or when the next one starts
    starts = df.select("start").to_series().to_list()
    ends = df.select("end").to_series().to_list()

    for i in range(len(df) - 1):
        assert ends[i] <= starts[i + 1], (
            f"Task at index {i} (ends at {ends[i]}) overlaps with task at index {i + 1} (starts at {starts[i + 1]})"
        )

    # Additionally verify that the makespan is at least the sum of durations
    # (since no overlap means sequential execution at minimum)
    total_duration = df.select("duration").sum().to_series()[0]
    makespan = max(ends)
    assert makespan >= total_duration, (
        f"Makespan {makespan} should be at least total duration {total_duration}"
    )


def test_optional_intervals():
    """Test optional intervals (tasks that may or may not be scheduled)."""
    from xplor.cplex_cp import XplorCplexCP

    xmodel = XplorCplexCP()

    # Tasks with optional flag
    df = pl.DataFrame(
        {
            "task": ["T1", "T2", "T3"],
            "duration": [3, 5, 4],
            "is_optional": [False, True, False],  # T2 is optional
        }
    )

    # Add interval variables with optional flag
    df = df.with_columns(
        xmodel.add_interval_vars("iv", duration=pl.col("duration"), optional=pl.col("is_optional"))
    )

    # Minimize makespan
    xmodel.minimize_makespan("iv")

    # Solve
    xmodel.optimize(LogVerbosity="Quiet")

    # Extract solution
    df = df.with_columns(
        iv_sol=xmodel.read_values(pl.col("iv")),
    ).with_columns(
        start=pl.col("iv_sol").struct.field("start"),
        present=pl.col("iv_sol").struct.field("present"),
    )

    # T1 and T3 should always be present (not optional)
    assert df.filter(pl.col("task") == "T1").select("present").to_series()[0] is True
    assert df.filter(pl.col("task") == "T3").select("present").to_series()[0] is True


def test_end_at_end():
    """Test end-at-end synchronization."""
    from xplor.cplex_cp import XplorCplexCP, var

    xmodel = XplorCplexCP()

    # Two tasks with different durations but should end together
    df = pl.DataFrame(
        {
            "dur1": [3],
            "dur2": [5],
        }
    )

    # Add interval variables
    df = df.with_columns(
        xmodel.add_interval_vars("iv1", duration=pl.col("dur1")),
        xmodel.add_interval_vars("iv2", duration=pl.col("dur2")),
    )

    # Synchronize ends
    xmodel.add_constrs(df, sync_end=var.iv1.end_at_end(var.iv2))

    # Minimize makespan
    xmodel.minimize_makespan("iv1")

    # Solve
    xmodel.optimize(LogVerbosity="Quiet")

    # Extract solution
    df = df.with_columns(
        iv1_sol=xmodel.read_values(pl.col("iv1")),
        iv2_sol=xmodel.read_values(pl.col("iv2")),
    ).with_columns(
        end1=pl.col("iv1_sol").struct.field("end"),
        end2=pl.col("iv2_sol").struct.field("end"),
    )

    # Verify both tasks end at the same time
    t1_end = df.select("end1").to_series()[0]
    t2_end = df.select("end2").to_series()[0]

    assert t1_end == t2_end, "Tasks should end at the same time"


def test_multiple_rows():
    """Test constraints applied to multiple rows."""
    from xplor.cplex_cp import XplorCplexCP, var

    xmodel = XplorCplexCP()

    # Multiple pairs of tasks with precedence
    df = pl.DataFrame(
        {
            "task_id": [1, 2, 3],
            "dur1": [3, 4, 2],
            "dur2": [5, 3, 6],
        }
    )

    # Add interval variables for both tasks in each row
    df = df.with_columns(
        xmodel.add_interval_vars("iv1", duration=pl.col("dur1")),
        xmodel.add_interval_vars("iv2", duration=pl.col("dur2")),
    )

    # Each iv1 must end before its corresponding iv2 starts
    xmodel.add_constrs(df, precedence=var.iv1.end_before_start(var.iv2))

    # Minimize maximum end time across all iv2
    xmodel.minimize_makespan("iv2")

    # Solve
    xmodel.optimize(LogVerbosity="Quiet")

    # Extract solution
    df = df.with_columns(
        iv1_sol=xmodel.read_values(pl.col("iv1")),
        iv2_sol=xmodel.read_values(pl.col("iv2")),
    ).with_columns(
        start1=pl.col("iv1_sol").struct.field("start"),
        end1=pl.col("iv1_sol").struct.field("end"),
        start2=pl.col("iv2_sol").struct.field("start"),
    )

    # Verify precedence for all rows
    for row in df.rows(named=True):
        assert row["end1"] <= row["start2"], (
            f"Task pair {row['task_id']}: iv1 should end before iv2 starts"
        )
