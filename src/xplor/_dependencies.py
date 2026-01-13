from __future__ import annotations

import importlib
import importlib.metadata
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import gurobipy as gp
    from ortools.math_opt.python import mathopt


def is_installed(distribution_name: str) -> bool:
    """Check if gurobipy module is already imported."""
    try:
        importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def is_ortools_imported() -> Any:
    """Check if ortools module is already imported."""
    return "ortools" in sys.modules


def is_hexay_imported() -> Any:
    """Check if ortools module is already imported."""
    return "hexaly" in sys.modules


def get_gurobipy_model_type() -> type[gp.Model] | None:
    if "gurobipy" in sys.modules:
        import gurobipy as gp

        return gp.Model
    else:
        return None


def get_ortools_model_type() -> type[mathopt.Model] | None:
    if "ortools" in sys.modules:
        from ortools.math_opt.python import mathopt

        return mathopt.Model
    else:
        return None
