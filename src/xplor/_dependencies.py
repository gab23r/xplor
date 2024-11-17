import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import gurobipy as gp
    from ortools.math_opt.python import mathopt


def is_gurobipy_imported() -> bool:
    """Check if gurobipy module is already imported."""
    return "gurobipy" in sys.modules


def is_ortools_imported() -> Any:
    """Check if ortools module is already imported."""
    return sys.modules.get("ortools", None)


def get_gurobipy_model_type() -> "type[gp.Model] | None":
    if "gurobipy" in sys.modules:
        import gurobipy as gp

        return gp.Model
    else:
        return None


def get_ortools_model_type() -> "type[mathopt.Model] | None":
    if "ortools" in sys.modules:
        from ortools.math_opt.python import mathopt

        return mathopt.Model
    else:
        return None
