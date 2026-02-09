from __future__ import annotations

from typing import Literal, TypeAlias, TypeVar

VariableType: TypeAlias = Literal["CONTINUOUS", "INTEGER", "BINARY"]

# Type variable for the underlying solver expression type (backend-specific)
ExpressionType = TypeVar("ExpressionType")

# Type variable for the underlying solver model (backend-specific)
ModelType = TypeVar("ModelType")

# Type variable for the underlying solver variable type (backend-specific)
VarType = TypeVar("VarType")
