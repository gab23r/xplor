from __future__ import annotations

from typing import Self

import polars as pl
from docplex.mp.model import Model

from xplor.exprs import VarExpr


class CplexVarExpr(VarExpr):
    """A specialized custom Polars Expression wrapper, extending ObjExpr,
    designed for constructing composite expressions or linear expressions for CPLEX.

    Methods
    -------
    sum(): Calculates the sum of optimization objects (e.g., for objective function creation).
    abs(): Applies CPLEX's absolute value function.

    """

    def sum(
        self,
    ) -> Self:
        """Get sum value.

        Examples
        --------
        >>> df.group_by('group').agg(xmodel.var("x").sum())

        """
        name = str(self) if self.meta.is_column() else f"({self})"
        return self.__class__(
            self.map_batches(
                lambda d: Model().sum(d.to_list()), return_dtype=pl.Object, returns_scalar=True
            ),
            name=name + ".sum()",
        )

    def abs(self) -> Self:
        """Apply CPLEX's absolute value function to elements in each group.

        Parameters
        ----------
        expr : pl.Expr | str
            Column name or polars expression containing CPLEX variables or expressions

        Examples
        --------
        >>> df.with_columns(xmodel.var("x").abs())

        """
        return self.__class__(
            self.map_elements(lambda d: abs(d), return_dtype=pl.Object),
            name=f"{self}.abs()",
        )


class _ProxyCplexVarExpr:
    """Proxy class for creating CPLEX variable expressions."""

    def __call__(self, name: str, /, *more_names: str) -> CplexVarExpr:
        """Create a CPLEX variable expression from column name(s).

        Parameters
        ----------
        name : str
            Primary column name
        *more_names : str
            Additional column names

        Returns
        -------
        CplexVarExpr
            Expression wrapping the specified columns

        """
        return CplexVarExpr(pl.col(name, *more_names))

    def __getattr__(self, name: str) -> CplexVarExpr:
        """Create a CPLEX variable expression via attribute access.

        Parameters
        ----------
        name : str
            Column name

        Returns
        -------
        CplexVarExpr
            Expression wrapping the specified column

        """
        return CplexVarExpr(pl.col(name))
