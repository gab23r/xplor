"""Utility functions for Gurobi optimization with xplor.

This module provides low-level utilities for efficiently converting between
Gurobi's vectorized matrix representations (MLinExpr, MVar) and lists of
individual expression objects (LinExpr, Var).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gurobipy as gp


def mlinexpr_to_linexpr_list(mlinexpr: gp.MLinExpr) -> list[gp.LinExpr]:
    """Convert MLinExpr to list of LinExpr using fastest available method.

    Gurobi's MLinExpr represents a vector of linear expressions but often needs
    to be converted to a Python list of individual LinExpr objects. This function
    uses optimized paths based on MLinExpr's internal representation.

    Performance characteristics (100k expressions):
    - FAST PATH (_learr): 171x faster than baseline (~1ms vs 171ms)
    - CSR PATH (compact): 2.35x faster than baseline (~95ms vs 223ms)
    - FALLBACK (.item()): baseline performance

    Parameters
    ----------
    mlinexpr : gp.MLinExpr
        Vectorized Gurobi linear expression to convert.

    Returns
    -------
    list[gp.LinExpr]
        List of individual linear expressions, one per row.

    Notes
    -----
    MLinExpr has two internal representations:

    1. **Expanded format** (_learr is not None):
       - Created by MLinExpr._from_linexprs()
       - Stores expressions as numpy array of LinExpr objects
       - Fast path: direct .tolist() conversion (171x faster!)

    2. **Compact format** (_learr is None, _is_compact=True):
       - Created by MVar arithmetic (e.g., MVar(x) + MVar(y) * 2)
       - Stores as sparse matrix (CSR) + variable mapping + constants
       - CSR path: manual matrix-vector multiplication (2.35x faster)
       - Structure: sparse_matrix @ mvar + const

    The CSR (Compressed Sparse Row) path manually performs matrix-vector
    multiplication row-by-row to avoid creating intermediate MLinExpr and
    unpacking overhead. For each row i, it computes:

        LinExpr[i] = sum(coeffs[j] * vars[j] for j in row i) + const[i]

    This is mathematically equivalent to `sparse_matrix @ mvar + const` but
    avoids the slow .item() unpacking step by directly constructing LinExpr.

    Examples
    --------
    >>> import gurobipy as gp
    >>> import polars as pl
    >>> from xplor.gurobi import XplorGurobi
    >>> xmodel = XplorGurobi()
    >>> df = pl.DataFrame({"id": [0, 1, 2]}).with_columns(
    ...     xmodel.add_vars("x"), xmodel.add_vars("y")
    ... )
    >>> # Create MLinExpr via vectorized operations
    >>> mlinexpr = gp.MVar(df["x"]) + gp.MVar(df["y"]) * 2
    >>> # Convert to list efficiently
    >>> linexprs = mlinexpr_to_linexpr_list(mlinexpr)
    >>> linexprs[0]
    <gurobi.LinExpr: x[0] + 2.0 y[0]>

    See Also
    --------
    gp.MLinExpr : Gurobi's vectorized linear expression class
    gp.LinExpr : Gurobi's individual linear expression class
    scipy.sparse.csr_matrix : Compressed Sparse Row matrix format

    References
    ----------
    .. [1] scipy.sparse.csr_matrix documentation
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    .. [2] Gurobi Matrix API documentation
       https://www.gurobi.com/documentation/current/refman/py_mlinexpr.html

    """
    import gurobipy as gp

    # FAST PATH: Expanded format with _learr populated
    # This is the common case when using MLinExpr._from_linexprs()
    if mlinexpr._learr is not None:  # ty:ignore[unresolved-attribute]
        # _learr is a numpy array of LinExpr objects - can directly convert to list
        # This is 171x faster than iterating with .item()!
        return mlinexpr._learr.tolist()  # ty:ignore[unresolved-attribute]

    # CSR FAST PATH: Compact format using sparse matrix representation
    # This happens when MLinExpr is created from MVar arithmetic (e.g., MVar(x) + MVar(y))
    if mlinexpr._is_compact and mlinexpr._matmulparts:  # ty:ignore[unresolved-attribute]
        # Extract internal sparse matrix representation
        # Structure: MLinExpr = sparse_matrix @ mvar + const
        sparse_matrix, mvar = mlinexpr._matmulparts[0]  # ty:ignore[unresolved-attribute]
        const = mlinexpr._const  # ty:ignore[unresolved-attribute]

        # Convert sparse CSR matrix components to Python lists for fast indexing
        # These are PUBLIC scipy.sparse.csr_matrix attributes:
        # - data: array of non-zero coefficient values
        # - indices: array of column indices for each non-zero value
        # - indptr: array of index pointers for each row's data
        data_list = sparse_matrix.data.tolist()
        indices_list = sparse_matrix.indices.tolist()
        indptr_list = sparse_matrix.indptr.tolist()
        mvar_list = mvar.tolist()  # Convert MVar to list of Var objects
        const_list = const.tolist()

        # Build LinExprs row-by-row using CSR structure
        # For each row i, extract coefficients and variable indices from CSR format
        n_rows = mlinexpr.shape[0]
        result = []
        for i in range(n_rows):
            # CSR format: row i's data is stored in data[indptr[i]:indptr[i+1]]
            start_idx = indptr_list[i]
            end_idx = indptr_list[i + 1]

            # Extract coefficients and corresponding variable indices for this row
            coeffs = data_list[start_idx:end_idx]
            var_idx = indices_list[start_idx:end_idx]
            vars_row = [mvar_list[j] for j in var_idx]

            # Construct LinExpr directly: sum(coeff * var) + constant
            # This is equivalent to: coeffs @ vars_row + const[i]
            le = gp.LinExpr(coeffs, vars_row)
            if const_list[i] != 0:
                le += const_list[i]
            result.append(le)

        return result

    # FALLBACK: Ultimate fallback for other cases (rarely happens)
    # Iterate through MLinExpr and extract each element using .item()
    # This is the baseline performance (slowest path)
    return list(map(lambda r: r.item(), mlinexpr))  # noqa: C417  # ty:ignore[invalid-argument-type]
