"""
WO-09: Convex solve (CVXPY one-shot)

Solves the DCP-valid program from WO-08 once to get the soft solution
X_prob ∈ [0,1]^(H×W×C) with solver receipts.

Uses CVXPY for modeling and solving (no custom algorithmic work).
Supported solvers: CLARABEL (default), OSQP, SCS.
"""

from __future__ import annotations
from typing import TypedDict, Dict, Any
import cvxpy as cp
import numpy as np
import scipy.sparse as sp


class SolveReceipts(TypedDict):
    """Solver receipts returned by solve_convex."""
    status: str
    optimal_value: float | None
    duality_gap: float | None
    eq_residual_max: float
    simplex_residual_max: float
    solver_name: str
    solve_time_s: float


def solve_convex(
    H: int,
    W: int,
    C: int,
    prog: Dict[str, Any],
    solver: str = "CLARABEL",
    eps_eq: float = 1e-7,
) -> tuple[np.ndarray, SolveReceipts]:
    """
    Solve the CVXPY program once and return (X_prob, receipts).

    Args:
        H, W: Grid dimensions
        C: Number of colors (10 for ARC)
        prog: Program dict from WO-08 containing:
            - "X": cp.Variable (H, W, C)
            - "objective": cp.Expression (scalar)
            - "constraints": List[cp.Constraint]
            - "A_gamma": Optional[sp.csr_matrix] for residual checking
        solver: Solver name ("CLARABEL", "OSQP", or "SCS")
        eps_eq: Tolerance for equality constraint residuals (unused, for future)

    Returns:
        X_prob: Solution array (H, W, C) with entries in [0,1], per-pixel sums ~ 1
        receipts: SolveReceipts dict with status, gap, residuals, time

    Raises:
        AssertionError: If program is not DCP-valid
    """
    # 1. Extract program pieces from WO-08
    X: cp.Variable = prog["X"]
    objective: cp.Expression = prog["objective"]
    constraints: list = prog["constraints"]

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # 2. DCP guard (fail fast if modeling bug)
    assert prob.is_dcp(), "Program is not DCP (see CVXPY DCP rules)"

    # 3. Choose solver (default CLARABEL)
    solve_kwargs = dict(verbose=False, warm_start=True)

    solver_upper = solver.upper()
    if solver_upper == "OSQP":
        solve_kwargs["solver"] = cp.OSQP
    elif solver_upper == "SCS":
        solve_kwargs["solver"] = cp.SCS
    else:
        # Default to CLARABEL (handles LP/QP/SOCP/EXP/POW)
        solve_kwargs["solver"] = cp.CLARABEL

    # 4. Solve (CVXPY orchestrates solver call)
    prob.solve(**solve_kwargs)

    # 5. Read receipts (status, gap, time)
    status = prob.status  # e.g., OPTIMAL, OPTIMAL_INACCURATE, INFEASIBLE
    stats = prob.solver_stats
    gap = getattr(stats, "duality_gap", None)  # Some solvers expose it, others not
    val = prob.value

    # 6. Pull solution & clamp numerics
    X_prob = np.asarray(X.value, dtype=float)  # shape (H, W, C)
    X_prob = np.clip(X_prob, 0.0, 1.0)

    # 7. Compute residuals (post-solve receipts)
    # Simplex residual: max|sum_c X[i,j,c] - 1|
    simplex_sum = X_prob.sum(axis=2)  # shape (H, W)
    simplex_resid = float(np.max(np.abs(simplex_sum - 1.0)))

    # Γ residual: If A_gamma exists, vectorize in row-major and check ||A@vec(X)||_∞
    A_gamma = prog.get("A_gamma", None)
    if isinstance(A_gamma, sp.csr_matrix) and A_gamma.shape[0] > 0:
        # Explicit C-order (row-major) to match WO-07/WO-08 convention
        X_vec = X_prob.reshape(H * W * C, order="C")
        eq_resid = float(np.max(np.abs(A_gamma @ X_vec)))
    else:
        eq_resid = 0.0

    # 8. Assemble receipts
    solver_obj = solve_kwargs["solver"]
    solver_name = solver_obj.__name__ if hasattr(solver_obj, "__name__") else str(solver_obj)

    receipts: SolveReceipts = {
        "status": status,
        "optimal_value": float(val) if val is not None else None,
        "duality_gap": float(gap) if gap is not None else None,
        "eq_residual_max": eq_resid,
        "simplex_residual_max": simplex_resid,
        "solver_name": solver_name,
        "solve_time_s": float(getattr(stats, "solve_time", np.nan)),
    }

    return X_prob, receipts
