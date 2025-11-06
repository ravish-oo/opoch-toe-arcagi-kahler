"""
D builder (convex objective) using CVXPY.

Assembles DCP-valid objective D(X) and constraints for test grid (H, W, C).
All terms use production-grade CVXPY atoms (WO-08).

Uses only documented CVXPY atoms: sum, norm1, sum_squares, tv, reshape.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def build_potential_cvxpy(
    inv: Dict[str, Any],
    H: int,
    W: int,
    C: int,
    A_gamma: Optional[sp.csr_matrix] = None,
) -> Dict[str, Any]:
    """
    Build convex objective D(X) and constraints for test grid.

    Args:
        inv: Invariants dict from stage 03 (color_counts, periods, symmetries, etc.)
        H, W: Test grid dimensions
        C: Number of colors (10 for ARC)
        A_gamma: Optional Gamma constraint matrix [M × H*W*C] from WO-07

    Returns:
        {
            "X": cp.Variable (H, W, C),
            "objective": cp.Expression (scalar),
            "constraints": List[cp.Constraint],
            "__meta__": {
                "dcp_ok": bool,
                "gamma_rows": int,
                "term_weights": dict,
                "solver_default": str
            }
        }

    Spec requirements:
      - DCP-valid (prob.is_dcp() must be True)
      - Only CVXPY documented atoms
      - Simplex constraints per pixel
      - Gamma integration via reshape + matrix multiply
      - Deterministic expression tree
    """
    # Decision variable: X ∈ [0,1]^(H×W×C)
    X = cp.Variable((H, W, C), nonneg=True)

    # Constraints: per-pixel simplex (sum to 1 across colors)
    constraints = [cp.sum(X, axis=2) == 1]

    # Gamma constraints: A·vec(X) = 0
    # Only add if A_gamma has rows (M > 0)
    if A_gamma is not None and A_gamma.shape[0] > 0 and A_gamma.shape[1] == H * W * C:
        # Reshape X to vector using C-order (row-major) to match WO-07 convention
        X_vec = cp.reshape(X, (H * W * C, 1), order="C")
        constraints += [A_gamma @ X_vec == 0]

    # Build objective terms
    terms = []
    weights = {}

    # Extract invariant sub-dicts
    color_counts = inv.get("color_counts", {})
    periods = inv.get("periods", {})
    symmetries = inv.get("symmetries", {})

    # 1) Color histogram / area preservation (WO-03 v2: scale-aware)
    # Try exact counts first, fall back to proportions scaled to test size
    n_hat = None

    # Option A: Exact counts (when train outputs all same size with same counts)
    exact_counts = color_counts.get("color_counts")
    if exact_counts is not None:
        # Pad to full ARC palette size (C=10) if needed
        counts_arr = np.asarray(exact_counts, dtype=float)
        if len(counts_arr) < C:
            n_hat = np.pad(counts_arr, (0, C - len(counts_arr)), constant_values=0)
        else:
            n_hat = counts_arr[:C]

    # Option B: Proportions (when train outputs have same ratios, scale to test size)
    elif color_counts.get("color_props") is not None:
        color_props = color_counts.get("color_props")
        # Pad to full ARC palette size if needed
        props_arr = np.asarray(color_props, dtype=float)
        if len(props_arr) < C:
            props_padded = np.pad(props_arr, (0, C - len(props_arr)), constant_values=0)
        else:
            props_padded = props_arr[:C]
        # Scale proportions to test grid area, round outside CVXPY
        test_area = H * W
        n_hat = np.round(props_padded * test_area)

    # Add histogram term if we have a target
    if n_hat is not None:
        # Sum X over H and W to get total per color
        totals = cp.sum(cp.sum(X, axis=0), axis=0)  # shape (C,)
        D_hist = cp.norm1(totals - n_hat)
        terms.append(D_hist)
        weights["hist"] = 1.0

    # 2) Horizontal periodicity (WO-04)
    period_h = periods.get("period_h")
    periods_meta = periods.get("__meta__", {})
    stable_h = periods_meta.get("stable_h", False)

    if period_h is not None and period_h > 0 and stable_h and period_h < W:
        p = int(period_h)
        diffs = []
        for j in range(W - p):
            diffs.append(cp.norm1(X[:, j, :] - X[:, j + p, :]))
        D_tile_h = cp.sum(diffs)
        terms.append(D_tile_h)
        weights["tile_h"] = 1.0

    # 3) Vertical periodicity (WO-04)
    period_v = periods.get("period_v")
    stable_v = periods_meta.get("stable_v", False)

    if period_v is not None and period_v > 0 and stable_v and period_v < H:
        q = int(period_v)
        diffs = []
        for i in range(H - q):
            diffs.append(cp.norm1(X[i, :, :] - X[i + q, :, :]))
        D_tile_v = cp.sum(diffs)
        terms.append(D_tile_v)
        weights["tile_v"] = 1.0

    # 4) Horizontal mirror (midline, WO-05 patch)
    mirror_h = symmetries.get("mirror_h", False)

    if mirror_h and W % 2 == 0:
        W2 = W // 2
        left = X[:, :W2, :]
        right = X[:, W2:, :]
        # Flip right half horizontally: [:, ::-1, :]
        D_mir_h = cp.norm1(left - right[:, ::-1, :])
        terms.append(D_mir_h)
        weights["mir_h"] = 1.0

    # 5) Vertical mirror (midline, WO-05 patch)
    mirror_v = symmetries.get("mirror_v", False)

    if mirror_v and H % 2 == 0:
        H2 = H // 2
        top = X[:H2, :, :]
        bottom = X[H2:, :, :]
        # Flip bottom half vertically: [::-1, :, :]
        D_mir_v = cp.norm1(top - bottom[::-1, :, :])
        terms.append(D_mir_v)
        weights["mir_v"] = 1.0

    # 6) Horizontal concat (WO-05)
    concat_axes = symmetries.get("concat_axes", [])

    if "h" in concat_axes and W % 2 == 0:
        W2 = W // 2
        D_concat_h = cp.norm1(X[:, :W2, :] - X[:, W2:, :])
        terms.append(D_concat_h)
        weights["concat_h"] = 1.0

    # 7) Vertical concat (WO-05)
    if "v" in concat_axes and H % 2 == 0:
        H2 = H // 2
        D_concat_v = cp.norm1(X[:H2, :, :] - X[H2:, :, :])
        terms.append(D_concat_v)
        weights["concat_v"] = 1.0

    # 8) Block substitution (WO-06) - skip in v1 (optional)
    # Block-based penalties can be added later

    # 9) Optional TV regularizer
    use_tv = inv.get("use_tv", False)
    if use_tv:
        # Total variation on intensity (sum across colors)
        Y = cp.sum(X, axis=2)
        D_tv = cp.tv(Y)
        terms.append(D_tv)
        weights["tv"] = 0.1  # Conservative weight

    # Compose objective
    if terms:
        objective = cp.sum(terms)
    else:
        # No active terms - use zero objective (still DCP-valid)
        objective = 0

    # Build problem to check DCP
    prob = cp.Problem(cp.Minimize(objective), constraints)

    # Receipts
    meta = {
        "dcp_ok": bool(prob.is_dcp()),
        "gamma_rows": int(A_gamma.shape[0]) if A_gamma is not None else 0,
        "term_weights": weights,
        "solver_default": "CLARABEL",  # Default solver (OSQP, SCS also available)
        "num_terms": len(terms),
        "num_constraints": len(constraints),
    }

    return {
        "X": X,
        "objective": objective,
        "constraints": constraints,
        "problem": prob,
        "__meta__": meta,
    }


def evaluate_on_train(
    train_grid: np.ndarray,
    inv: Dict[str, Any],
    A_gamma: Optional[sp.csr_matrix] = None,
) -> float:
    """
    Evaluate objective D on a train output (as one-hot constant).

    This is the "train reproduction check" - D should be ≈ 0 on train outputs.

    Args:
        train_grid: Train output grid [H × W] with integer colors 0-9
        inv: Invariants dict
        A_gamma: Optional Gamma constraints (not used for evaluation, just for matching signature)

    Returns:
        Objective value (expect ≈ 0 for exact reproduction)
    """
    H, W = train_grid.shape
    C = 10

    # Convert train grid to one-hot encoding
    Y_onehot = np.zeros((H, W, C), dtype=float)
    for i in range(H):
        for j in range(W):
            color = int(train_grid[i, j])
            if 0 <= color < C:
                Y_onehot[i, j, color] = 1.0

    # Build objective with Y as a Constant
    Y_const = cp.Constant(Y_onehot)

    # Rebuild objective terms with Y_const instead of Variable
    terms = []

    color_counts = inv.get("color_counts", {})
    periods = inv.get("periods", {})
    symmetries = inv.get("symmetries", {})

    # 1) Color histogram (WO-03 v2: scale-aware)
    # Try exact counts first, fall back to proportions scaled to train grid size
    n_hat = None

    exact_counts = color_counts.get("color_counts")
    if exact_counts is not None:
        # Pad to full ARC palette size if needed
        counts_arr = np.asarray(exact_counts, dtype=float)
        if len(counts_arr) < C:
            n_hat = np.pad(counts_arr, (0, C - len(counts_arr)), constant_values=0)
        else:
            n_hat = counts_arr[:C]
    elif color_counts.get("color_props") is not None:
        color_props = color_counts.get("color_props")
        # Pad to full ARC palette size if needed
        props_arr = np.asarray(color_props, dtype=float)
        if len(props_arr) < C:
            props_padded = np.pad(props_arr, (0, C - len(props_arr)), constant_values=0)
        else:
            props_padded = props_arr[:C]
        # Scale proportions to this train grid's area
        train_area = H * W
        n_hat = np.round(props_padded * train_area)

    if n_hat is not None:
        totals = cp.sum(cp.sum(Y_const, axis=0), axis=0)
        terms.append(cp.norm1(totals - n_hat))

    # 2) Period H
    period_h = periods.get("period_h")
    periods_meta = periods.get("__meta__", {})
    stable_h = periods_meta.get("stable_h", False)
    if period_h is not None and period_h > 0 and stable_h and period_h < W:
        p = int(period_h)
        diffs = []
        for j in range(W - p):
            diffs.append(cp.norm1(Y_const[:, j, :] - Y_const[:, j + p, :]))
        terms.append(cp.sum(diffs))

    # 3) Period V
    period_v = periods.get("period_v")
    stable_v = periods_meta.get("stable_v", False)
    if period_v is not None and period_v > 0 and stable_v and period_v < H:
        q = int(period_v)
        diffs = []
        for i in range(H - q):
            diffs.append(cp.norm1(Y_const[i, :, :] - Y_const[i + q, :, :]))
        terms.append(cp.sum(diffs))

    # 4) Mirror H
    mirror_h = symmetries.get("mirror_h", False)
    if mirror_h and W % 2 == 0:
        W2 = W // 2
        left = Y_const[:, :W2, :]
        right = Y_const[:, W2:, :]
        terms.append(cp.norm1(left - right[:, ::-1, :]))

    # 5) Mirror V
    mirror_v = symmetries.get("mirror_v", False)
    if mirror_v and H % 2 == 0:
        H2 = H // 2
        top = Y_const[:H2, :, :]
        bottom = Y_const[H2:, :, :]
        terms.append(cp.norm1(top - bottom[::-1, :, :]))

    # 6) Concat H
    concat_axes = symmetries.get("concat_axes", [])
    if "h" in concat_axes and W % 2 == 0:
        W2 = W // 2
        terms.append(cp.norm1(Y_const[:, :W2, :] - Y_const[:, W2:, :]))

    # 7) Concat V
    if "v" in concat_axes and H % 2 == 0:
        H2 = H // 2
        terms.append(cp.norm1(Y_const[:H2, :, :] - Y_const[H2:, :, :]))

    # 8) TV (if enabled)
    if inv.get("use_tv", False):
        Y_intensity = cp.sum(Y_const, axis=2)
        terms.append(0.1 * cp.tv(Y_intensity))

    # Evaluate objective
    if terms:
        obj = cp.sum(terms)
        return float(obj.value)
    else:
        return 0.0
