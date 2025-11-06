"""
Interface constraints (Γ) builder.

Assembles sparse linear equalities A·vec(X) = 0 from invariants (WO-07).

Each constraint encodes pixel equality: X[i,j,:] = X[i',j',:] across all colors.
Sources: periods (WO-04), mirror seams (WO-05), concats (WO-05).

Uses production-grade SciPy sparse and NumPy primitives only.
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import scipy.sparse as sp
import hashlib


def _tie_pair(
    rows: List[int],
    cols: List[int],
    data: List[float],
    row_id: int,
    i: int,
    j: int,
    ip: int,
    jp: int,
    C: int,
    H: int,
    W: int,
) -> int:
    """
    Add C rows to tie X[i,j,:] = X[ip,jp,:] (one row per color channel).

    Args:
        rows, cols, data: COO triplet lists (mutated in-place)
        row_id: current row counter
        i, j: first pixel coordinates
        ip, jp: second pixel coordinates
        C: number of color channels (10 for ARC)
        H, W: grid dimensions

    Returns:
        Updated row_id
    """
    # Map (i,j,c) → flat index using row-major order
    idx1 = np.ravel_multi_index(
        (i * np.ones(C, dtype=int), j * np.ones(C, dtype=int), np.arange(C)),
        dims=(H, W, C),
        order="C",
    )
    idx2 = np.ravel_multi_index(
        (ip * np.ones(C, dtype=int), jp * np.ones(C, dtype=int), np.arange(C)),
        dims=(H, W, C),
        order="C",
    )

    # Add one row per color: [+1 at idx1, -1 at idx2]
    for k1, k2 in zip(idx1, idx2):
        rows.extend([row_id, row_id])
        cols.extend([int(k1), int(k2)])
        data.extend([1.0, -1.0])
        row_id += 1

    return row_id


def build_interfaces(inv: Dict[str, Any], H: int, W: int, C: int) -> Dict[str, Any]:
    """
    Build Γ as linear equalities A·vec(X) = 0 from invariants.

    Constraint sources:
    - Horizontal/vertical periods (WO-04) - only if stable
    - Mirror seams (WO-05)
    - Concats (WO-05)
    - Block overlaps (WO-06) [optional, skipped in v1]

    Args:
        inv: invariants dict from stage 03 (periods, symmetries, block_codebook)
        H, W: grid dimensions (output grid for solving)
        C: number of color channels (10 for ARC)

    Returns:
        {
            "A": CSR sparse matrix [M × H*W*C],
            "b": zero vector [M],
            "__meta__": {
                "M": number of constraints,
                "rank": matrix rank (dense or approximate),
                "density": A.nnz / (M * H*W*C),
                "term_counts": dict of constraint counts per source,
                "shape": [M, H*W*C],
                "hash_A": SHA-256 of CSR triplets,
                "method": dict of library references
            }
        }
    """
    # Initialize COO triplet lists
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    row_id = 0

    term_counts = {
        "period_h": 0,
        "period_v": 0,
        "mirror_h": 0,
        "mirror_v": 0,
        "concat_h": 0,
        "concat_v": 0,
        "block_overlap": 0,
    }

    # Extract invariant sub-dicts
    periods = inv.get("periods", {})
    symmetries = inv.get("symmetries", {})

    # 1) Horizontal period (only if stable)
    period_h = periods.get("period_h")
    periods_meta = periods.get("__meta__", {})
    stable_h = periods_meta.get("stable_h", False)

    if period_h is not None and period_h > 0 and stable_h:
        p = period_h
        for i in range(H):
            for j in range(W - p):
                row_id = _tie_pair(rows, cols, data, row_id, i, j, i, j + p, C, H, W)
                term_counts["period_h"] += C

    # 2) Vertical period (only if stable)
    period_v = periods.get("period_v")
    stable_v = periods_meta.get("stable_v", False)

    if period_v is not None and period_v > 0 and stable_v:
        q = period_v
        for j in range(W):
            for i in range(H - q):
                row_id = _tie_pair(rows, cols, data, row_id, i, j, i + q, j, C, H, W)
                term_counts["period_v"] += C

    # 3) Horizontal mirror seams (column seams)
    symmetries_meta = symmetries.get("__meta__", {})
    mirror_h_seams = symmetries_meta.get("mirror_h_seams", [])

    for j0 in mirror_h_seams:
        w = min(j0, W - j0)
        for i in range(H):
            for t in range(w):
                row_id = _tie_pair(
                    rows, cols, data, row_id, i, j0 - 1 - t, i, j0 + t, C, H, W
                )
                term_counts["mirror_h"] += C

    # 4) Vertical mirror seams (row seams)
    mirror_v_seams = symmetries_meta.get("mirror_v_seams", [])

    for i0 in mirror_v_seams:
        h = min(i0, H - i0)
        for j in range(W):
            for t in range(h):
                row_id = _tie_pair(
                    rows, cols, data, row_id, i0 - 1 - t, j, i0 + t, j, C, H, W
                )
                term_counts["mirror_v"] += C

    # 5) Horizontal concat (even W)
    concat_axes = symmetries.get("concat_axes", [])

    if "h" in concat_axes and W % 2 == 0:
        k = W // 2
        for i in range(H):
            for j in range(k):
                row_id = _tie_pair(rows, cols, data, row_id, i, j, i, j + k, C, H, W)
                term_counts["concat_h"] += C

    # 6) Vertical concat (even H)
    if "v" in concat_axes and H % 2 == 0:
        k = H // 2
        for j in range(W):
            for i in range(k):
                row_id = _tie_pair(rows, cols, data, row_id, i, j, i + k, j, C, H, W)
                term_counts["concat_v"] += C

    # 7) Block overlaps - skip in v1 (optional)
    # Non-overlapping tilings have no intra-tile overlaps to tie

    # Assemble CSR from COO triplets
    if row_id > 0:
        A_coo = sp.coo_matrix((data, (rows, cols)), shape=(row_id, H * W * C))
        A = A_coo.tocsr()  # COO→CSR sums duplicates automatically
    else:
        # Empty constraint set
        A = sp.csr_matrix((0, H * W * C), dtype=float)

    b = np.zeros(A.shape[0], dtype=float)

    # Compute receipts
    M = int(A.shape[0])
    density = float(A.nnz) / (A.shape[0] * A.shape[1]) if M > 0 and A.shape[1] > 0 else 0.0

    # Rank computation
    rank = None
    try:
        if M == 0:
            rank = 0
        elif A.shape[0] * A.shape[1] <= 1_000_000:
            # Small matrix: use dense SVD
            rank = int(np.linalg.matrix_rank(A.toarray()))
        else:
            # Large matrix: approximate rank via sparse SVD
            from scipy.sparse.linalg import svds

            k = min(32, min(A.shape) - 1)
            if k > 0:
                s = svds(A, k=k, return_singular_vectors=False)
                rank = int(np.count_nonzero(s > 1e-10))
            else:
                rank = 0
    except Exception:
        rank = None

    # Hash CSR triplets for determinism check
    if M > 0:
        hash_A = hashlib.sha256(
            b"".join([A.data.tobytes(), A.indices.tobytes(), A.indptr.tobytes()])
        ).hexdigest()
    else:
        hash_A = hashlib.sha256(b"").hexdigest()

    return {
        "A": A,
        "b": b,
        "__meta__": {
            "M": M,
            "rank": rank,
            "density": density,
            "term_counts": term_counts,
            "shape": [int(A.shape[0]), int(A.shape[1])],
            "hash_A": hash_A,
            "method": {
                "coo": "scipy.sparse.coo_matrix",
                "csr": "scipy.sparse.csr_matrix",
                "kron": "scipy.sparse.kron",
                "ravel_multi_index": "numpy.ravel_multi_index",
                "matrix_rank": "numpy.linalg.matrix_rank",
                "svds": "scipy.sparse.linalg.svds",
            },
        },
    }
