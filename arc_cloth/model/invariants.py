"""
Invariant extraction from train pairs (WO-03+).

Implements color counts and period detection via autocorrelation.
Reuses D4/D2 transforms from canonicalize.py for verification.

Spec: docs/anchors/01-arc-on-the-cloth.md (§4–5)
      docs/anchors/03-invariants-catalog-v1.md (§2)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy.signal import correlate  # Production-grade autocorrelation backend

# Reuse production-grade D4/D2 transforms from canonicalize module
from arc_cloth.io.canonicalize import D4, D2_INDICES

Grid = List[List[int]]


class ColorCountsInvariant(dict):
    """
    Color counts invariant with FREE-invariance receipts.

    Structure:
        counts: Dict[int, int]  # color -> count
        __meta__: Dict[str, Any]  # receipts
    """
    pass


def _grid_to_counts(grid: Grid, palette_size: int) -> np.ndarray:
    """
    Count colors in grid using production-grade np.bincount.

    Args:
        grid: Input grid (list of lists)
        palette_size: Number of colors in palette [0..C-1]

    Returns:
        Array of shape (palette_size,) with counts for each color.
    """
    grid_np = np.array(grid, dtype=np.int32)
    flat = grid_np.flatten()

    # Use numpy's production-grade bincount
    counts = np.bincount(flat, minlength=palette_size)

    # Truncate to palette_size (bincount may extend beyond)
    return counts[:palette_size]


def _get_grid_valid_transforms(grid: Grid) -> List[int]:
    """
    Determine valid FREE transforms for a grid (shape-aware).

    - Squares (H==W): D4 (indices 0-7)
    - Rectangles (H≠W): D2 (indices 0,2,4,5)

    Reuses same logic as canonicalize.py for consistency.
    """
    H = len(grid)
    W = len(grid[0]) if grid else 0

    if H == W:
        # Square: all D4 transforms
        return list(range(8))
    else:
        # Rectangle: only D2 (shape-preserving)
        return D2_INDICES


def _verify_free_invariance(grid: Grid, counts: np.ndarray, palette_size: int) -> bool:
    """
    Verify that color counts are invariant under all valid FREE transforms.

    Args:
        grid: Input grid
        counts: Reference counts (from identity transform)
        palette_size: Number of colors

    Returns:
        True if counts are identical under all valid transforms.
    """
    grid_np = np.array(grid, dtype=np.int32)
    valid_transforms = _get_grid_valid_transforms(grid)

    for i in valid_transforms:
        # Apply transform using production-grade numpy operation
        transformed = D4[i](grid_np)

        # Count colors in transformed grid
        transformed_counts = np.bincount(
            transformed.flatten(),
            minlength=palette_size
        )[:palette_size]

        # Verify counts are identical
        if not np.array_equal(counts, transformed_counts):
            return False

    return True


def infer_color_counts(
    task: Dict[str, Any],
    _skip_verification: bool = False
) -> ColorCountsInvariant:
    """
    Extract color counts from train outputs (WO-03).

    ONLY uses train outputs (never test inputs - those are unknown!).
    Verifies FREE-invariance: counts must be identical under D4/D2 transforms.

    Args:
        task: ARC task (should be canonicalized first for consistent palette)
        _skip_verification: Skip expensive FREE-invariance checks (for testing)

    Returns:
        ColorCountsInvariant with:
          - counts: Dict[int, int] for each train output
          - __meta__: Receipts with free_invariance_ok, per_grid_counts, etc.

    Spec requirements:
      - Use np.bincount() for counting (production-grade)
      - Reuse D4/D2 from canonicalize.py (no custom transforms)
      - Verify FREE-invariance by testing all valid transforms
      - Generate receipts for verification
    """
    # Determine palette size from task metadata or infer from grids
    all_colors = set()

    # Collect colors from train outputs only (never test!)
    for pair in task["train"]:
        output_grid = pair["output"]
        for row in output_grid:
            for val in row:
                all_colors.add(val)

    palette_size = max(all_colors) + 1 if all_colors else 0

    # Extract counts from each train output
    per_grid_counts = []
    free_invariance_ok = True

    for pair_idx, pair in enumerate(task["train"]):
        output_grid = pair["output"]

        # Count colors using production-grade np.bincount
        counts = _grid_to_counts(output_grid, palette_size)

        # Verify FREE-invariance (unless skipped)
        if not _skip_verification:
            grid_free_ok = _verify_free_invariance(
                output_grid, counts, palette_size
            )
            if not grid_free_ok:
                free_invariance_ok = False

        # Store as dict for JSON serialization
        counts_dict = {i: int(counts[i]) for i in range(palette_size)}
        per_grid_counts.append({
            "pair_idx": pair_idx,
            "counts": counts_dict,
        })

    # Aggregate counts: compute hash of sorted counts for fingerprinting
    # This allows quick comparison of "is this the same invariant?"
    all_counts_list = [g["counts"] for g in per_grid_counts]
    counts_json = json.dumps(all_counts_list, sort_keys=True, separators=(',', ':'))
    hash_counts = hashlib.sha256(counts_json.encode('utf-8')).hexdigest()[:16]

    # Build receipts
    receipts = {
        "free_invariance_ok": free_invariance_ok,
        "per_grid_counts": per_grid_counts,
        "palette_size": palette_size,
        "hash_counts": hash_counts,
        "num_train_outputs": len(task["train"]),
    }

    # Create result
    result = ColorCountsInvariant({
        "type": "color_counts",
        "palette_size": palette_size,
        "per_grid_counts": per_grid_counts,
        "hash": hash_counts,
    })
    result["__meta__"] = receipts

    return result


# ============================================================================
# WO-04: Period Detection via Autocorrelation
# ============================================================================


def _row_periods(G: np.ndarray) -> List[int]:
    """
    Detect per-row period candidates using scipy.signal.correlate.

    For each row, compute autocorrelation and find the smallest lag
    with maximal correlation score (earliest strongest repetition).

    Args:
        G: Grid as numpy array (H, W)

    Returns:
        List of period candidates, one per row.
    """
    H, W = G.shape
    per = []
    for r in range(H):
        x = G[r, :].astype(np.int16)
        a = correlate(x, x, mode='full')  # length 2W-1, production-grade
        mid = W - 1
        pos = a[mid+1:]  # lags 1..W-1
        if len(pos) == 0:
            # Edge case: W=1, no positive lags available
            # Return 1 as the trivial period
            per.append(1)
        else:
            best = np.argmax(pos) + 1  # earliest max lag (argmax returns first index)
            per.append(int(best))
    return per


def _col_periods(G: np.ndarray) -> List[int]:
    """
    Detect per-column period candidates.

    Transpose and apply row logic.
    """
    return _row_periods(G.T)


def _stable_mode(votes: List[int]) -> Optional[int]:
    """
    Return earliest period if unanimous across all votes; else None.

    Args:
        votes: List of period candidates (from rows or from outputs)

    Returns:
        Unanimous value if all equal, else None.
    """
    if not votes:
        return None
    first = votes[0]
    if all(p == first for p in votes):
        return int(first)
    return None


def infer_periods(train: List[Dict[str, List[List[int]]]]) -> Dict[str, Any]:
    """
    Detect fundamental horizontal and vertical periods from train outputs (WO-04).

    Uses scipy.signal.correlate for autocorrelation-based period detection.
    Requires 100% agreement across train outputs; else returns None.

    Args:
        train: List of train pairs with "output" grids (Π-normalized)

    Returns:
        {
            "period_h": Optional[int],  # None or integer p, 1 <= p < W
            "period_v": Optional[int],  # None or integer q, 1 <= q < H
            "__meta__": {
                "per_grid": List[...],
                "stable_h": bool,
                "stable_v": bool,
                "conf_h": float,
                "conf_v": float,
                "free_invariance_ok": bool,
                "method": {...},
                "hash_periods": str
            }
        }

    Spec requirements:
      - Use scipy.signal.correlate(mode='full') for autocorrelation
      - Require unanimity across all train outputs (100% agreement)
      - Verify FREE-invariance: rotation swaps H/V for squares (D4),
        preserves for rectangles under 180° (D2)
      - Generate deterministic receipts
    """
    outs = [np.asarray(p["output"], dtype=np.int16) for p in train]
    if not outs:
        raise ValueError("No train outputs")

    # Per-grid candidates
    per_grid = []
    for G in outs:
        votes_h = _row_periods(G) if G.shape[1] > 1 else []
        votes_v = _col_periods(G) if G.shape[0] > 1 else []
        p_h = _stable_mode(votes_h)  # unanimity across rows
        p_v = _stable_mode(votes_v)  # unanimity across cols
        per_grid.append({
            "shape": [int(G.shape[0]), int(G.shape[1])],
            "p_h": p_h,
            "p_v": p_v,
            "votes_h": votes_h,
            "votes_v": votes_v
        })

    # Across-outputs stability
    def across(vals, n_outputs):
        """
        Require 100% agreement across ALL outputs (including None).

        Returns: (period_or_None, stable, conf)
          - stable=True iff every output proposes a non-None period and all are equal
          - conf = (# outputs equal to the unanimous value) / n_outputs when stable,
                   else (# outputs equal to the most common non-None value) / n_outputs
        """
        if n_outputs == 0:
            return None, False, 0.0

        # Any None means not all outputs found a period → not stable
        if any(v is None for v in vals):
            # Confidence still reported to help receipts (how close we are)
            non_none = [v for v in vals if v is not None]
            if not non_none:
                return None, False, 0.0
            mode_val = max(set(non_none), key=non_none.count)
            conf = sum(v == mode_val for v in vals) / n_outputs
            return None, False, float(conf)

        # All non-None: unanimity across outputs
        unanimous = all(v == vals[0] for v in vals)
        conf = (sum(v == vals[0] for v in vals) / n_outputs) if unanimous else 0.0
        return (int(vals[0]) if unanimous else None), unanimous, float(conf)

    period_h, stable_h, conf_h = across([g["p_h"] for g in per_grid], n_outputs=len(outs))
    period_v, stable_v, conf_v = across([g["p_v"] for g in per_grid], n_outputs=len(outs))

    # FREE invariance: shape-aware checks
    free_ok = True
    for G in outs:
        H, W = G.shape
        if H == W:
            # Square: D4 check - 90° rotation should swap H/V periods
            Gh = np.rot90(G, 1)  # 90° CCW
            gh_h = _row_periods(Gh)
            gh_v = _col_periods(Gh)
            p_h_G = _stable_mode(_row_periods(G))
            p_v_G = _stable_mode(_col_periods(G))
            p_h_Gh = _stable_mode(gh_h)
            p_v_Gh = _stable_mode(gh_v)
            if p_h_G is not None and p_v_G is not None:
                # Expect swap: original (p_h, p_v) → rotated (p_v, p_h)
                free_ok = free_ok and (p_h_G == p_v_Gh and p_v_G == p_h_Gh)
            # One square grid check is enough
            break
        else:
            # Rectangle: D2 check - 180° rotation should preserve periods (no swap)
            G2 = np.rot90(G, 2)
            if _stable_mode(_row_periods(G)) != _stable_mode(_row_periods(G2)):
                free_ok = False
                break
            if _stable_mode(_col_periods(G)) != _stable_mode(_col_periods(G2)):
                free_ok = False
                break

    # Generate receipts
    meta = {
        "per_grid": per_grid,
        "stable_h": bool(stable_h),
        "stable_v": bool(stable_v),
        "conf_h": float(conf_h),
        "conf_v": float(conf_v),
        "free_invariance_ok": bool(free_ok),
        "method": {
            "autocorr": {
                "backend": "scipy.signal.correlate",
                "mode": "full"
            }
        },
        "hash_periods": hashlib.sha256(
            np.array([
                period_h if period_h is not None else -1,
                period_v if period_v is not None else -1,
                int(1000 * conf_h),
                int(1000 * conf_v)
            ], dtype=np.int64).tobytes()
        ).hexdigest(),
    }

    return {
        "period_h": period_h,
        "period_v": period_v,
        "__meta__": meta
    }
