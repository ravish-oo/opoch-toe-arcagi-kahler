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

    # Stability analysis: check if counts or proportions are stable across train outputs
    # (WO-03 patch v2: scale-aware color targets)

    # For each train output, compute counts and proportions
    train_counts_list = []  # List of count vectors (np.array of shape [C])
    train_props_list = []   # List of proportion vectors (np.array of shape [C])
    train_areas = []        # List of grid areas (int)

    for pair in task["train"]:
        output_grid = np.asarray(pair["output"], dtype=np.int16)
        area = int(output_grid.size)
        counts = np.bincount(output_grid.ravel(), minlength=palette_size)
        props = counts.astype(float) / area if area > 0 else np.zeros(palette_size, dtype=float)

        train_counts_list.append(counts)
        train_props_list.append(props)
        train_areas.append(area)

    # Stability check 1: Exact counts (all counts identical AND all areas identical)
    stable_counts = False
    color_counts_stable = None

    if len(train_counts_list) > 0:
        # Check if all areas are identical
        all_areas_same = all(a == train_areas[0] for a in train_areas)

        if all_areas_same:
            # Check if all count vectors are identical
            ref_counts = train_counts_list[0]
            all_counts_same = all(np.array_equal(c, ref_counts) for c in train_counts_list)

            if all_counts_same:
                stable_counts = True
                color_counts_stable = ref_counts.tolist()

    # Stability check 2: Proportions (all proportion vectors identical with tolerance)
    stable_props = False
    color_props_stable = None

    if len(train_props_list) > 0:
        ref_props = train_props_list[0]
        # Use allclose with reasonable tolerance for floating-point comparison
        all_props_same = all(np.allclose(p, ref_props, rtol=1e-9, atol=1e-12) for p in train_props_list)

        if all_props_same:
            stable_props = True
            color_props_stable = ref_props.tolist()

    # Aggregate counts: compute hash of sorted counts for fingerprinting
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
        "stable_counts": stable_counts,
        "stable_props": stable_props,
        "train_areas": train_areas,
    }

    # Create result with optional color_counts and color_props
    result = ColorCountsInvariant({
        "type": "color_counts",
        "palette_size": palette_size,
        "per_grid_counts": per_grid_counts,
        "hash": hash_counts,
    })

    # Add stable targets (only if stability checks pass)
    if stable_counts:
        result["color_counts"] = color_counts_stable
    if stable_props:
        result["color_props"] = color_props_stable

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


# ============================================================================
# WO-05: Mirror Seams and Band Concatenation Detection
# ============================================================================


def _mirror_h_seams(G: np.ndarray) -> List[int]:
    """
    Detect horizontal mirror seams (left|right symmetry around vertical seam).

    For each potential vertical seam at column j, check if:
      - left half (G[:, :j]) equals fliplr(right half (G[:, j:]))
      - and widths match

    Uses numpy.fliplr for exact axis flip.

    Args:
        G: Grid as numpy array (H, W)

    Returns:
        List of column indices where mirror seams exist.
    """
    H, W = G.shape
    seams = []
    for j in range(1, W):  # seam between columns j-1 and j
        left = G[:, :j]
        right = G[:, j:]
        if left.shape[1] == right.shape[1] and np.array_equal(left, np.fliplr(right)):
            seams.append(j)
    return seams


def _mirror_v_seams(G: np.ndarray) -> List[int]:
    """
    Detect vertical mirror seams (top|bottom symmetry around horizontal seam).

    For each potential horizontal seam at row i, check if:
      - top half (G[:i, :]) equals flipud(bottom half (G[i:, :]))
      - and heights match

    Uses numpy.flipud for exact axis flip.

    Args:
        G: Grid as numpy array (H, W)

    Returns:
        List of row indices where mirror seams exist.
    """
    H, W = G.shape
    seams = []
    for i in range(1, H):  # seam between rows i-1 and i
        top = G[:i, :]
        bottom = G[i:, :]
        if top.shape[0] == bottom.shape[0] and np.array_equal(top, np.flipud(bottom)):
            seams.append(i)
    return seams


def _concat_h(G: np.ndarray) -> bool:
    """
    Check for exact horizontal band concatenation (left half == right half).

    Requires even width. Uses numpy.array_equal for strict equality.

    Args:
        G: Grid as numpy array (H, W)

    Returns:
        True if left half exactly equals right half.
    """
    H, W = G.shape
    if W % 2:  # odd width, no exact halves
        return False
    return np.array_equal(G[:, :W//2], G[:, W//2:])


def _concat_v(G: np.ndarray) -> bool:
    """
    Check for exact vertical band concatenation (top half == bottom half).

    Requires even height. Uses numpy.array_equal for strict equality.

    Args:
        G: Grid as numpy array (H, W)

    Returns:
        True if top half exactly equals bottom half.
    """
    H, W = G.shape
    if H % 2:  # odd height, no exact halves
        return False
    return np.array_equal(G[:H//2, :], G[H//2:, :])


def infer_symmetries(train: List[Dict[str, List[List[int]]]]) -> Dict[str, Any]:
    """
    Detect exact mirror seams and band concatenations from train outputs (WO-05).

    Uses numpy primitives for exact equality checks:
    - np.fliplr/flipud for axis flips
    - np.array_equal for strict equality
    - np.rot90 for FREE-invariance verification

    Args:
        train: List of train pairs with "output" grids (Π-normalized)

    Returns:
        {
            "mirror_h": bool,  # True iff EVERY output has ≥1 h-seam
            "mirror_v": bool,  # True iff EVERY output has ≥1 v-seam
            "mirror_h_seams": List[int],  # all h-seams from all outputs
            "mirror_v_seams": List[int],  # all v-seams from all outputs
            "concat_axes": List["h"|"v"],  # axes where ALL outputs concat
            "__meta__": {
                "per_grid": List[...],
                "free_invariance_ok": bool,
                "n_outputs", "n_square", "n_rect",
                "n_h_seams", "n_v_seams",
                "hash_sym": str,
                "method": {...}
            }
        }

    Spec requirements:
      - Use np.fliplr, np.flipud, np.array_equal, np.rot90 only
      - Task-level flags require ALL outputs to satisfy condition
      - FREE-invariance: squares swap H↔V under 90°, rectangles preserve under 180°
      - Deterministic receipts
    """
    outs = [np.asarray(p["output"], dtype=np.int16) for p in train]
    if not outs:
        raise ValueError("No train outputs")

    # Per-grid detection
    per_grid = []
    n_square = 0
    n_rect = 0

    # Track: do ALL outputs satisfy each condition?
    all_have_mirror_h = True
    all_have_mirror_v = True
    all_have_concat_h = True
    all_have_concat_v = True

    for G in outs:
        H, W = G.shape
        group = "D4" if H == W else "D2"
        if group == "D4":
            n_square += 1
        else:
            n_rect += 1

        # Check midline mirrors (exact equal-halves only)
        # Horizontal mirror: W must be even, left half = flipped right half
        mirror_h_ok = (W % 2 == 0) and np.array_equal(
            G[:, :W//2], np.fliplr(G[:, W//2:])
        )

        # Vertical mirror: H must be even, top half = flipped bottom half
        mirror_v_ok = (H % 2 == 0) and np.array_equal(
            G[:H//2, :], np.flipud(G[H//2:, :])
        )

        # Concat checks
        concat_h = _concat_h(G)
        concat_v = _concat_v(G)

        # Update task-level flags (require ALL outputs to satisfy)
        all_have_mirror_h &= mirror_h_ok
        all_have_mirror_v &= mirror_v_ok
        all_have_concat_h &= concat_h
        all_have_concat_v &= concat_v

        per_grid.append({
            "shape": [int(H), int(W)],
            "group": group,
            "mirror_h_midline_ok": mirror_h_ok,
            "mirror_v_midline_ok": mirror_v_ok,
            "concat_h": concat_h,
            "concat_v": concat_v
        })

    # Task-level flags
    mirror_h = bool(all_have_mirror_h)
    mirror_v = bool(all_have_mirror_v)
    concat_axes = []
    if all_have_concat_h:
        concat_axes.append("h")
    if all_have_concat_v:
        concat_axes.append("v")

    # FREE-invariance: shape-aware checks
    # CRITICAL: Check properties of G vs properties of transformed G
    # NOT task-level flags vs transformed grid properties
    free_ok = True
    for G in outs:
        H, W = G.shape
        if H == W:
            # Square: D4 check - 90° rotation should swap H↔V properties
            G90 = np.rot90(G, 1)

            # Midline mirror checks on both G and G90
            mirror_h_G = (W % 2 == 0) and np.array_equal(
                G[:, :W//2], np.fliplr(G[:, W//2:])
            )
            mirror_v_G = (H % 2 == 0) and np.array_equal(
                G[:H//2, :], np.flipud(G[H//2:, :])
            )
            mirror_h_G90 = (H % 2 == 0) and np.array_equal(
                G90[:, :H//2], np.fliplr(G90[:, H//2:])
            )
            mirror_v_G90 = (W % 2 == 0) and np.array_equal(
                G90[:W//2, :], np.flipud(G90[W//2:, :])
            )

            concat_h_G = _concat_h(G)
            concat_v_G = _concat_v(G)
            concat_h_G90 = _concat_h(G90)
            concat_v_G90 = _concat_v(G90)

            # H-mirror on G should become V-mirror on G90 (swap)
            if mirror_h_G != mirror_v_G90:
                free_ok = False
                break
            # V-mirror on G should become H-mirror on G90 (swap)
            if mirror_v_G != mirror_h_G90:
                free_ok = False
                break
            # Concat H on G should become concat V on G90 (swap)
            if concat_h_G != concat_v_G90:
                free_ok = False
                break
            # Concat V on G should become concat H on G90 (swap)
            if concat_v_G != concat_h_G90:
                free_ok = False
                break

            # One square grid check is enough
            break
        else:
            # Rectangle: D2 check - 180° rotation should preserve properties (no swap)
            G2 = np.rot90(G, 2)

            # Midline mirror checks on both G and G2
            mirror_h_G = (W % 2 == 0) and np.array_equal(
                G[:, :W//2], np.fliplr(G[:, W//2:])
            )
            mirror_v_G = (H % 2 == 0) and np.array_equal(
                G[:H//2, :], np.flipud(G[H//2:, :])
            )
            mirror_h_G2 = (W % 2 == 0) and np.array_equal(
                G2[:, :W//2], np.fliplr(G2[:, W//2:])
            )
            mirror_v_G2 = (H % 2 == 0) and np.array_equal(
                G2[:H//2, :], np.flipud(G2[H//2:, :])
            )

            # H-mirror existence should be preserved
            if mirror_h_G != mirror_h_G2:
                free_ok = False
                break
            # V-mirror existence should be preserved
            if mirror_v_G != mirror_v_G2:
                free_ok = False
                break
            # Concat properties should be preserved
            if _concat_h(G) != _concat_h(G2):
                free_ok = False
                break
            if _concat_v(G) != _concat_v(G2):
                free_ok = False
                break

    # Generate receipts
    # Return symbolic "mid" for midline mirrors
    mirror_h_seams = ["mid"] if mirror_h else []
    mirror_v_seams = ["mid"] if mirror_v else []

    meta = {
        "per_grid": per_grid,
        "free_invariance_ok": bool(free_ok),
        "n_outputs": len(outs),
        "n_square": int(n_square),
        "n_rect": int(n_rect),
        "n_h_seams": 1 if mirror_h else 0,
        "n_v_seams": 1 if mirror_v else 0,
        "mirror_h_seams": mirror_h_seams,
        "mirror_v_seams": mirror_v_seams,
        "hash_sym": hashlib.sha256(
            "|".join([
                str(mirror_h),
                str(mirror_v),
                ",".join(concat_axes),
                ",".join(mirror_h_seams),
                ",".join(mirror_v_seams)
            ]).encode()
        ).hexdigest(),
        "method": {
            "flipud": "numpy.flipud",
            "fliplr": "numpy.fliplr",
            "array_equal": "numpy.array_equal",
            "rot90": "numpy.rot90"
        }
    }

    return {
        "mirror_h": mirror_h,
        "mirror_v": mirror_v,
        "mirror_h_seams": mirror_h_seams,
        "mirror_v_seams": mirror_v_seams,
        "concat_axes": concat_axes,
        "__meta__": meta
    }


# ============================================================================
# WO-06: Block Codebook Learning (Minimal Patch Map)
# ============================================================================


def _build_codebook_for_k(
    pairs: List[Tuple[np.ndarray, np.ndarray]], k: int
) -> Tuple[Optional[Dict[bytes, bytes]], Dict[str, Any]]:
    """
    Try to build a single-valued, bijective block codebook for block size k.

    Args:
        pairs: List of (input, output) numpy arrays
        k: Block size (2 or 3)

    Returns:
        (codebook_or_None, metadata)
        - codebook is Dict[bytes, bytes] if valid, else None
        - metadata contains eligibility, single_valued_ok, bijection_ok, etc.
    """
    from skimage.util import view_as_blocks

    # Eligibility check: ALL pairs must have matching shapes divisible by k
    for X, Y in pairs:
        H, W = X.shape
        if X.shape != Y.shape or (H % k) or (W % k):
            return None, {
                "eligible": False,
                "reason": "shape_mismatch_or_not_divisible",
                "single_valued_ok": False,
                "bijection_ok": False,
                "n_blocks_total": 0,
                "coverage": 0.0
            }

    # Collect aligned blocks from ALL pairs
    codebook: Dict[bytes, bytes] = {}
    seen_positions = 0
    conflicts = False

    for X, Y in pairs:
        # Extract non-overlapping k×k blocks using production-grade view_as_blocks
        Bx = view_as_blocks(X, block_shape=(k, k))  # shape: [H/k, W/k, k, k]
        By = view_as_blocks(Y, block_shape=(k, k))

        # Reshape to 2D list of patches
        Bx_flat = Bx.reshape(-1, k, k)
        By_flat = By.reshape(-1, k, k)

        for bx, by in zip(Bx_flat, By_flat):
            # Convert to bytes for exact comparison
            key = bx.astype("uint8").tobytes()
            val = by.astype("uint8").tobytes()
            seen_positions += 1

            # Check single-valued: same input must map to same output
            if key in codebook:
                if codebook[key] != val:
                    conflicts = True
                    break
            else:
                codebook[key] = val

        if conflicts:
            break

    if conflicts:
        return None, {
            "eligible": True,
            "single_valued_ok": False,
            "bijection_ok": False,
            "n_blocks_total": seen_positions,
            "coverage": 0.0,
            "reason": "conflicting_mappings"
        }

    # Check bijection: |image| == |domain| (one-to-one)
    single_valued_ok = True
    image_vals = set(codebook.values())
    bijection_ok = (len(image_vals) == len(codebook))

    # Coverage (for v1, should be 1.0 since we process all aligned blocks)
    coverage = 1.0 if seen_positions > 0 else 0.0

    # Hash for receipts
    hash_str = (
        f"{k}|{len(codebook)}|" +
        "|".join(f"{k_.hex()}:{v_.hex()}" for k_, v_ in sorted(codebook.items()))
    )
    hash_codebook = hashlib.sha256(hash_str.encode()).hexdigest()

    meta = {
        "eligible": True,
        "single_valued_ok": single_valued_ok,
        "bijection_ok": bijection_ok,
        "coverage": coverage,
        "n_blocks_total": seen_positions,
        "hash_codebook": hash_codebook,
    }

    return (codebook if bijection_ok else None), meta


def infer_block_codebook(train: List[Dict[str, List[List[int]]]]) -> Dict[str, Any]:
    """
    Learn exact input→output patch dictionary for k∈{2,3} (WO-06).

    Only accepts if mapping is:
    - Single-valued: same input block → same output block always
    - Bijective: one-to-one (no two inputs → same output)

    Uses skimage.util.view_as_blocks for production-grade patch extraction.

    Args:
        train: List of train pairs with "input" and "output" grids (Π-normalized)

    Returns:
        {
            "block_size": (k, k) or None,
            "codebook": Dict[bytes, bytes] or None,
            "__meta__": {
                "k_tried": [2, 3],
                "accepted_k": int or None,
                "n_pairs": int,
                "n_blocks_total": int,
                "single_valued_ok": bool,
                "bijection_ok": bool,
                "coverage": float,
                "free_invariance_ok": bool,
                "hash_codebook": str,
                "method": {...}
            }
        }

    Spec requirements:
      - Use skimage.util.view_as_blocks for extraction
      - Try k=2 first, then k=3; accept first valid
      - Require single-valued AND bijective
      - FREE-invariance: structural consistency after same transform
      - No exceptions: return clean None if no valid k
    """
    if not train:
        raise ValueError("No train pairs")

    # Convert to numpy arrays
    pairs = [
        (np.asarray(p["input"], dtype=np.int16),
         np.asarray(p["output"], dtype=np.int16))
        for p in train
    ]

    k_tried: List[int] = []
    attempts: List[Dict[str, Any]] = []  # Track all k attempts with real metadata
    accepted_k = None
    final_codebook = None
    final_meta = {}

    for k in (2, 3):
        k_tried.append(k)
        codebook, meta = _build_codebook_for_k(pairs, k)
        attempts.append({"k": k, **meta})  # Preserve real metadata from each attempt

        # FREE-invariance check: apply same transform to both X,Y
        free_ok = True
        if codebook is not None:
            X, Y = pairs[0]
            H, W = X.shape

            # Shape-aware transform (same as WO-05)
            if H == W:
                # Square: try 90° rotation (D4)
                Xt = np.rot90(X, 1)
                Yt = np.rot90(Y, 1)
            else:
                # Rectangle: try 180° rotation (D2)
                Xt = np.rot90(X, 2)
                Yt = np.rot90(Y, 2)

            # Rebuild codebook on transformed pair
            cb_transformed, _ = _build_codebook_for_k([(Xt, Yt)], k)

            # Structural sanity: transformed should have same # of entries
            # (exact equality would require perfect D4/D2 invariance of the mapping itself)
            if cb_transformed is None:
                free_ok = False
            elif len(cb_transformed) != len(codebook):
                free_ok = False

        # If we found a valid codebook with good FREE check, accept it
        if codebook is not None and free_ok:
            accepted_k = k
            final_codebook = codebook
            final_meta = {**meta, "accepted_k": k, "free_invariance_ok": free_ok}
            break

        # If codebook exists but FREE check failed, still record and reject
        if codebook is not None and not free_ok:
            final_meta = {**meta, "accepted_k": None, "free_invariance_ok": free_ok}

    # Build final result
    if accepted_k is not None:
        result_meta = {
            "k_tried": k_tried,
            "accepted_k": accepted_k,
            "n_pairs": len(pairs),
            **final_meta,
            "attempts": attempts,  # Include full audit trail
            "method": {
                "view_as_blocks": "skimage.util.view_as_blocks",
                "array_equal": "numpy.array_equal",
                "unique": "numpy.unique",
                "rot90": "numpy.rot90",
                "hashlib": "hashlib.sha256"
            }
        }
        return {
            "block_size": (accepted_k, accepted_k),
            "codebook": final_codebook,
            "__meta__": result_meta
        }
    else:
        # No valid k found - preserve LAST attempt's real metadata for debugging
        last_meta = attempts[-1] if attempts else {
            "eligible": False,
            "reason": "no_eligible_k",
            "single_valued_ok": False,
            "bijection_ok": False,
            "n_blocks_total": 0,
            "coverage": 0.0,
        }

        result_meta = {
            "k_tried": k_tried,
            "accepted_k": None,
            "n_pairs": len(pairs),
            **last_meta,  # ← Preserve actual metadata from last k attempt
            "free_invariance_ok": True,  # vacuously true if no codebook
            "hash_codebook": "",
            "attempts": attempts,  # Full per-k audit trail
            "method": {
                "view_as_blocks": "skimage.util.view_as_blocks",
                "array_equal": "numpy.array_equal",
                "unique": "numpy.unique",
                "rot90": "numpy.rot90",
                "hashlib": "hashlib.sha256"
            }
        }
        return {
            "block_size": None,
            "codebook": None,
            "__meta__": result_meta
        }
