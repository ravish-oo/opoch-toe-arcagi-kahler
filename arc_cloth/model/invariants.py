"""
Invariant extraction from train pairs (WO-03+).

Implements color counts and FREE-invariance verification.
Reuses D4/D2 transforms from canonicalize.py for verification.

Spec: docs/anchors/01-arc-on-the-cloth.md (§4–5)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import numpy as np

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
