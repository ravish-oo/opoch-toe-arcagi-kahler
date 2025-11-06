"""
D4 pose canonicalization and palette normalization (Π).

Implements A0 from 00-vision-universe.md: quotient by free isometries (D4 pose)
and label minting (palette remap) to make tasks presentation-invariant.

Spec: docs/anchors/00-vision-universe.md (§1–3), 01-arc-on-the-cloth.md (§6–7)
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

Grid = List[List[int]]


class CanonicalTask(dict):
    """
    Canonicalized task with receipts in __meta__.

    Structure:
        train: List[Dict[str, Grid]]  # with "input" and "output" keys
        test: List[Dict[str, Grid]]    # with "input" key only
        __meta__: Dict[str, Any]       # receipts
    """
    pass


# D4 group for squares: 8 transforms using production-grade numpy operations
# Order matches dihedral group structure: rotations first, then reflections
D4 = [
    lambda a: a,                          # R0:   identity
    lambda a: np.rot90(a, 1),            # R90:  rotate 90° CCW
    lambda a: np.rot90(a, 2),            # R180: rotate 180°
    lambda a: np.rot90(a, 3),            # R270: rotate 270° CCW (= 90° CW)
    lambda a: np.flip(a, 0),             # Fv:   flip vertical (rows)
    lambda a: np.flip(a, 1),             # Fh:   flip horizontal (columns)
    lambda a: np.flip(np.rot90(a, 1), 0),# Fd1:  diagonal flip (\ diagonal)
    lambda a: np.flip(np.rot90(a, 1), 1),# Fd2:  diagonal flip (/ diagonal)
]

# D2 group for rectangles: 4 shape-preserving transforms (subset of D4)
# Excludes R90 and R270 which change H×W ↔ W×H
D2_INDICES = [0, 2, 4, 5]  # R0, R180, Fv, Fh

# Transform names for receipts
D4_NAMES = ["R0", "R90", "R180", "R270", "Fv", "Fh", "Fd1", "Fd2"]


def _grid_to_bytes(grid: np.ndarray) -> bytes:
    """Convert grid to bytes for lexicographic comparison."""
    return grid.astype(np.uint8).tobytes()


def _canonicalize_grid_pose(grid: Grid) -> Tuple[Grid, int]:
    """
    Find canonical pose for a single grid (shape-aware).

    SHAPE-AWARE CANONICALIZATION:
    - Squares (H==W): Use D4 (8 transforms) - full dihedral group
    - Rectangles (H≠W): Use D2 (4 transforms) - shape-preserving only
      D2 = {R0, R180, Fv, Fh} (indices 0, 2, 4, 5)

    Returns (canonical_grid, chosen_transform_idx).
    The canonical grid is the lexicographically minimal byte representation
    among the appropriate transform set.
    """
    grid_np = np.array(grid, dtype=np.int32)
    H, W = grid_np.shape

    # Determine transform set based on shape
    if H == W:
        # Square: use full D4 (8 transforms)
        transform_indices = list(range(8))
    else:
        # Rectangle: use D2 only (4 shape-preserving transforms)
        # R0, R180, Fv, Fh (indices from D2_INDICES)
        transform_indices = D2_INDICES

    # Generate transforms and their byte keys using numpy operations directly
    candidates = []
    for i in transform_indices:
        transformed = D4[i](grid_np)  # Apply transform via numpy lambda
        byte_key = _grid_to_bytes(transformed)
        candidates.append((byte_key, transformed, i))

    # Sort by byte key (lexicographic)
    candidates.sort(key=lambda x: x[0])

    # Pick the minimal
    _, canonical_np, chosen_idx = candidates[0]

    # Convert back to list-of-lists
    canonical_grid = canonical_np.tolist()

    return canonical_grid, chosen_idx


def _collect_global_palette(grids: List[Grid]) -> Tuple[List[int], Dict[int, int]]:
    """
    Collect global palette from all grids in pose-invariant order.

    POSE-INVARIANT RULE: Map colors to [0..C-1] by ascending numeric value.
    This ensures palette mapping is independent of grid orientation/pose.

    Returns:
        (sorted_palette, palette_map)
        sorted_palette: [0, 1, 2, ..., C-1]
        palette_map: {original_color: new_index}
    """
    # Collect all unique colors from all grids
    all_colors = set()
    for grid in grids:
        for row in grid:
            for val in row:
                all_colors.add(val)

    # Sort by ascending numeric value (pose-invariant!)
    sorted_colors = sorted(all_colors)

    # Create mapping: original_color -> index in [0..C-1]
    palette_map = {color: idx for idx, color in enumerate(sorted_colors)}
    sorted_palette = list(range(len(sorted_colors)))

    return sorted_palette, palette_map


def _remap_grid_palette(grid: Grid, palette_map: Dict[int, int]) -> Grid:
    """Apply palette mapping to a grid."""
    return [[palette_map[val] for val in row] for row in grid]


def _task_to_hash(task: Dict[str, Any]) -> str:
    """
    Compute stable hash of a task (grids only, sorted keys).

    Uses SHA-256 of canonical JSON representation.
    """
    # Extract just train and test (no metadata)
    task_clean = {
        "train": task["train"],
        "test": task["test"],
    }

    # Stable JSON dump
    json_str = json.dumps(task_clean, sort_keys=True, separators=(',', ':'))

    # SHA-256
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def _get_task_valid_transforms(task: Dict[str, Any]) -> List[int]:
    """
    Determine valid transforms for orbit collapse testing.

    For shape-aware canonicalization:
    - If ALL grids are square: test D4 (indices 0-7)
    - If ANY grid is rectangular: test only D2 (indices 0,2,4,6)

    This ensures transforms preserve the shape of all grids in the task.
    """
    all_square = True

    # Check all train grids
    for pair in task["train"]:
        for grid in [pair["input"], pair["output"]]:
            H = len(grid)
            W = len(grid[0]) if grid else 0
            if H != W:
                all_square = False
                break
        if not all_square:
            break

    # Check test grids if still all square
    if all_square:
        for pair in task["test"]:
            grid = pair["input"]
            H = len(grid)
            W = len(grid[0]) if grid else 0
            if H != W:
                all_square = False
                break

    if all_square:
        # All grids are square: test all D4 transforms
        return list(range(8))
    else:
        # At least one rectangular grid: test only D2 (shape-preserving)
        return D2_INDICES


def _apply_d4_to_task(task: Dict[str, Any], transform_idx: int) -> Dict[str, Any]:
    """
    Apply a D4 transform to all grids in a task.

    Uses numpy operations directly via D4 lambda list.
    Used for orbit collapse testing.
    """
    transformed = deepcopy(task)
    transform = D4[transform_idx]  # Get numpy lambda directly

    # Transform all train grids
    for pair in transformed["train"]:
        grid_np = np.array(pair["input"], dtype=np.int32)
        pair["input"] = transform(grid_np).tolist()

        grid_np = np.array(pair["output"], dtype=np.int32)
        pair["output"] = transform(grid_np).tolist()

    # Transform test grids
    for pair in transformed["test"]:
        grid_np = np.array(pair["input"], dtype=np.int32)
        pair["input"] = transform(grid_np).tolist()

    return transformed


def canonicalize_task(task: Dict[str, Any], _skip_receipts: bool = False) -> CanonicalTask:
    """
    Apply Π to all grids in the task (shape-aware).

    1) Shape-aware pose canonicalization:
       - Squares (H==W): choose lex-min among 8 D4 transforms
       - Rectangles (H≠W): choose lex-min among 4 D2 transforms (R0, R180, Fh, Fv)
    2) Pose-invariant palette remap: map colors to [0..C-1] by ascending numeric
       value (independent of grid orientation).

    Returns:
        CanonicalTask with __meta__ containing receipts:
          - pi_idempotent: bool
          - orbit_collapse_ok: bool
          - chosen_pose: Dict[str, str]  # grid_id -> transform name
          - palette_map: Dict[int, int]
          - shape_summary: List[Tuple[int, int]]
          - format_ok: bool (passthrough)

    Determinism requirements:
      - Same input -> same output (1:1)
      - canonicalize_task(canonicalize_task(task)) == canonicalize_task(task)
      - All D4 transforms of original task -> same canonical task
    """
    # Deep copy to avoid mutating input
    canonical = deepcopy(task)

    # Track chosen poses for receipts
    chosen_poses = {}

    # Step 1: Canonicalize pose for each grid
    grid_id = 0

    for pair_idx, pair in enumerate(canonical["train"]):
        # Canonicalize input
        pair["input"], chosen_idx = _canonicalize_grid_pose(pair["input"])
        chosen_poses[f"train[{pair_idx}].input"] = D4_NAMES[chosen_idx]
        grid_id += 1

        # Canonicalize output
        pair["output"], chosen_idx = _canonicalize_grid_pose(pair["output"])
        chosen_poses[f"train[{pair_idx}].output"] = D4_NAMES[chosen_idx]
        grid_id += 1

    for pair_idx, pair in enumerate(canonical["test"]):
        # Canonicalize test input
        pair["input"], chosen_idx = _canonicalize_grid_pose(pair["input"])
        chosen_poses[f"test[{pair_idx}].input"] = D4_NAMES[chosen_idx]
        grid_id += 1

    # Step 2: Build global palette from canonical grids
    all_grids = []
    for pair in canonical["train"]:
        all_grids.append(pair["input"])
        all_grids.append(pair["output"])
    for pair in canonical["test"]:
        all_grids.append(pair["input"])

    sorted_palette, palette_map = _collect_global_palette(all_grids)

    # Step 3: Apply palette remapping to all grids
    for pair in canonical["train"]:
        pair["input"] = _remap_grid_palette(pair["input"], palette_map)
        pair["output"] = _remap_grid_palette(pair["output"], palette_map)

    for pair in canonical["test"]:
        pair["input"] = _remap_grid_palette(pair["input"], palette_map)

    # Step 4: Collect shape summary
    shape_summary = []
    for pair in canonical["train"]:
        shape_summary.append((len(pair["input"]), len(pair["input"][0]) if pair["input"] else 0))
        shape_summary.append((len(pair["output"]), len(pair["output"][0]) if pair["output"] else 0))
    for pair in canonical["test"]:
        shape_summary.append((len(pair["input"]), len(pair["input"][0]) if pair["input"] else 0))

    # Step 5: Generate receipts

    # Skip expensive receipt tests during recursive calls to avoid infinite loops
    if _skip_receipts:
        pi_idempotent = True  # Assume true for recursive calls
        orbit_collapse_ok = True  # Assume true for recursive calls
        format_ok = True
    else:
        # 5a: Idempotence test - canonicalize again and compare
        # Use _skip_receipts=True to prevent infinite recursion
        canonical_clean = {"train": canonical["train"], "test": canonical["test"]}
        canonical_again = canonicalize_task(canonical_clean, _skip_receipts=True)
        canonical_again_clean = {"train": canonical_again["train"], "test": canonical_again["test"]}
        pi_idempotent = (
            json.dumps(canonical_clean, sort_keys=True) ==
            json.dumps(canonical_again_clean, sort_keys=True)
        )

        # 5b: Orbit collapse test - apply valid transforms to original, canonicalize each
        # Use shape-aware transform set (D4 for all-square tasks, D2 for rectangular tasks)
        valid_transforms = _get_task_valid_transforms(task)
        canonical_hash = _task_to_hash(canonical)
        orbit_collapse_ok = True

        for i in valid_transforms:
            transformed_task = _apply_d4_to_task(task, i)
            transformed_canonical = canonicalize_task(transformed_task, _skip_receipts=True)
            transformed_hash = _task_to_hash(transformed_canonical)

            if transformed_hash != canonical_hash:
                orbit_collapse_ok = False
                break

        # 5c: Format OK (passthrough from loader, assume True if we got here)
        format_ok = True

    # Build receipts
    receipts = {
        "pi_idempotent": pi_idempotent,
        "orbit_collapse_ok": orbit_collapse_ok,
        "chosen_pose": chosen_poses,
        "palette_map": palette_map,
        "shape_summary": shape_summary,
        "format_ok": format_ok,
    }

    # Attach metadata
    result = CanonicalTask(canonical)
    result["__meta__"] = receipts

    return result
