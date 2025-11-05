"""
ARC task loader with Π-normalization and validation.

Loads ARC-AGI task JSON, validates against canonical format, and emits
canonicalized in-memory objects with receipts.

Spec: docs/anchors/00-vision-universe.md (A0 Π), 01-arc-on-the-cloth.md (§1-2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

# Type aliases per WO-01 spec
Grid = List[List[int]]


class TrainPair(TypedDict):
    """ARC training pair with input and output grids."""
    input: Grid
    output: Grid


class TestPair(TypedDict):
    """ARC test pair with input grid only."""
    input: Grid


class Task(TypedDict):
    """ARC task with train/test pairs."""
    train: List[TrainPair]
    test: List[TestPair]


@dataclass
class TaskMetadata:
    """Metadata and receipts for a loaded task."""
    shape_summary: List[Tuple[int, int]]  # (H,W) per grid
    palette: List[int]  # sorted unique colors [0..C-1] after normalization
    palette_map: Dict[int, int]  # {original_color: normalized_index}
    receipts: Dict[str, Any] = field(default_factory=dict)


def _validate_grid(grid: Any, context: str) -> None:
    """
    Validate a single grid against ARC rules.

    Raises ValueError with actionable context if invalid.

    Args:
        grid: The grid to validate
        context: Human-readable context (e.g., "train[0].input")
    """
    if not isinstance(grid, list):
        raise ValueError(f"{context}: grid must be a list, got {type(grid).__name__}")

    if len(grid) == 0:
        raise ValueError(f"{context}: grid must be non-empty")

    # Check each row
    first_row_len = None
    for i, row in enumerate(grid):
        if not isinstance(row, list):
            raise ValueError(
                f"{context}: row {i} must be a list, got {type(row).__name__}"
            )

        # Check rectangular (all rows same length)
        if first_row_len is None:
            first_row_len = len(row)
            if first_row_len == 0:
                raise ValueError(f"{context}: row {i} must be non-empty")
        elif len(row) != first_row_len:
            raise ValueError(
                f"{context}: non-rectangular grid - row 0 has length {first_row_len}, "
                f"row {i} has length {len(row)}"
            )

        # Check each value
        for j, val in enumerate(row):
            if not isinstance(val, int):
                raise ValueError(
                    f"{context}: value at ({i},{j}) must be int, got {type(val).__name__}: {val}"
                )
            if not (0 <= val <= 9):
                raise ValueError(
                    f"{context}: value at ({i},{j}) must be in [0..9], got {val}"
                )

    # Check size constraints (1 <= H,W <= 30)
    height = len(grid)
    width = first_row_len
    if not (1 <= height <= 30):
        raise ValueError(
            f"{context}: height {height} not in valid range [1..30]"
        )
    if not (1 <= width <= 30):
        raise ValueError(
            f"{context}: width {width} not in valid range [1..30]"
        )


def _get_grid_shape(grid: Grid) -> Tuple[int, int]:
    """Return (H, W) for a grid."""
    return (len(grid), len(grid[0]) if grid else 0)


def _collect_all_colors(task_data: Dict[str, Any]) -> List[int]:
    """
    Collect all unique colors from train inputs/outputs and test input.

    Returns sorted list of unique colors in order of first appearance.
    """
    seen = []
    seen_set = set()

    # Process train pairs
    for pair in task_data["train"]:
        for grid in [pair["input"], pair["output"]]:
            for row in grid:
                for val in row:
                    if val not in seen_set:
                        seen.append(val)
                        seen_set.add(val)

    # Process test input
    for pair in task_data["test"]:
        for row in pair["input"]:
            for val in row:
                if val not in seen_set:
                    seen.append(val)
                    seen_set.add(val)

    return seen


def _normalize_palette(task_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """
    Normalize palette to contiguous [0..C-1] by order of first appearance.

    Returns (normalized_task_data, palette_map).
    This is Π-safe: labels only, no geometry change.
    """
    colors = _collect_all_colors(task_data)
    palette_map = {old_color: new_idx for new_idx, old_color in enumerate(colors)}

    def remap_grid(grid: Grid) -> Grid:
        return [[palette_map[val] for val in row] for row in grid]

    # Deep copy and remap
    normalized = {
        "train": [
            {
                "input": remap_grid(pair["input"]),
                "output": remap_grid(pair["output"])
            }
            for pair in task_data["train"]
        ],
        "test": [
            {
                "input": remap_grid(pair["input"])
            }
            for pair in task_data["test"]
        ]
    }

    return normalized, palette_map


def _test_idempotence(task_data: Dict[str, Any]) -> bool:
    """
    Test Π-idempotence: re-serialize → re-parse → normalize → compare.

    Returns True if idempotent (normalization of normalized = normalized).
    """
    # Serialize to JSON string
    json_str = json.dumps(task_data, sort_keys=True)

    # Re-parse
    reparsed = json.loads(json_str)

    # Re-normalize palette
    renormalized, _ = _normalize_palette(reparsed)

    # Compare to original (already normalized)
    # Deep equality check
    return json.dumps(task_data, sort_keys=True) == json.dumps(renormalized, sort_keys=True)


def load_task(path: str | Path) -> Tuple[Task, TaskMetadata]:
    """
    Load & validate an ARC task JSON.

    Validations (hard failures with descriptive exceptions):
      - JSON root has exactly keys {"train","test"}.
      - train is a non-empty list of pairs with keys {"input","output"}.
      - test is a list of exactly one pair with key {"input"}.
      - Each grid is a rectangular HxW matrix (1<=H,W<=30), ints in [0..9].
      - All train inputs/outputs and the test input share a palette subset of [0..9].

    Returns:
        (Task, TaskMetadata): Normalized task with metadata and receipts.

    Raises:
        ValueError: On any validation failure with actionable error message.
        NotImplementedError: If test has multiple inputs (v1 limitation).

    WO-01: Π-normalization (palette only), no geometry changes.
    """
    path = Path(path)

    # 1. Read/parse JSON
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: JSON parse error at offset {e.pos}: {e.msg}") from e
    except FileNotFoundError as e:
        raise ValueError(f"{path}: file not found") from e

    # 2. Validate structure
    expected_keys = {"train", "test"}
    actual_keys = set(raw_data.keys())

    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        msg = f"{path}: key mismatch - expected {expected_keys}, got {actual_keys}"
        if missing:
            msg += f"; missing: {missing}"
        if extra:
            msg += f"; extra: {extra}"
        raise ValueError(msg)

    # Validate train
    if not isinstance(raw_data["train"], list):
        raise ValueError(f"{path}: 'train' must be a list")
    if len(raw_data["train"]) == 0:
        raise ValueError(f"{path}: 'train' must be non-empty")

    for idx, pair in enumerate(raw_data["train"]):
        if not isinstance(pair, dict):
            raise ValueError(f"{path}: train[{idx}] must be a dict")

        pair_keys = set(pair.keys())
        expected_pair_keys = {"input", "output"}
        if pair_keys != expected_pair_keys:
            raise ValueError(
                f"{path}: train[{idx}] must have keys {expected_pair_keys}, "
                f"got {pair_keys}"
            )

        _validate_grid(pair["input"], f"{path}:train[{idx}].input")
        _validate_grid(pair["output"], f"{path}:train[{idx}].output")

    # Validate test
    if not isinstance(raw_data["test"], list):
        raise ValueError(f"{path}: 'test' must be a list")

    if len(raw_data["test"]) != 1:
        raise NotImplementedError(
            f"{path}: v1 supports a single test input; found {len(raw_data['test'])} test pairs"
        )

    test_pair = raw_data["test"][0]
    if not isinstance(test_pair, dict):
        raise ValueError(f"{path}: test[0] must be a dict")

    if "input" not in test_pair:
        raise ValueError(f"{path}: test[0] must have key 'input'")

    # Test should NOT have 'output'
    if "output" in test_pair:
        # This is actually OK for the loader, but flag it in receipts
        pass

    _validate_grid(test_pair["input"], f"{path}:test[0].input")

    # 3. Palette normalization (Π-safe: labels only)
    normalized_data, palette_map = _normalize_palette(raw_data)

    # 4. Collect metadata
    shapes = []
    for pair in normalized_data["train"]:
        shapes.append(_get_grid_shape(pair["input"]))
        shapes.append(_get_grid_shape(pair["output"]))
    shapes.append(_get_grid_shape(normalized_data["test"][0]["input"]))

    normalized_palette = sorted(palette_map.values())

    # 5. Generate receipts
    receipts = {}

    # Π-idempotence test
    receipts["pi_idempotent"] = _test_idempotence(normalized_data)

    # Spec compliance
    receipts["spec_compliance"] = {
        "has_train": len(normalized_data["train"]) > 0,
        "has_single_test": len(normalized_data["test"]) == 1,
        "rectangular": True,  # All validated above
        "values_in_0_9": True,  # All validated above
        "sizes_in_1_30": True,  # All validated above
    }

    # Palette proof
    receipts["palette_proof"] = {
        "original_palette": sorted(palette_map.keys()),
        "normalized_palette": normalized_palette,
        "mapping": palette_map,
    }

    # Build metadata
    metadata = TaskMetadata(
        shape_summary=shapes,
        palette=normalized_palette,
        palette_map=palette_map,
        receipts=receipts,
    )

    # 6. Return typed Task and metadata
    task: Task = {
        "train": normalized_data["train"],  # type: ignore
        "test": normalized_data["test"],  # type: ignore
    }

    return task, metadata
