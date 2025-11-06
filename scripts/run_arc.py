#!/usr/bin/env python3
"""
Batch ARC task runner.

Runs the full pipeline on ARC corpus or individual tasks, collecting receipts
and results.

Usage:
    python scripts/run_arc.py --root data/ --mode convex --limit 10
    python scripts/run_arc.py --root data/test_task_simple.json --mode convex

WO-01+: Extensible runner for all pipeline stages.
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_cloth.runner.run_task import TaskResult, solve_task


def load_corpus_file(path: Path) -> Dict[str, Any]:
    """
    Load a corpus file (dict of {task_id: task_data}).

    Returns dict of task_id -> task_data.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "train" in data and "test" in data:
            # Single task file
            return {path.stem: data}
        elif isinstance(data, dict):
            # Corpus file (multiple tasks)
            return data
        else:
            print(f"WARNING: {path} has unexpected format, skipping")
            return {}
    except Exception as e:
        print(f"ERROR: Failed to load {path}: {e}")
        return {}


def process_task(
    task_id: str, task_data: Dict[str, Any], mode: str, save_dir: Path | None
) -> TaskResult:
    """
    Process a single task from corpus.

    Writes to temp file, calls solve_task, optionally saves output.
    """
    # Write to temp file (runner expects file path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(task_data, f)
        temp_path = f.name

    try:
        result = solve_task(temp_path, mode=mode)
        result.task_id = task_id  # Override with actual task ID

        # Save output if requested
        if save_dir and result.output_grid:
            out_path = save_dir / f"{task_id}.json"
            out_path.write_text(
                json.dumps({"output": result.output_grid}, indent=2)
            )

        return result

    finally:
        # Clean up temp file
        Path(temp_path).unlink()


def print_receipt_summary(result: TaskResult) -> None:
    """Print a compact summary of receipts for one task."""
    print(f"\n{'='*80}")
    print(f"Task: {result.task_id}")
    print(f"Status: {result.status} | Mode: {result.mode} | Time: {result.total_time_s:.3f}s")
    print(f"{'='*80}")

    for receipt in result.receipts:
        status_marker = "✓" if receipt.status == "ok" else "⊘" if receipt.status == "skip" else "✗"
        print(f"{status_marker} {receipt.stage:25s} {receipt.status:8s} {receipt.time_s:.3f}s")

        if receipt.error:
            print(f"    ERROR: {receipt.error}")

        # Print key details for completed stages
        if receipt.status == "ok" and receipt.details:
            if receipt.stage == "01_load":
                details = receipt.details
                print(f"    Π-idempotent: {details.get('pi_idempotent', 'N/A')}")
                print(f"    Train/Test: {details.get('train_count')}/{details.get('test_count')}")
                palette = details.get('palette_proof', {}).get('normalized_palette', [])
                print(f"    Palette: {palette}")
            elif receipt.stage == "02_canonicalize_pose":
                details = receipt.details
                print(f"    Π-idempotent: {details.get('pi_idempotent', 'N/A')}")
                print(f"    Orbit collapse: {details.get('orbit_collapse_ok', 'N/A')}")
                print(f"    Grids canonicalized: {details.get('num_grids', 0)}")
                palette_map = details.get('palette_map', {})
                if palette_map:
                    print(f"    Palette map: {palette_map}")
            elif receipt.stage == "03_infer_invariants":
                details = receipt.details
                print(f"    FREE-invariant: {details.get('free_invariance_ok', 'N/A')}")
                print(f"    Palette size: {details.get('palette_size', 0)}")
                print(f"    Train outputs: {details.get('num_train_outputs', 0)}")
                print(f"    Hash: {details.get('hash_counts', 'N/A')}")
                # Show first grid counts as example
                per_grid = details.get('per_grid_counts', [])
                if per_grid:
                    print(f"    Example counts: {per_grid[0].get('counts', {})}")


def main():
    """Run batch ARC task processing."""
    parser = argparse.ArgumentParser(
        description="ARC-AGI batch task runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to ARC data directory or single task JSON file",
    )
    parser.add_argument(
        "--mode",
        choices=["convex", "cloth", "both"],
        default="convex",
        help="Solver mode (default: convex)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save output grids (optional)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to process (for quick smoke tests)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to save CSV results (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed receipts for each task",
    )

    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.exists():
        print(f"ERROR: Path does not exist: {root_path}")
        sys.exit(1)

    # Create save directory if requested
    save_dir = None
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Outputs will be saved to: {save_dir}")

    # Collect all tasks
    all_tasks = {}

    if root_path.is_file():
        # Single task or corpus file
        all_tasks.update(load_corpus_file(root_path))
    elif root_path.is_dir():
        # Directory: find all JSON files
        corpus_files = [
            "arc-agi_training_challenges.json",
            "arc-agi_evaluation_challenges.json",
            "arc-agi_test_challenges.json",
        ]

        for corpus_file in corpus_files:
            corpus_path = root_path / corpus_file
            if corpus_path.exists():
                print(f"Loading: {corpus_file}")
                all_tasks.update(load_corpus_file(corpus_path))

        # Also load individual test task files
        for task_file in sorted(root_path.glob("test_task_*.json")):
            all_tasks.update(load_corpus_file(task_file))

    if not all_tasks:
        print("ERROR: No tasks found")
        sys.exit(1)

    # Apply limit if requested
    task_items = list(all_tasks.items())
    if args.limit:
        task_items = task_items[: args.limit]

    print(f"\n{'='*80}")
    print(f"ARC BATCH RUNNER")
    print(f"{'='*80}")
    print(f"Total tasks: {len(task_items)}")
    print(f"Mode: {args.mode}")
    print(f"{'='*80}\n")

    # Process all tasks
    results = []
    for idx, (task_id, task_data) in enumerate(task_items, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"Processing {idx}/{len(task_items)}: {task_id}", end="\r")

        result = process_task(task_id, task_data, args.mode, save_dir)
        results.append(result)

        if args.verbose:
            print_receipt_summary(result)

    print(f"Processing {len(task_items)}/{len(task_items)}: DONE{' '*20}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(results)
        print(f"{status:15s}: {count:4d} ({pct:5.1f}%)")

    avg_time = sum(r.total_time_s for r in results) / len(results)
    print(f"\nAverage time per task: {avg_time:.3f}s")

    # Save CSV if requested
    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task_id",
                    "status",
                    "mode",
                    "total_time_s",
                    "receipts_json",
                    "duality_gap",
                    "gamma_max_resid",
                ],
            )
            writer.writeheader()

            for r in results:
                writer.writerow(
                    {
                        "task_id": r.task_id,
                        "status": r.status,
                        "mode": r.mode,
                        "total_time_s": f"{r.total_time_s:.4f}",
                        "receipts_json": json.dumps(
                            [
                                {
                                    "stage": rec.stage,
                                    "status": rec.status,
                                    "time_s": rec.time_s,
                                    "error": rec.error,
                                    "details": rec.details,
                                }
                                for rec in r.receipts
                            ]
                        ),
                        "duality_gap": r.duality_gap or "",
                        "gamma_max_resid": r.gamma_max_resid or "",
                    }
                )

        print(f"\nCSV saved to: {csv_path}")

    # Exit code
    error_count = status_counts.get("error", 0)
    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
