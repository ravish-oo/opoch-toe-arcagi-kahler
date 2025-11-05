#!/usr/bin/env python3
"""
Test determinism: load_task twice should produce identical results.

WO-01 Acceptance Criteria B: "Running load_task twice returns identical objects"
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_cloth.io.arc_loader import load_task


def test_determinism_on_task(task_id: str, task_data: dict) -> tuple[bool, str]:
    """Test that loading the same task twice produces identical results."""
    try:
        # Write task to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(task_data, f)
            temp_path = f.name

        try:
            # Load twice
            task1, meta1 = load_task(temp_path)
            task2, meta2 = load_task(temp_path)

            # Compare tasks (deep equality via JSON serialization)
            task1_json = json.dumps(task1, sort_keys=True)
            task2_json = json.dumps(task2, sort_keys=True)

            if task1_json != task2_json:
                return False, "Task objects differ between runs"

            # Compare metadata
            if meta1.shape_summary != meta2.shape_summary:
                return False, "shape_summary differs"
            if meta1.palette != meta2.palette:
                return False, "palette differs"
            if meta1.palette_map != meta2.palette_map:
                return False, "palette_map differs"
            if meta1.receipts != meta2.receipts:
                return False, "receipts differ"

            return True, "OK"

        finally:
            Path(temp_path).unlink()

    except Exception as e:
        return False, f"Exception: {e}"


def main():
    """Test determinism on a sample of tasks."""
    data_dir = Path(__file__).parent.parent / "data"
    corpus_path = data_dir / "arc-agi_training_challenges.json"

    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found")
        sys.exit(1)

    print("="*80)
    print("DETERMINISM TEST (WO-01 Criterion B)")
    print("="*80)

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    # Test first 50 tasks
    test_count = min(50, len(corpus))
    tasks_to_test = list(corpus.items())[:test_count]

    failures = []
    for idx, (task_id, task_data) in enumerate(tasks_to_test, 1):
        success, msg = test_determinism_on_task(task_id, task_data)
        if not success:
            failures.append((task_id, msg))
        if idx % 10 == 0:
            print(f"Progress: {idx}/{test_count}...", end='\r')

    print(f"Progress: {test_count}/{test_count}... DONE")
    print(f"\n✓ Deterministic: {test_count - len(failures)}/{test_count}")
    print(f"✗ Non-deterministic: {len(failures)}/{test_count}")

    if failures:
        print("\nFAILURES:")
        for task_id, msg in failures:
            print(f"  {task_id}: {msg}")
        sys.exit(1)
    else:
        print("\n✅ All tasks are deterministic!")
        sys.exit(0)


if __name__ == "__main__":
    main()
