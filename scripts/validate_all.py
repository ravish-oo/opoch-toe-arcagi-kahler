#!/usr/bin/env python3
"""
Batch validator for ARC task corpus.

Validates all tasks in ARC-AGI corpus files against the Π-loader.

Usage:
    python scripts/validate_all.py

WO-01: Tests load_task() on all training/evaluation/test datasets.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_cloth.io.arc_loader import load_task, TaskMetadata


def validate_task_dict(task_id: str, task_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate a single task by writing to temp file and loading.

    Returns (success, error_message, receipts).
    """
    try:
        # Write task to temp file (loader expects file path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(task_data, f)
            temp_path = f.name

        try:
            task, metadata = load_task(temp_path)

            # Verify receipts
            if not metadata.receipts.get("pi_idempotent", False):
                return False, "Π-idempotence failed", metadata.receipts

            spec_compliance = metadata.receipts.get("spec_compliance", {})
            if not all(spec_compliance.values()):
                failing = [k for k, v in spec_compliance.items() if not v]
                return False, f"Spec compliance failed: {failing}", metadata.receipts

            return True, "", metadata.receipts

        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    except NotImplementedError as e:
        return False, f"NotImplementedError: {e}", {}
    except ValueError as e:
        return False, f"ValueError: {e}", {}
    except Exception as e:
        return False, f"Unexpected {type(e).__name__}: {e}", {}


def validate_corpus_file(path: Path) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Validate all tasks in a corpus file.

    Returns (total_count, success_count, failures).
    failures is list of (task_id, error_message).
    """
    print(f"\n{'='*80}")
    print(f"Validating: {path.name}")
    print(f"{'='*80}")

    try:
        corpus = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to load {path}: {e}")
        return 0, 0, []

    if not isinstance(corpus, dict):
        print(f"ERROR: {path} is not a dict of tasks")
        return 0, 0, []

    total = len(corpus)
    successes = 0
    failures = []

    for idx, (task_id, task_data) in enumerate(corpus.items(), 1):
        if idx % 50 == 0 or idx == total:
            print(f"  Progress: {idx}/{total} tasks...", end='\r')

        success, error_msg, receipts = validate_task_dict(task_id, task_data)

        if success:
            successes += 1
        else:
            failures.append((task_id, error_msg))

    print(f"  Progress: {total}/{total} tasks... DONE")
    print(f"\n  ✓ Passed: {successes}/{total}")
    print(f"  ✗ Failed: {len(failures)}/{total}")

    return total, successes, failures


def validate_individual_files(data_dir: Path) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Validate individual task JSON files.

    Returns (total_count, success_count, failures).
    """
    print(f"\n{'='*80}")
    print(f"Validating individual task files in: {data_dir}")
    print(f"{'='*80}")

    task_files = sorted(data_dir.glob("test_task_*.json"))

    total = len(task_files)
    successes = 0
    failures = []

    for path in task_files:
        try:
            task, metadata = load_task(path)

            # Verify receipts
            if not metadata.receipts.get("pi_idempotent", False):
                failures.append((path.name, "Π-idempotence failed"))
                continue

            spec_compliance = metadata.receipts.get("spec_compliance", {})
            if not all(spec_compliance.values()):
                failing = [k for k, v in spec_compliance.items() if not v]
                failures.append((path.name, f"Spec compliance failed: {failing}"))
                continue

            successes += 1

        except Exception as e:
            failures.append((path.name, f"{type(e).__name__}: {e}"))

    print(f"\n  ✓ Passed: {successes}/{total}")
    print(f"  ✗ Failed: {len(failures)}/{total}")

    return total, successes, failures


def main():
    """Run batch validation on all ARC datasets."""
    data_dir = Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Track overall stats
    all_failures = []
    total_tasks = 0
    total_successes = 0

    # Corpus files to validate
    corpus_files = [
        "arc-agi_training_challenges.json",
        "arc-agi_evaluation_challenges.json",
        "arc-agi_test_challenges.json",
    ]

    print("\n" + "="*80)
    print("ARC-AGI CORPUS VALIDATION (WO-01)")
    print("="*80)

    # Validate individual test files first
    ind_total, ind_success, ind_failures = validate_individual_files(data_dir)
    total_tasks += ind_total
    total_successes += ind_success
    all_failures.extend([(f"individual:{name}", msg) for name, msg in ind_failures])

    # Validate each corpus file
    for corpus_file in corpus_files:
        path = data_dir / corpus_file
        if not path.exists():
            print(f"\nWARNING: {corpus_file} not found, skipping")
            continue

        count, successes, failures = validate_corpus_file(path)
        total_tasks += count
        total_successes += successes
        all_failures.extend([(f"{corpus_file}:{task_id}", msg) for task_id, msg in failures])

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tasks checked: {total_tasks}")
    print(f"✓ Passed:           {total_successes} ({100*total_successes/total_tasks:.1f}%)")
    print(f"✗ Failed:           {len(all_failures)} ({100*len(all_failures)/total_tasks:.1f}%)")

    # Show first 50 failures
    if all_failures:
        print(f"\n{'='*80}")
        print(f"FAILURES (showing first 50 of {len(all_failures)})")
        print(f"{'='*80}")
        for task_id, error_msg in all_failures[:50]:
            print(f"\n{task_id}")
            print(f"  → {error_msg}")

    # Exit code
    sys.exit(0 if len(all_failures) == 0 else 1)


if __name__ == "__main__":
    main()
