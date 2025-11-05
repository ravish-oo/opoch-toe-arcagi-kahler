#!/usr/bin/env python3
"""
Inspect receipts from loaded tasks in detail.

WO-01: Verify receipts contain all required fields and values are correct.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_cloth.io.arc_loader import load_task


def inspect_task(task_id: str, task_data: dict) -> dict:
    """Load task and return detailed receipt inspection."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(task_data, f)
            temp_path = f.name

        try:
            task, meta = load_task(temp_path)

            # Inspect receipts structure
            receipts = meta.receipts

            result = {
                "task_id": task_id,
                "success": True,
                "receipts": receipts,
                "metadata": {
                    "shape_summary": meta.shape_summary,
                    "palette": meta.palette,
                    "palette_map": meta.palette_map,
                },
                "checks": {}
            }

            # Check required receipt fields
            required_fields = ["pi_idempotent", "spec_compliance", "palette_proof"]
            for field in required_fields:
                result["checks"][f"has_{field}"] = field in receipts

            # Check spec_compliance structure
            if "spec_compliance" in receipts:
                spec = receipts["spec_compliance"]
                required_spec_keys = ["has_train", "has_single_test", "rectangular",
                                     "values_in_0_9", "sizes_in_1_30"]
                for key in required_spec_keys:
                    result["checks"][f"spec_has_{key}"] = key in spec
                    if key in spec:
                        result["checks"][f"spec_{key}_value"] = spec[key]

            # Check palette_proof structure
            if "palette_proof" in receipts:
                palette_proof = receipts["palette_proof"]
                result["checks"]["palette_has_original"] = "original_palette" in palette_proof
                result["checks"]["palette_has_normalized"] = "normalized_palette" in palette_proof
                result["checks"]["palette_has_mapping"] = "mapping" in palette_proof

                # Verify palette is contiguous [0..C-1]
                if "normalized_palette" in palette_proof:
                    norm_pal = palette_proof["normalized_palette"]
                    expected = list(range(len(norm_pal)))
                    result["checks"]["palette_is_contiguous"] = norm_pal == expected

            return result

        finally:
            Path(temp_path).unlink()

    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "error": str(e),
            "checks": {}
        }


def main():
    """Inspect receipts from sample tasks."""
    data_dir = Path(__file__).parent.parent / "data"
    corpus_path = data_dir / "arc-agi_training_challenges.json"

    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found")
        sys.exit(1)

    print("="*80)
    print("RECEIPTS INSPECTION (WO-01)")
    print("="*80)

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    # Inspect first 10 tasks in detail
    sample_size = 10
    sample_tasks = list(corpus.items())[:sample_size]

    all_checks_pass = True

    for task_id, task_data in sample_tasks:
        print(f"\n{'─'*80}")
        print(f"Task: {task_id}")
        print(f"{'─'*80}")

        result = inspect_task(task_id, task_data)

        if not result["success"]:
            print(f"❌ FAILED: {result.get('error', 'unknown')}")
            all_checks_pass = False
            continue

        print(f"✅ Loaded successfully")

        # Print metadata summary
        meta = result["metadata"]
        print(f"\nMetadata:")
        print(f"  Shapes: {meta['shape_summary']}")
        print(f"  Palette: {meta['palette']}")
        print(f"  Palette map: {meta['palette_map']}")

        # Print receipts
        print(f"\nReceipts:")
        receipts = result["receipts"]

        print(f"  π-idempotent: {receipts.get('pi_idempotent', 'MISSING')}")

        if "spec_compliance" in receipts:
            print(f"  Spec compliance:")
            for k, v in receipts["spec_compliance"].items():
                print(f"    {k}: {v}")

        if "palette_proof" in receipts:
            pp = receipts["palette_proof"]
            print(f"  Palette proof:")
            print(f"    Original: {pp.get('original_palette', 'MISSING')}")
            print(f"    Normalized: {pp.get('normalized_palette', 'MISSING')}")

        # Print checks
        print(f"\nValidation Checks:")
        checks = result["checks"]
        for check_name, check_value in checks.items():
            status = "✓" if check_value else "✗"
            print(f"  {status} {check_name}: {check_value}")
            if not check_value:
                all_checks_pass = False

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if all_checks_pass:
        print("✅ All receipt checks passed!")
        sys.exit(0)
    else:
        print("❌ Some receipt checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
