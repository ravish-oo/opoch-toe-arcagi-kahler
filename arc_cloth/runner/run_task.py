"""
Single-task orchestrator for ARC-AGI solving.

Orchestrates: load → canonicalize → infer invariants → build D, Γ → solve → snap.
Each stage prints receipts and can fail-fast on violations.

Spec: docs/anchors/00-vision-universe.md (A0–A2), 01-arc-on-the-cloth.md (§2–10)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from arc_cloth.io.arc_loader import Grid, Task, TaskMetadata, load_task
from arc_cloth.io.canonicalize import canonicalize_task, CanonicalTask
from arc_cloth.model.invariants import (
    infer_color_counts,
    infer_periods,
    infer_symmetries,
    infer_block_codebook,
    ColorCountsInvariant
)
from arc_cloth.model.interfaces import build_interfaces
from arc_cloth.model.potential import build_potential_cvxpy, evaluate_on_train
from arc_cloth.solvers.convex_one_shot import solve_convex


@dataclass
class StageReceipt:
    """Receipt from a single pipeline stage."""
    stage: str
    status: str  # "ok" | "error" | "skip"
    time_s: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TaskResult:
    """Complete result from solving a task."""
    task_id: str
    status: str  # "validated" | "solved" | "error"
    output_grid: Optional[Grid] = None
    receipts: List[StageReceipt] = field(default_factory=list)
    total_time_s: float = 0.0
    mode: str = "convex"

    # Solver-specific (populated in later WOs)
    duality_gap: Optional[float] = None
    gamma_max_resid: Optional[float] = None


def _stage_01_load(path: Path) -> tuple[StageReceipt, Optional[Task], Optional[TaskMetadata]]:
    """
    Stage 1: Load and Π-validate task (WO-01).

    Returns (receipt, task, metadata).
    """
    t0 = time.time()
    try:
        task, metadata = load_task(path)

        # Extract receipts from WO-01 loader
        details = {
            "pi_idempotent": metadata.receipts.get("pi_idempotent", False),
            "spec_compliance": metadata.receipts.get("spec_compliance", {}),
            "palette_proof": metadata.receipts.get("palette_proof", {}),
            "shape_summary": metadata.shape_summary,
            "train_count": len(task["train"]),
            "test_count": len(task["test"]),
        }

        # Verify all spec compliance passed
        spec = metadata.receipts.get("spec_compliance", {})
        if not all(spec.values()):
            failing = [k for k, v in spec.items() if not v]
            return (
                StageReceipt(
                    stage="01_load",
                    status="error",
                    time_s=time.time() - t0,
                    error=f"Spec compliance failed: {failing}"
                ),
                None,
                None,
            )

        # Verify Π-idempotence
        if not metadata.receipts.get("pi_idempotent", False):
            return (
                StageReceipt(
                    stage="01_load",
                    status="error",
                    time_s=time.time() - t0,
                    error="Π-idempotence check failed"
                ),
                None,
                None,
            )

        receipt = StageReceipt(
            stage="01_load",
            status="ok",
            time_s=time.time() - t0,
            details=details,
        )

        return receipt, task, metadata

    except Exception as e:
        return (
            StageReceipt(
                stage="01_load",
                status="error",
                time_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}"
            ),
            None,
            None,
        )


def _stage_02_canonicalize_pose(
    task: Task, metadata: TaskMetadata
) -> tuple[StageReceipt, Optional[CanonicalTask]]:
    """
    Stage 2: D4 pose canonicalization (WO-02).

    Returns (receipt, canonicalized_task).
    """
    t0 = time.time()

    try:
        # Apply D4 canonical pose selection and palette remapping
        canonical_task = canonicalize_task(task)

        # Extract receipts from canonicalization
        canon_meta = canonical_task.get("__meta__", {})

        details = {
            "pi_idempotent": canon_meta.get("pi_idempotent", False),
            "orbit_collapse_ok": canon_meta.get("orbit_collapse_ok", False),
            "chosen_pose": canon_meta.get("chosen_pose", {}),
            "palette_map": canon_meta.get("palette_map", {}),
            "num_grids": len(canon_meta.get("chosen_pose", {})),
        }

        # Verify receipts
        if not canon_meta.get("pi_idempotent", False):
            return (
                StageReceipt(
                    stage="02_canonicalize_pose",
                    status="error",
                    time_s=time.time() - t0,
                    error="Π-idempotence failed for canonical task",
                    details=details,
                ),
                None,
            )

        if not canon_meta.get("orbit_collapse_ok", False):
            return (
                StageReceipt(
                    stage="02_canonicalize_pose",
                    status="error",
                    time_s=time.time() - t0,
                    error="D4 orbit collapse failed",
                    details=details,
                ),
                None,
            )

        receipt = StageReceipt(
            stage="02_canonicalize_pose",
            status="ok",
            time_s=time.time() - t0,
            details=details,
        )

        return receipt, canonical_task

    except Exception as e:
        return (
            StageReceipt(
                stage="02_canonicalize_pose",
                status="error",
                time_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            ),
            None,
        )


def _stage_03_infer_invariants(
    task: Task, metadata: TaskMetadata
) -> tuple[StageReceipt, Optional[Dict[str, Any]]]:
    """
    Stage 3: Infer invariants from train pairs (WO-03+).

    Currently implements:
    - Color counts (WO-03) ✓
    - Periods via autocorrelation (WO-04) ✓
    - Mirror seams and concatenations (WO-05) ✓
    - Block codebook learning (WO-06) ✓

    Future WOs:
    - Component mappings (WO-07+)

    Returns (receipt, invariants_dict).
    """
    t0 = time.time()

    try:
        # WO-03: Extract color counts from train outputs
        color_counts = infer_color_counts(task)

        # WO-04: Extract periods from train outputs
        periods = infer_periods(task["train"])

        # WO-05: Extract mirror seams and concatenations
        symmetries = infer_symmetries(task["train"])

        # WO-06: Extract block codebook (input→output patch map)
        block_codebook = infer_block_codebook(task["train"])

        # Extract receipts
        counts_meta = color_counts.get("__meta__", {})
        periods_meta = periods.get("__meta__", {})
        symmetries_meta = symmetries.get("__meta__", {})
        blockmap_meta = block_codebook.get("__meta__", {})

        details = {
            # Color counts receipts
            "color_counts_free_ok": counts_meta.get("free_invariance_ok", False),
            "palette_size": counts_meta.get("palette_size", 0),
            "num_train_outputs": counts_meta.get("num_train_outputs", 0),
            "hash_counts": counts_meta.get("hash_counts", ""),
            "stable_counts": counts_meta.get("stable_counts", False),
            "stable_props": counts_meta.get("stable_props", False),
            # Periods receipts
            "period_h": periods.get("period_h"),
            "period_v": periods.get("period_v"),
            "periods_free_ok": periods_meta.get("free_invariance_ok", False),
            "stable_h": periods_meta.get("stable_h", False),
            "stable_v": periods_meta.get("stable_v", False),
            "conf_h": periods_meta.get("conf_h", 0.0),
            "conf_v": periods_meta.get("conf_v", 0.0),
            "hash_periods": periods_meta.get("hash_periods", ""),
            # Symmetries receipts
            "mirror_h": symmetries.get("mirror_h", False),
            "mirror_v": symmetries.get("mirror_v", False),
            "concat_axes": symmetries.get("concat_axes", []),
            "symmetries_free_ok": symmetries_meta.get("free_invariance_ok", False),
            "n_h_seams": symmetries_meta.get("n_h_seams", 0),
            "n_v_seams": symmetries_meta.get("n_v_seams", 0),
            "hash_sym": symmetries_meta.get("hash_sym", ""),
            # Block codebook receipts
            "block_size": block_codebook.get("block_size"),
            "blockmap_free_ok": blockmap_meta.get("free_invariance_ok", False),
            "single_valued_ok": blockmap_meta.get("single_valued_ok", False),
            "bijection_ok": blockmap_meta.get("bijection_ok", False),
            "n_blocks_total": blockmap_meta.get("n_blocks_total", 0),
            "coverage": blockmap_meta.get("coverage", 0.0),
            "hash_codebook": blockmap_meta.get("hash_codebook", ""),
            "attempts": blockmap_meta.get("attempts", []),  # Per-k audit trail
        }

        # Verify FREE-invariance for all invariants
        if not counts_meta.get("free_invariance_ok", False):
            return (
                StageReceipt(
                    stage="03_infer_invariants",
                    status="error",
                    time_s=time.time() - t0,
                    error="Color counts not FREE-invariant",
                    details=details,
                ),
                None,
            )

        if not periods_meta.get("free_invariance_ok", False):
            return (
                StageReceipt(
                    stage="03_infer_invariants",
                    status="error",
                    time_s=time.time() - t0,
                    error="Periods not FREE-invariant",
                    details=details,
                ),
                None,
            )

        if not symmetries_meta.get("free_invariance_ok", False):
            return (
                StageReceipt(
                    stage="03_infer_invariants",
                    status="error",
                    time_s=time.time() - t0,
                    error="Symmetries not FREE-invariant",
                    details=details,
                ),
                None,
            )

        if not blockmap_meta.get("free_invariance_ok", False):
            return (
                StageReceipt(
                    stage="03_infer_invariants",
                    status="error",
                    time_s=time.time() - t0,
                    error="Block codebook not FREE-invariant",
                    details=details,
                ),
                None,
            )

        # Build invariants dict
        invariants = {
            "color_counts": color_counts,
            "periods": periods,
            "symmetries": symmetries,
            "block_codebook": block_codebook,
        }

        receipt = StageReceipt(
            stage="03_infer_invariants",
            status="ok",
            time_s=time.time() - t0,
            details=details,
        )

        return receipt, invariants

    except Exception as e:
        return (
            StageReceipt(
                stage="03_infer_invariants",
                status="error",
                time_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            ),
            None,
        )


def _stage_04_build_potential(
    invariants: Optional[Dict[str, Any]], task: Task, metadata: TaskMetadata
) -> tuple[StageReceipt, Optional[Any], Optional[Any]]:
    """
    Stage 4: Build convex potential D and interfaces Γ (WO-07-09).

    Returns (receipt, D_terms, Gamma_constraints).
    """
    t0 = time.time()

    try:
        if invariants is None:
            return (
                StageReceipt(
                    stage="04_build_potential",
                    status="skip",
                    time_s=time.time() - t0,
                    details={"reason": "No invariants available"},
                ),
                None,
                None,
            )

        # Get dimensions from first train output (for probe/testing)
        # For actual solving, use test input dimensions
        first_output = task["train"][0]["output"]
        H = len(first_output)
        W = len(first_output[0]) if H > 0 else 0
        C = 10  # ARC colors 0-9

        # Build Gamma (WO-07)
        gamma = build_interfaces(invariants, H, W, C)
        gamma_meta = gamma.get("__meta__", {})

        # Build D (WO-08)
        A_gamma = gamma.get("A")
        D_prog = build_potential_cvxpy(invariants, H, W, C, A_gamma)
        D_meta = D_prog.get("__meta__", {})

        # Train reproduction check: evaluate D on each train output
        train_D_values = []
        for pair in task["train"]:
            train_grid = np.asarray(pair["output"], dtype=np.int16)
            # Only evaluate if dimensions match
            if train_grid.shape == (H, W):
                D_val = evaluate_on_train(train_grid, invariants, A_gamma)
                train_D_values.append(D_val)

        details = {
            # Gamma receipts
            "gamma_M": gamma_meta.get("M", 0),
            "gamma_rank": gamma_meta.get("rank"),
            "gamma_density": gamma_meta.get("density", 0.0),
            "gamma_shape": gamma_meta.get("shape", [0, 0]),
            "gamma_term_counts": gamma_meta.get("term_counts", {}),
            "gamma_hash_A": gamma_meta.get("hash_A", ""),
            # D receipts (WO-08)
            "D_dcp_ok": D_meta.get("dcp_ok", False),
            "D_num_terms": D_meta.get("num_terms", 0),
            "D_num_constraints": D_meta.get("num_constraints", 0),
            "D_term_weights": D_meta.get("term_weights", {}),
            "D_solver_default": D_meta.get("solver_default", ""),
            "D_train_values": train_D_values,
            "D_train_mean": float(np.mean(train_D_values)) if train_D_values else 0.0,
            "D_train_max": float(np.max(train_D_values)) if train_D_values else 0.0,
        }

        receipt = StageReceipt(
            stage="04_build_potential",
            status="ok",
            time_s=time.time() - t0,
            details=details,
        )

        return receipt, D_prog, gamma

    except Exception as e:
        return (
            StageReceipt(
                stage="04_build_potential",
                status="error",
                time_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            ),
            None,
            None,
        )


def _stage_05_solve_convex(
    D_terms: Optional[Any],
    Gamma: Optional[Any],
    task: Task,
    metadata: TaskMetadata,
) -> tuple[StageReceipt, Optional[np.ndarray]]:
    """
    Stage 5: Solve via CVXPY (WO-09).

    Returns (receipt, X_prob).
    """
    t0 = time.time()

    try:
        if D_terms is None or Gamma is None:
            return (
                StageReceipt(
                    stage="05_solve_convex",
                    status="skip",
                    time_s=time.time() - t0,
                    details={"reason": "No D_terms or Gamma available"},
                ),
                None,
            )

        # Get dimensions from first train output
        first_output = task["train"][0]["output"]
        H = len(first_output)
        W = len(first_output[0]) if H > 0 else 0
        C = 10  # ARC colors 0-9

        # Assemble full program dict: add A_gamma to D_prog
        prog = {**D_terms, "A_gamma": Gamma.get("A")}

        # Solve (WO-09)
        X_prob, solve_receipts = solve_convex(H, W, C, prog, solver="CLARABEL")

        # Assemble details from solver receipts
        details = {
            "status": solve_receipts["status"],
            "optimal_value": solve_receipts["optimal_value"],
            "duality_gap": solve_receipts["duality_gap"],
            "eq_residual_max": solve_receipts["eq_residual_max"],
            "simplex_residual_max": solve_receipts["simplex_residual_max"],
            "solver_name": solve_receipts["solver_name"],
            "solve_time_s": solve_receipts["solve_time_s"],
        }

        receipt = StageReceipt(
            stage="05_solve_convex",
            status="ok",
            time_s=time.time() - t0,
            details=details,
        )

        return receipt, X_prob

    except Exception as e:
        return (
            StageReceipt(
                stage="05_solve_convex",
                status="error",
                time_s=time.time() - t0,
                details={"error": str(e)},
            ),
            None,
        )


def _stage_06_solve_cloth(
    D_terms: Optional[Any],
    Gamma: Optional[Any],
    task: Task,
    metadata: TaskMetadata,
) -> tuple[StageReceipt, Optional[np.ndarray]]:
    """
    Stage 6: Solve via Kähler-Hessian descent (WO-11-12, stub for now).

    Returns (receipt, solution_grid).
    """
    t0 = time.time()

    # TODO (WO-11-12): Solve on product-simplex manifold
    # - Use Pymanopt/Geomstats for Riemannian descent
    # - Natural gradient under g = ∇²D
    # - Stop at FY-tightness tolerance
    # - Verify orthogonality and ledger receipts

    receipt = StageReceipt(
        stage="06_solve_cloth",
        status="skip",
        time_s=time.time() - t0,
        details={"reason": "WO-11-12 not yet implemented"},
    )

    return receipt, None


def _stage_07_snap(
    solution: Optional[np.ndarray], task: Task, metadata: TaskMetadata
) -> tuple[StageReceipt, Optional[Grid]]:
    """
    Stage 7: Snap to one-hot via argmax with tie-break (WO-10, stub for now).

    Returns (receipt, discrete_grid).
    """
    t0 = time.time()

    # TODO (WO-10): Snap relaxed solution to discrete
    # - Argmax over color dimension per cell
    # - Lexicographic (row, col, color) tie-break
    # - Convert to integer grid

    receipt = StageReceipt(
        stage="07_snap",
        status="skip",
        time_s=time.time() - t0,
        details={"reason": "WO-10 not yet implemented"},
    )

    return receipt, None


def solve_task(
    path: str | Path,
    mode: Literal["convex", "cloth", "both"] = "convex",
) -> TaskResult:
    """
    Orchestrate full ARC task solving pipeline.

    Pipeline stages:
      01. Load & Π-validate (WO-01) ✓
      02. D4 pose canonicalization (WO-02, stub)
      03. Infer invariants (WO-03-07, stub)
      04. Build D & Γ (WO-08-09, stub)
      05. Solve convex (WO-10, stub)
      06. Solve cloth (WO-11-12, stub)
      07. Snap to discrete (WO-10, stub)

    Args:
        path: Path to task JSON file
        mode: Solver mode ("convex", "cloth", or "both")

    Returns:
        TaskResult with status, output_grid, and receipts.
    """
    path = Path(path)
    task_id = path.stem
    receipts = []
    t0_total = time.time()

    # Stage 1: Load & validate (WO-01) ✓
    receipt, task, metadata = _stage_01_load(path)
    receipts.append(receipt)

    if receipt.status != "ok":
        return TaskResult(
            task_id=task_id,
            status="error",
            receipts=receipts,
            total_time_s=time.time() - t0_total,
            mode=mode,
        )

    # Stage 2: D4 canonicalization (stub)
    receipt, task = _stage_02_canonicalize_pose(task, metadata)
    receipts.append(receipt)

    # Stage 3: Infer invariants (stub)
    receipt, invariants = _stage_03_infer_invariants(task, metadata)
    receipts.append(receipt)

    # Stage 4: Build D & Γ (stub)
    receipt, D_terms, Gamma = _stage_04_build_potential(invariants, task, metadata)
    receipts.append(receipt)

    # Stage 5-6: Solve (stub)
    if mode in ["convex", "both"]:
        receipt, solution_convex = _stage_05_solve_convex(D_terms, Gamma, task, metadata)
        receipts.append(receipt)

    if mode in ["cloth", "both"]:
        receipt, solution_cloth = _stage_06_solve_cloth(D_terms, Gamma, task, metadata)
        receipts.append(receipt)

    # Stage 7: Snap (stub)
    # For now, use whichever solution we have
    solution = None
    if mode == "convex":
        solution = None  # Will be solution_convex when implemented
    elif mode == "cloth":
        solution = None  # Will be solution_cloth when implemented

    receipt, output_grid = _stage_07_snap(solution, task, metadata)
    receipts.append(receipt)

    # Current status: only validated, no solving yet
    status = "validated"

    return TaskResult(
        task_id=task_id,
        status=status,
        output_grid=output_grid,
        receipts=receipts,
        total_time_s=time.time() - t0_total,
        mode=mode,
    )
