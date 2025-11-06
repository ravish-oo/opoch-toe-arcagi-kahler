## Bottom-up implementation plan (atomic WOs)

Each WO stands alone, produces shippable code, and has acceptance checks against real ARC JSON. No future dependency stubs. Order is buildable straight through.

### WO-01 — ARC loader (Π inputs) ✅ COMPLETE

**Goal:** Parse one ARC task JSON reliably.
**File:** `io/arc_loader.py`
**API:** `load_task(path) -> {"train":[{"input","output"}], "test":[{"input"}]}`
**Acc:**

* Validates rectangular grids, entries ∈[0..9], sizes 1..30 as per ARC repo. ([GitHub][1])
* Fails fast on malformed JSON.
* Receipt: “format OK” flag and shape summary.

### WO-02 — Canonicalizer (Π idempotence) ✅ COMPLETE

**Goal:** D4 pose + palette canonicalization.
**File:** `io/canonicalize.py`
**API:** `canonicalize_task(task)->task_c`
**Acc:**

* Π idempotence: canonicalize twice == once.
* D4 transforms are recognized; palette remapped to contiguous [0..C-1].
* Receipt: hash before/after; record chosen pose.

### WO-03 — Invariants: color counts ✅ COMPLETE

**Goal:** Extract per-color totals from train outputs.
**File:** `model/invariants.py`
**API:** `infer_color_counts(train)->{color_counts}`
**Acc:**

* Invariance under D4 on trains (apply D4 to both sides, counts unchanged).
* Receipt: counts vector and tolerance.

### WO-04 — Invariants: periods (H/V) ✅ COMPLETE

**Goal:** Detect fundamental row/col periods by autocorr.
**API:** `infer_periods(train)->{period_h, period_v}`
**Acc:**

* Stable across all train pairs or return `None`.
* D4 invariance (rotation swaps h/v as expected).
* Receipt: period(s) with confidence score.

### WO-05 — Invariants: mirror / concat flags ✅ COMPLETE

**Goal:** Detect exact mirror seams and band concats.
**API:** `infer_symmetries(train)->{mirror_h, mirror_v, concat_axes}`
**Acc:**

* Verifies equality across seam on outputs.
* D4 invariance respected.
* Receipt: seam indices.

### WO-06 — Invariants: minimal block map (2×2 or 3×3) ✅ COMPLETE

**Goal:** Learn tiny patch dictionary if single-valued across trains.
**API:** `infer_block_codebook(train)->{block_size, codebook}`
**Acc:**

* Only emits if bijective/consistent across pairs; else `None`.
* Receipt: hash of codebook and coverage.

> These 4 invariant WOs are small; together ~300–500 LOC.

### WO-07 — Γ builder (seams as linear equalities) ✅ COMPLETE

**Goal:** Build interface constraints as A·vec(X)=b.
**File:** `model/interfaces.py`
**API:** `build_interfaces(inv,H,W,C)->Interfaces{A,b}`
**Acc:**

* Assembles mirror, period overlaps, concat ties, block overlaps as linear equalities.
* Receipt: rank(A), #constraints.

### WO-08 — D builder (CVXPY backend) ✅ COMPLETE

**Goal:** Compose one convex objective from invariants.
**File:** `model/potential.py`
**API:** `build_potential_cvxpy(inv,H,W,C)->ConvexProgram{X,objective,constraints}`
**Acc:**

* Passes CVXPY DCP checks; uses only atoms from docs. ([CVXPY][3])
* Reproduces **train** outputs with objective ≈ 0 (within penalty choice).
* Receipt: DCP-valid flag; solver status/duality gap.

### WO-09 — Convex solve 

**Goal:** Solve one task via CVXPY once.
**File:** `solvers/convex_one_shot.py`
**API:** `solve_convex(H,W,C,prog,solver="ECOS")->X_prob`
**Acc:**

* Optimal status; equality constraints satisfied to tol.
* Returns X_prob in [0,1] with per-cell simplex sum 1.
* Receipt: duality gap and constraint residuals.

### WO-10 — Snap (deterministic)

**Goal:** Project X_prob→one-hot with frozen tie-break.
**File:** `solvers/snap.py`
**API:** `snap_to_one_hot(X_prob, tie_break="lex")->X_int`
**Acc:**

* Deterministic equal-max handling (lex over color).
* Receipt: hash of final grid.

### WO-11 — D builder (Cloth backend)

**Goal:** Same D as CVXPY but as value/grad/(hvp).
**File:** `model/potential.py`
**API:** `build_potential_cloth(inv,H,W,C)->ClothPotential{value,grad,hess_vec?}`
**Acc:**

* Numerical agreement with CVXPY objective at random X_prob samples.
* Grad check: finite diff vs grad within tol.

### WO-12 — Manifold: product-simplex + Pymanopt problem

**Goal:** Define product of simplexes manifold and problem.
**File:** `solvers/cloth_ng.py`
**API:** `solve_on_cloth(test_in,potential,interfaces,C,...)->X_prob`
**Acc:**

* Uses Pymanopt `Problem(manifold, cost)`; either explicit grad/HVP or autodiff decorator. ([Pymanopt][4])
* Converges on train outputs (min D near 0).
* Receipt: iterations, ΔD history, gradient norm.

*(Geomstats context provides manifold/metric background; we don’t need heavy use beyond the product-simplex modeling here.) ([Geomstats][5])*

### WO-13 — Runner glue

**Goal:** End-to-end CLI.
**File:** `runner/run_task.py`
**API:** `solve_task(path, mode={"convex","cloth"}) -> X_int`
**Acc:**

* Runs both paths, snaps, and prints PNG/JSON.
* On sample tasks, cloth == convex outputs.

### WO-14 — Receipts tests (real data)

**Goal:** Minimal tests that ship receipts.
**Files:** `tests/test_receipts.py`
**Acc:**

* Π idempotence on a few tasks.
* FREE: invariants unchanged under D4.
* Γ: A·vec(X)=b within tol after solve.
* FY (convex): DCP-valid; gap~0 on trains.
* Agreement: cloth vs convex equal after snap.

### WO-15 — Starter examples

**Goal:** One notebook to demo both paths on 2–3 public tasks.
**File:** `examples/notebook_arc_walkthrough.ipynb`
**Acc:**

* Shows invariants found, Γ built, D composed, solution produced.

---

## Why this will stay “minimal-commit”

* Π/D4 applied up-front: maximal use of free moves (no cost).
* Γ forbids non-solutions by construction.
* One paid descent (cloth) or one convex program (one-shot) selects the unique truth; no search loops.
* Receipts wired into each WO to catch drift early.

## Build notes to keep Claude Code honest

* Keep WOs ≤300 LOC target, ≤500 LOC cap.
* Never ask it to “invent APIs.” All signatures above are fixed.
* Use ARC JSON as the test harness early; don’t defer integration.
* CVXPY: adhere to DCP; if a term isn’t an atom composition in the docs, don’t add it. ([CVXPY][3])
* Pymanopt: prefer its autodiff decorators to avoid manual calculus bugs; the quickstart shows the pattern. ([Pymanopt][4])

If you want, I can generate the exact function stubs (docstrings, type hints) for WO-01..05 so you can paste them and start coding immediately.

[1]: https://github.com/fchollet/ARC-AGI?utm_source=chatgpt.com "fchollet/ARC-AGI: The Abstraction and Reasoning Corpus"
[2]: https://martinfowler.com/articles/scaling-architecture-conversationally.html?utm_source=chatgpt.com "Scaling the Practice of Architecture, Conversationally"
[3]: https://www.cvxpy.org/tutorial/dcp/index.html?utm_source=chatgpt.com "Disciplined Convex Programming -"
[4]: https://pymanopt.org/docs/stable/quickstart.html?utm_source=chatgpt.com "Quickstart — Pymanopt stable (2.2.1) documentation"
[5]: https://geomstats.github.io/geomstats.github.io/index.html?utm_source=chatgpt.com "Geomstats — Geomstats latest documentation"
