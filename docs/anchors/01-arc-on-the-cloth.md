# 01 — ARC on the Kähler–Hessian Cloth (spec, not vibes)

> This binds the ARC-AGI benchmark to the universe engine in `00-vision-universe.md`. No feature may contradict this mapping.

## 0) Scope

* Target corpus: ARC-AGI (v1 and v2), tasks are tiny colored grids in JSON with `train` pairs and a `test` input. Colors are digits 0–9. Each task has 3–5 train pairs and one test pair. ([GitHub][1])
* ARC intent: measure skill-acquisition efficiency and compositional generalization from few examples, not operator catalogs. ([arXiv][2])

## 1) Data model (ground truth)

* A task JSON has two top-level keys:

  * `"train"`: list of dicts `{ "input": grid, "output": grid }`
  * `"test"`: list with one dict `{ "input": grid }`
    Grids are 2-D arrays of ints in `[0..9]`. ([GitHub][1])

## 2) Cloth mapping (Z, G_free, D, Γ, E)

* **Z (state space):** the unknown test output grid as a **product of simplexes**: for each cell `(i,j)`, a probability vector over `C` colors (C ≤ 10). During solve X ∈ [0,1]^{H×W×C}, with per-cell simplex constraints; snap to one-hot at the end.
* **G_free (free symmetries):** D4 pose group on grids (rotations, flips), optionally translations when task semantics allow. These are exact isometries; pose selection is free. ([ARC Prize][3])
* **D (paid potential):** one convex functional encoding the invariants extracted from all train pairs (see §3). Minimize D on the test input modulo Γ. No other “operators.”
* **Γ (interfaces):** linear equalities enforcing overlap consistency where local laws meet (row/patch/component seams). Composition is Schur on these bands.
* **E (free energies):** typically none for ARC; optional if a continuous reparameterization leaves D invariant.

Engine (from 00-doc): (\dot z = J\nabla E(z) - g^{-1}\nabla D(z)); here we primarily use the paid descent term (natural gradient) after quotienting by G_free.

## 3) Invariants catalogue v1 → convex terms for D

ARC patterns must become convex terms (DCP-valid) or exact equalities, not bespoke ops. CVXPY’s DCP rules/atoms apply. ([CVXPY][4])

For each item: **detect from trains**, **express in D**:

1. **Color histogram / area preservation**

   * Detect: per-color counts on outputs given inputs.
   * D term: (\sum_c w_c, \lvert \sum_{ij} X_{ijc} - \hat n_c \rvert) or squared form.
   * Notes: linear/sum + abs or squared → DCP. ([CVXPY][5])

2. **Tiling periodicity (1D/2D)**

   * Detect: fundamental period via autocorrelation on rows/cols.
   * D term: penalties forcing equality between cells separated by period p along the axis: (\sum |X_{i,j,:} - X_{i,j+p,:}|_1). Linear+norm → DCP.

3. **Mirror / reflection**

   * Detect: equality across a mirror seam in trains.
   * Γ equalities or D penalty: (X_{i,j,:} = X_{i,,j^\star,:}) where (j^\star) is mirrored index. Linear equality → exact.

4. **Concatenation / copy-paste bands**

   * Detect: output bands equal to translated bands of input.
   * D term: linear equalities/penalties tying bands to their sources.

5. **Block substitution (patch dictionary)**

   * Detect: small k×k input patches map to output patches; learn a finite codebook.
   * D term: linear selection consistency across occurrences (one-hot over codewords) or convex relaxation with simplex on codeword weights.

6. **Component mapping (connected components)**

   * Detect: 4-conn components in input map to placements/colors in output.
   * D term: per-component consistency penalties (same color choice across all cells of a mapped component), linearized via component masks.

7. **Line/border consistency (when present)**

   * Detect: lines or frames as in many ARC tasks.
   * D term: anisotropic TV-like penalty to enforce straight segments; keep convex.

All terms are convex or exact equalities; combine with nonnegative weights inferred from train residuals. CVXPY must remain DCP-compliant. ([CVXPY][4])

## 4) Interfaces Γ (lawful gluing)

* Purpose: ensure local laws agree globally. Γ includes:

  * Patch overlap equalities (tilings, block substitution).
  * Row/column seam equalities (mirror/concat).
  * Component-wise equalities (mapped parts share one color).
* Implementation: linear equalities (A X = b) over the flattened X tensor; Schur elimination used when solving blockwise.

## 5) Solver modes (two equivalent paths)

* **Universe-aligned cloth descent (paid move):**

  * Z as product of simplexes; metric (g=\nabla^2 D) in paid chart; run natural-gradient (Riemannian) descent on X under Γ as hard equalities or via penalties. Use Pymanopt (manifold optimisation) over a product-simplex manifold from Geomstats. ([arXiv][6])
* **Convex one-shot mirror:**

  * CVXPY variable (X\in[0,1]^{H\times W\times C}); per-cell (\sum_c X_{ijc}=1); add D terms and Γ equalities; solve once with a cone/QP/SOCP solver; project to one-hot. DCP rules apply. ([CVXPY][4])

Both must agree on well-posed tasks (up to tie-break). The one-shot path is a correctness mirror.

## 6) Free symmetries (G_free)

* Apply D4 pose changes and allowed translations **before** solving (Π). Choose a canonical pose (e.g., lexicographic min of bytes) and remap palette. These are 0-bit isometries; they must not change D nor the invariants. ([ARC Prize][3])

## 7) Discretization and tie-break

* After solve, snap each cell’s simplex to one-hot by `argmax` with a **single frozen tie-break**: lexicographic `(row, col, color)`. Deterministic. No sampling.

## 8) Receipts (task-level)

* Π idempotence: canonicalize(task) twice = once.
* FREE: D4 transforms leave D terms invariant on trains.
* FY: on each train pair, D at the ground-truth output ≈ 0; duality gap ~ 0 for convex solve (when used).
* Γ: all seam equalities satisfied in the returned X.
* Agreement: cloth descent and CVXPY outputs match after snap.

## 9) What is out-of-scope in v1

* Nonconvex global topology (exact Euler characteristic, etc.) as hard constraints. Use convex proxies or skip in v1.
* MILP/ILP backends. Keep v1 pure convex + manifold descent.
* Per-task branching or operator catalogs.

(We will revisit once the v1 cloth pipeline is stable.)

## 10) Minimal worked mapping (informative, not executable)

Given trains that show a 2-period horizontal tiling and vertical mirror:

* Detect: period p=2 on rows; vertical mirror about midline; color counts preserved.
* Build D:

  * tiling: (\sum_{i,j} |X_{i,j,:} - X_{i,j+2,:}|_1)
  * mirror: equalities (X_{i,j,:} = X_{i,j^\star,:})
  * counts: (\sum_c \big|\sum_{ij} X_{ijc} - \hat n_c\big|)
* Γ: enforce mirror seam and tile overlaps.
* Solve once (cloth descent or CVXPY), snap to one-hot.

## 11) Citations / canonical sources

* ARC repo (format, JSON keys train/test). ([GitHub][1])
* ARC Prize guide (grids as JSON of ints; 3–5 trains; one test). ([ARC Prize][3])
* Chollet 2019 “On the Measure of Intelligence” (ARC motivation and priors). ([arXiv][2])
* Kaggle data pages for competition files. ([Kaggle][7])
* CVXPY DCP rules and atoms (convexity guarantees). ([CVXPY][4])

---

**Enforcement note:** Any PR that adds a procedure not expressible as:

* a D4 isometry or allowed shift (free),
* a convex term in D, or
* a linear Γ equality

…will be rejected per `00-vision-universe.md` (fluff rule).

## References
[1]: https://github.com/fchollet/ARC-AGI?utm_source=chatgpt.com "fchollet/ARC-AGI: The Abstraction and Reasoning Corpus"
[2]: https://arxiv.org/abs/1911.01547?utm_source=chatgpt.com "On the Measure of Intelligence"
[3]: https://arcprize.org/guide?utm_source=chatgpt.com "ARC Prize - Guide"
[4]: https://www.cvxpy.org/version/1.4/tutorial/dcp/index.html?utm_source=chatgpt.com "Disciplined Convex Programming"
[5]: https://www.cvxpy.org/version/1.2/tutorial/functions/index.html?utm_source=chatgpt.com "Atomic Functions — CVXPY 1.2 documentation"
[6]: https://arxiv.org/abs/2505.11831?utm_source=chatgpt.com "ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems"
[7]: https://www.kaggle.com/competitions/arc-prize-2024/data?select=sample_submission.json&utm_source=chatgpt.com "ARC Prize 2024 | Kaggle"
