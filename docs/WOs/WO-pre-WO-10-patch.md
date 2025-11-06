## WO-A1 — Color budgets that scale (patch WO-03 + pickup in WO-08)

**Anchors:** 00, 01, 03, 04
**Why:** Histogram term is critical; without a task-level target the solver drifts.
**Libs we reuse:**

* **NumPy `bincount`** to count non-negative ints with `minlength=C` (canonical API). ([NumPy][1])

**Implement (WO-03):**

* For each Π-normalized train **output** grid `G_t`, compute:

  * `counts_t = np.bincount(G_t.ravel(), minlength=C)` (all 0..9 covered). ([NumPy][1])
  * `props_t = counts_t / G_t.size`.
* Emit **exactly one** of:

  * `color_counts` if **all** `counts_t` are identical **and** areas equal; else
  * `color_props` if **all** `props_t` are identical; else neither.
* Receipts: `stable_counts | stable_props`, plus SHA-256 of whichever target you emit.

**Pickup (WO-08):**

* If `color_counts` present: use it directly in
  `D_hist = cp.norm1(sum_{ij} X[i,j,:] − n_hat)` (atoms: `sum`, `norm1`, DCP-valid). ([CVXPY][2])
* Else if `color_props` present: compute `n_hat = round(color_props * (H*W))` **in NumPy**, then same `norm1` term (rounding outside CVXPY; only atoms inside).

**Runner:**

* `--check-invariants color_counts` must show at least one of `color_counts | color_props` (or cleanly none) for 100% tasks.
* `--check-D` logs `dcp_ok` and `D_train_mean≈0` when train outputs are probed as Constants.

**Checklist:** mature libs only ✔︎ (NumPy `bincount`), receipts ✔︎, corpus-wide pass rule ✔︎, anchored ✔︎.

---

## WO-A2 — Mirror seams = **midline-only** (patch WO-05) and Γ hookup (WO-07)

**Anchors:** 00, 01, 03, 04
**Why:** Absolute seam indices pooled across different-sized grids are meaningless.
**Libs:**

* **NumPy flips** `fliplr/flipud` (exact views; O(1) to create). ([NumPy][3])

**Implement (WO-05):**

* For each output grid `G`:

  * `mirror_h_ok = (W%2==0) and np.array_equal(G[:, :W//2], np.fliplr(G[:, W//2:]))`. ([NumPy][3])
  * `mirror_v_ok = (H%2==0) and np.array_equal(G[:H//2, :], np.flipud(G[H//2:, :]))`. ([NumPy][4])
* Task flags: `mirror_h = all(mirror_h_ok)`, `mirror_v = all(mirror_v_ok)`.
* Return `mirror_h_seams = ["mid"] if mirror_h else []` and similarly for `mirror_v_seams`.

**Γ (WO-07):**

* If `"mid"` present and parity holds: compute seam = `W//2` or `H//2` for **current** `(H,W)` and tie pairs.
* Sparse build in **COO → CSR**; duplicates are **summed** by SciPy by design. ([docs.scipy.org][5])

**Runner:**

* `--check-invariants symmetries` prints `mirror_*` and `["mid"]` seams; 100% tasks pass.
* `--check-gamma` shows mirror rows only when even width/height.

**Checklist:** mature libs only ✔︎ (NumPy flips, SciPy COO→CSR), receipts ✔︎, anchored ✔︎.

---

## WO-A3 — **Row-major** vec convention lock (WO-07, WO-08, WO-09)

**Anchors:** 00, 01, 04
**Why:** CVXPY’s default reshape is column-major; our Γ rows were built for **row-major** vec.
**Libs:**

* **NumPy `ravel_multi_index(order='C')`** to encode `(i,j,c) → flat` indices. ([NumPy][6])
* **NumPy reshape `order='C'`** and CVXPY reshape `order='C'` to keep consistent vec convention. ([NumPy][7])

**Implement:**

* WO-07: use `np.ravel_multi_index(..., order='C')` everywhere (documented order flag). ([NumPy][6])
* WO-08: `X_vec = cp.reshape(X, (H*W*C,1), order='C')` (CVXPY reshape doc shows order semantics). ([NumPy][8])
* WO-09: `X_vec = X_prob.reshape(H*W*C, order='C')` before residuals. ([NumPy][7])

**Runner:**

* `--mode convex` must report `eq_residual_max ≤ 1e-6` on solved tasks across the corpus.

**Checklist:** mature docs only ✔︎, receipts ✔︎, corpus-wide gate ✔︎, anchored ✔︎.

---

## WO-A4 — Period unanimity **including None** (patch WO-04)

**Anchors:** 00, 03, 04
**Why:** Stability must be 100% across outputs; filtering `None` violates the rule.
**Libs:** none new (pure Python).

**Implement:**

* `across(vals, n)` returns stable only if **no** `None` in `vals` **and** all equal; else `(None, False, conf_over_n)`.

**Runner:**

* `--check-invariants periods` must enforce: if `stable_* == True`, then `conf_* == 1.0` (hard gate, 100% tasks pass receipts).

**Checklist:** receipts ✔︎, anchored ✔︎.

---

## WO-A5 — Output shape helper (08a) **inside WO-08**

**Anchors:** 00, 01, 03, 04
**Why:** ARC test output size varies; we must pick `(H*,W*)` before building (X).
**Libs:** none new (pure NumPy/loop).

**Implement:**

* Enumerate ≤ 5 **candidates** using:

  * train size relations (same-size, swap, tiny affine with small ints),
  * invariant constraints:
    divisibility by period p/q; evenness if `mirror_*` or `concat_*`; divisibility by block k.
  * enforce ARC bounds 1..30 for H and W.
* For each candidate: build (D, Γ) (no solve) and **evaluate (D)** on each train output Constant (its one-hot) **reshaped to that candidate** only when the example’s rule implies this size (else score ∞). Pick the candidate with **lowest total (D)** (tie: smallest area).
* Return chosen `(H*,W*)` plus the candidate table in receipts.

**Runner:**

* `--check-D` prints candidate list and the chosen one per task, so the reviewer sees immediate impact.

**Checklist:** anchored ✔︎, receipts ✔︎, no heavy compute ✔︎.

---

## WO-A6 — CVXPY solve + residual receipts + snap probe (WO-09 + light WO-10)

**Anchors:** 00, 01, 04
**Why:** This is where matches begin (once WO-10 snap runs).
**Libs:**

* **CVXPY** Problem/solve/DCP/stats (official tutorial & API). ([CVXPY][9])
* **Solvers:** **CLARABEL** default; **OSQP** (QP), **SCS** (cones) — all shipped in CVXPY. ([docs.scipy.org][10])
* **SciPy sparse**: CSR residuals, `sum_duplicates()` if needed. ([docs.scipy.org][11])

**Implement:**

* Guard: `prob.is_dcp()` True (DCP checker). ([CVXPY][2])
* Solve with chosen solver; collect `status`, `solver_stats.solve_time`, `duality_gap` if exposed.
* **Residuals:**

  * simplex: `max|∑_c X[i,j,c] − 1|`
  * Γ: `max|A @ vec_C(X_prob)|` (`order='C'`).
* Return `(X_prob, receipts)`; add a **snap probe** (argmax + frozen tie-break) to count exact matches vs train outputs when shapes align (no JSON write yet).

**Runner:**

* `--mode convex` prints `status/gap/eq_resid/simplex_resid/time` and `%train_matched`.
* 100% tasks must produce receipts (no exceptions); unsatisfied tasks will show non-optimal status, not crashes.

**Checklist:** mature libs only ✔︎ (CVXPY, SciPy), receipts ✔︎, anchored ✔︎, CPU-friendly ✔︎.

---

## One-screen reviewer script additions

* `--check-invariants color_counts | periods | symmetries | blockmap` (already defined in earlier WOs).
* `--check-gamma` prints Γ receipts: `M, rank, density, term_counts`.
* `--check-D` prints DCP status and train-probe values.
* `--mode convex` prints solve receipts and percentage matched on trains.

---

## What to read before each WO (anchors)

* Always skim: `docs/anchors/00-vision-universe.md`, `docs/anchors/01-arc-on-the-cloth.md`.
* Then the specific anchor:

  * **A1:** `03-invariants-catalog-v1.md` §1; `04-receipts-checklist.md`.
  * **A2:** `03` §§3–4; `04`.
  * **A3:** `02-impl-contracts.md` (Γ vectorization); `04`.
  * **A4:** `03` §2; `04`.
  * **A5:** `01` (§2 on Z/G_free/Γ/D, and size variability); `03`.
  * **A6:** `03` (terms), `04` (DCP & residuals).

---

## Citations index (APIs Claude must use, not invent)

* NumPy `bincount` (counts, `minlength`) — mature counting primitive. ([NumPy][1])
* NumPy `rot90`, `fliplr`, `flipud` — exact isometries on integer grids. ([NumPy][8])
* NumPy `ravel_multi_index(order='C')`, `reshape(order='C')` — row-major vec convention. ([NumPy][6])
* SciPy sparse **COO→CSR**, `sum_duplicates()` — duplicates **sum** (lawful GLUE rows). ([docs.scipy.org][5])
* SciPy sparse `kron` — available if a Kronecker block is ever useful. ([docs.scipy.org][10])
* CVXPY **DCP** rules, **atoms** (`sum`, `norm1`, `sum_squares`, `reshape`, `total_variation`), **Problem/solve/stats**, **solvers** (CLARABEL/OSQP/SCS). ([CVXPY][2])

---

[1]: https://numpy.org/doc/2.1/reference/generated/numpy.bincount.html?utm_source=chatgpt.com "numpy.bincount — NumPy v2.1 Manual"
[2]: https://www.cvxpy.org/tutorial/dcp/index.html?utm_source=chatgpt.com "Disciplined Convex Programming -"
[3]: https://numpy.org/doc/2.1/reference/generated/numpy.fliplr.html?utm_source=chatgpt.com "numpy.fliplr — NumPy v2.1 Manual"
[4]: https://numpy.org/doc/2.2/reference/generated/numpy.flipud.html?utm_source=chatgpt.com "numpy.flipud — NumPy v2.2 Manual"
[5]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html?utm_source=chatgpt.com "coo_matrix — SciPy v1.16.2 Manual"
[6]: https://numpy.org/doc/2.1/reference/generated/numpy.ravel_multi_index.html?utm_source=chatgpt.com "numpy.ravel_multi_index — NumPy v2.1 Manual"
[7]: https://numpy.org/doc/2.3/reference/generated/numpy.ndarray.reshape.html?utm_source=chatgpt.com "numpy.ndarray.reshape — NumPy v2.3 Manual"
[8]: https://numpy.org/doc/2.2/reference/generated/numpy.rot90.html?utm_source=chatgpt.com "numpy.rot90 — NumPy v2.2 Manual"
[9]: https://www.cvxpy.org/tutorial/?utm_source=chatgpt.com "User Guide -"
[10]: https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.sparse.kron.html?utm_source=chatgpt.com "kron — SciPy v1.15.2 Manual"
[11]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sum_duplicates.html?utm_source=chatgpt.com "sum_duplicates — SciPy v1.16.2 Manual"
