# 03 — Invariants Catalog (v1) for ARC on the Cloth

> This document lists the only invariants we implement in v1, how to **detect** each from ARC train pairs, and how to encode it as a **convex term for D** and/or **linear Γ equalities**. Everything here maps to the ARC JSON format and the cloth mechanics in 00/01. If it isn’t here, it doesn’t ship in v1.

**ARC ground truth.** Tasks are tiny colored grids in JSON: `"train"` is a list of input→output pairs, `"test"` has one input. Colors are ints 0–9. We infer from a few train pairs, then produce the single test output. ([GitHub][1])

**Convexity guard.** Every D-term is DCP-valid in CVXPY (atoms + rules), and differentiable on the simplex interior for the cloth descent. ([CVXPY][2])

---

## Palette & pose (Π prelude)

* **What:** Canonicalize D4 pose (rot/flip) and palette remap to contiguous `[0..C-1]`.
* **Why:** Free isometries mint no differences; invariants must be pose-agnostic.
* **Receipt:** Π idempotence (canonicalize twice = once).
  ARC JSON spec confirms the format we normalize from. ([GitHub][1])

---

## 1) Color histogram / area preservation

**Intuition.** Many tasks conserve per-color totals across input→output.

**Detect (train)**
Sum counts by color on outputs; if stable across train pairs (up to small noise like cropping/growth), mark as active.

**D term (two choices)**

* **L1**: ( D_{\text{hist}}=\sum_{c} w_c,\big|\sum_{i,j} X_{ijc} - \hat n_c\big| )
* **L2**: ( D_{\text{hist}}=\sum_{c} w_c,\big(\sum_{i,j} X_{ijc} - \hat n_c\big)^2 )

**Γ** None.

**DCP:** sums, abs, squares are atomic/convex. ([CVXPY][3])

---

## 2) Tiling / periodicity (1D or 2D)

**Intuition.** Rows/cols (or blocks) repeat with a fundamental period.

**Detect (train)**
Per row/col, compute discrete autocorrelation; pick smallest nontrivial lag with strong alignment that is consistent across pairs. (We only use positive detections.) ARC is designed to surface such minimal priors from few examples. ([arXiv][4])

**D term (row-period p example)**
( D_{\text{tile-h}}=\sum_{i,j}\big|X_{i,j,:}-X_{i,j+p,:}\big|_{1}) over valid windows.
Similarly for vertical (period q) or 2D blocks.

**Γ** Optional: promote some equalities to hard constraints at seams.

**DCP:** L1 of affine differences is convex. ([CVXPY][2])

---

## 3) Mirror / reflection symmetry

**Intuition.** Output mirrors along a seam (vertical/horizontal).

**Detect (train)**
Test equality across candidate seams on outputs (indices reflect around midline); adopt only if exact across pairs.

**Γ (preferred)**
Linear equalities: (X_{i,j,:}=X_{i,j^*,:}) for mirrored index (j^*). Exact, no slack.

**D term (backup)**
Squared penalty on the same equalities if we choose soft form.

**DCP:** Linear equalities and squared norms are DCP-valid. ([CVXPY][2])

---

## 4) Concatenation / copy bands

**Intuition.** Output bands equal shifted/cut bands of input or other bands.

**Detect (train)**
Find bands whose RGB (one-hot) signatures match after a fixed shift; confirm across pairs.

**Γ / D**

* **Γ:** hard equalities tying target band cells to source band cells (after shift).
* **D:** optional L1 penalties for soft enforcement when band borders are noisy.

**DCP:** Equalities / L1 are convex. ([CVXPY][2])

---

## 5) Block substitution (patch dictionary)

**Intuition.** Small (k\times k) input patches map to output patches via a finite codebook.

**Detect (train)**
Enumerate (k\times k) patches (small k, e.g., 2 or 3); hash patch bytes; build a codebook input→output if the mapping is single-valued across pairs. Keep only bijective/consistent entries.

**D term (relaxed assignment)**
For each patch position r, introduce codeword weights (\alpha_{r,m}\in\Delta), and reconstruct the target patch as (\sum_m \alpha_{r,m},\Pi_m) (prototype (\Pi_m) is one-hot patch from codebook). Penalize deviation:
( D_{\text{block}}=\sum_r\big|X_{\text{patch}(r)}-\sum_m \alpha_{r,m}\Pi_m\big|*1 )
with simplex constraints (\sum_m \alpha*{r,m}=1, \alpha_{r,m}\ge0).

**Γ** Overlap equalities so adjacent reconstructed patches agree on shared cells (lawful gluing).

**DCP:** L1 with affine mixture + simplex constraints is convex. ([CVXPY][2])

---

## 6) Component consistency (connected components)

**Intuition.** Each 4-connected input component maps coherently (same color or same transform) in output.

**Detect (train)**
Compute 4-connected components in inputs; check each component’s output footprint has uniform color or coherent change across train pairs.

**D term (uniform color case)**
For component mask (M_s), drive a single color choice:
( D_{\text{comp}}=\sum_{s}\sum_{(i,j)\in M_s}\big|X_{ij,:}-\bar X_{M_s}\big|*1 )
where (\bar X*{M_s}) is the mean simplex over the component (implemented via linear averages).

**Γ** Optional hard equalities tying all cells in a component to share the same argmax color index (soft in v1: convex relaxation).

**DCP:** L1 around linear averages is convex. ([CVXPY][2])

---

## 7) Line / border consistency (straight segments & frames)

**Intuition.** Many tasks draw axis-aligned lines or frames.

**Detect (train)**
Hough-lite: scan rows/cols for long constant runs; confirm across pairs.

**D term (anisotropic TV-like)**
Encourage straightness with convex TV:
( D_{\text{tv}}=\sum_{i,j} \big(|X_{i,j,:}-X_{i+1,j,:}|*1+|X*{i,j,:}-X_{i,j+1,:}|_1\big))
Optionally restrict to rows/cols flagged by detection.

**Γ** Equalities at known corners/seams if implied exactly by trains.

**DCP:** TV is convex sum of L1 differences. ([CVXPY][2])

---

## 8) Color remapping / palette logic

**Intuition.** Role-based recoloring (e.g., “largest object becomes color k”).

**Detect (train)**
From outputs, estimate a deterministic color map (\pi) (e.g., rank-by-size → color id).

**D term**
Linearizes as selection over fixed palette indices per role; in v1 we handle only **global remaps** evidenced identically across pairs:
( D_{\text{map}}=\sum_{i,j}|X_{ij,:}-P,Y_{ij,:}|_1 )
where (Y) is the input one-hot (after pose/palette Π) and (P) is a fixed permutation inferred from trains.

**Γ** None.

**DCP:** L1 with fixed permutation matrix is convex. ([CVXPY][2])

---

## 9) Periodic translate / center (simple placement)

**Intuition.** Move or center a component; translation is free if allowed.

**Detect (train)**
Component displacement vectors; if consistent, treat translation as a **free** pose parameter (no cost), not as a paid term.

**G_free / Γ**
Realize the shift as part of Π/pose (apply before solving) or as Γ ties between source component cells and target cells (hard equalities).

**DCP:** Pure equalities if we encode as Γ; otherwise free isometry (no D term). ([ARC Prize][5])

---

## 10) Weights & composition

* Each active term gets a nonnegative weight (w_k).
* v1 default: normalize each term to comparable scale on train pairs; set (w_k=1) unless evidence suggests stronger emphasis (e.g., exact mirror).
* The **overall potential** is ( D(X)=\sum_k w_k D_k(X)), subject to Γ. This is what both backends (CVXPY and cloth descent) must implement identically.

---

## Interfaces Γ (lawful gluing, once)

Where local laws meet, add **linear equalities**:

* Tiling overlaps: equality of repeated cells at period p/q.
* Mirrors/concats: seam equalities.
* Block substitution: overlaps of reconstructed patches.
* Components: optional equalities for uniform color on a component.

Schur elimination or direct equality constraints; same Γ for both solver paths.

---

## Fail-safe / ambiguity rule

If multiple (X) minimize (D) under Γ, snap to one-hot with a **single frozen tie-break** (lexicographic color index). This matches ARC’s “one answer required” setup and keeps determinism. ([Kaggle][6])

---

## Notes on cloth descent vs. convex one-shot

* **Cloth (paid = natural gradient).** Treat (X) as a point on the product of simplexes with an information-geometry metric (Geomstats). Use Pymanopt to minimise (D) with exact Γ or large penalties; provide value/grad (and optional hvp). ([Geomstats][7])
* **Convex one-shot.** Build the same (D,\Gamma) as a DCP program in CVXPY; solve once and project to one-hot. Use the atoms/rules tables to keep convexity. ([CVXPY][2])

On well-posed tasks, both paths must agree (modulo tie-break).

---

## References (canonical)

* **ARC format & intent:** official repo and ARC Prize guide. ([GitHub][1])
* **Motivation / priors:** Chollet, *On the Measure of Intelligence*. ([arXiv][4])
* **DCP rules & atoms:** CVXPY docs. ([CVXPY][2])
* **Manifold optimisation & info geometry:** Pymanopt and Geomstats docs. ([Pymanopt][8])

---

**Contract:** No new “operators” are permitted in v1. If an idea cannot be expressed as a convex term here (or a Γ equality, or a free isometry in Π), it is out per `00-vision-universe.md`.

[1]: https://github.com/fchollet/ARC-AGI?utm_source=chatgpt.com "fchollet/ARC-AGI: The Abstraction and Reasoning Corpus"
[2]: https://www.cvxpy.org/tutorial/dcp/index.html?utm_source=chatgpt.com "Disciplined Convex Programming -"
[3]: https://www.cvxpy.org/tutorial/functions/index.html?utm_source=chatgpt.com "Atomic Functions -"
[4]: https://arxiv.org/abs/1911.01547?utm_source=chatgpt.com "On the Measure of Intelligence"
[5]: https://arcprize.org/guide?utm_source=chatgpt.com "ARC Prize - Guide"
[6]: https://www.kaggle.com/competitions/arc-prize-2024/data?select=sample_submission.json&utm_source=chatgpt.com "ARC Prize 2024 | Kaggle"
[7]: https://geomstats.github.io/notebooks/01_foundations__manifolds.html?utm_source=chatgpt.com "1. Introduction — Geomstats latest documentation"
[8]: https://pymanopt.org/docs/stable/problem.html?utm_source=chatgpt.com "Problem — Pymanopt stable (2.2.1) documentation"
