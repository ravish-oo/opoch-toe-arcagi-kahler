# 04 — Receipts Checklist (what we *prove* per run)

> These are the machine-checkable witnesses that our code is faithful to the universe rules in `00-vision-universe.md` and the ARC mapping in `01-arc-on-the-cloth.md`. Each item states: **what to check**, **how to check**, and **where it’s grounded** (docs/specs).

---

## A) Π — Truth closure (idempotence & data shape)

**What to check**

* Canonicalization is idempotent: `canonicalize(canonicalize(task)) == canonicalize(task)`.
* ARC task format is respected: `{"train":[{"input","output"}…],"test":[{"input"}]}`, grid entries in `0..9`.
  Ground truth format comes from the ARC repo/spec. ([GitHub][1])

**How**

* Unit test: deep-equal JSON after double canonicalization.
* Validate grids are rectangular, bounded, 1–30 per side (per spec). ([GitHub][1])

---

## B) FREE — Isometries mint no differences

**What to check**

* D4 pose transforms (rotate/flip) preserve:

  * inferred invariants (periods, mirrors, counts, etc.),
  * the value/structure of D terms on train pairs.

**How**

* Property test: for each train pair, apply all eight D4 ops to inputs/outputs before invariant extraction; re-extract invariants; assert equality of invariants and equality of each active D-term’s value.
* This enforces that free moves are true isometries (pose is free). ([GitHub][1])

**Tooling hint**

* Use property-based tests (Hypothesis) to generate random small grids and verify invariance holds under D4 actions. ([Hypothesis][2])

---

## C) Γ — Lawful gluing (Schur/Stokes on seams)

**What to check**

* Overlap consistency constraints (seams) are satisfied by the returned `X`: `A @ vec(X) == b`.
* If blockwise elimination is used internally, eliminating interface variables via Schur complement yields the same reduced objective as solving with the full system (sanity on small cases).

**How**

* Assert all linear equalities hold to tolerance after solve.
* For a toy block system, compare “solve then eliminate” vs “eliminate then solve” to the same residual—this is the discrete Schur face of GLUE. (Continuous face is Stokes; see references.) ([Virtual Math][3])

---

## D) FY — Exact paid move (convexity + tightness)

**What to check**

1. **Convexity:** the CVXPY program for (D(X)) with Γ is DCP-valid.
   CVXPY enforces Disciplined Convex Programming (DCP) rules and will reject non-convex compositions. ([CVXPY][4])

2. **Duality gap ~ 0:** on train pairs, the convex solve attains negligible primal-dual gap (convex problems have zero gap under standard conditions). (Use solver status & gap fields.)

3. **Train reproduction:** evaluate (D) at the ground-truth output of each train pair; it should be ≈ 0 (or as low as the chosen penalties allow).

**How**

* Build the same (D,\Gamma) in CVXPY; call a DCP-compatible solver; assert `prob.status` is optimal and `prob.value` ≈ 0 on trains; check `prob.solution` satisfies Γ.
* Log the solver’s duality gap; expect ≈ 0 for well-posed convex programs. ([CVXPY][4])

---

## E) Paid = Natural gradient on the cloth (manifold side)

**What to check**

* On the product-simplex manifold, the descent uses the *manifold* cost/grad (not Euclidean), and basic Riemannian consistency checks pass.

**How**

* Define a Pymanopt `Problem(manifold, cost)`; let Pymanopt autogenerate gradient/Hessian via decorators or provide them explicitly. Pymanopt’s `tools.check_gradient` / Hessian tests verify second-order consistency on the manifold. ([Pymanopt][5])
* Ensure the manifold is a product of probability simplexes; the metric choice (e.g., Fisher-Rao flavor) is supplied by Geomstats (information geometry notes). ([Geomstats][6])

---

## F) Orthogonality — Free vs paid have zero cross-term

**What to check**

* If any continuous free energy (E) is used (optional demo), verify ( \mathcal{L}_{J\nabla E} D = 0 ) numerically and hence
  ( g(J\nabla E,, g^{-1}\nabla D) = 0 ) along the trajectory (up to tolerance).

**How**

* Finite-difference the directional derivative of (D) along an implemented free flow and assert ≈ 0.
* (ARC v1 mainly uses discrete D4 isometries, so this is a demo/guard, not a blocker.)

---

## G) Agreement — Cloth descent ≡ Convex one-shot

**What to check**

* For well-posed tasks, the cloth path (Pymanopt on the product simplexes) and the CVXPY path return the same snapped grid (up to the single tie-break rule).

**How**

* Solve both, snap to one-hot with the frozen rule; assert equality. Differences indicate a modeling bug, not a “preference.”

---

## H) Ledger sanity (optional diagnostic)

**What to check**

* Over iterations of the cloth descent, the *bit-power identity* holds numerically: ( P(t) \approx -k_B T,\dot D(t) ). This is a physics sanity, not required for ARC output correctness. (Landauer’s bound for erasure: (E_{\min}=k_BT\ln 2) per bit.) ([PMC][7])

**How**

* Pick units (arbitrary (k_B T)=1 for normalization); estimate (\dot D) from successive iterates; compare against a proxy “power” if you include any explicit dissipative step size model.

---

## I) Determinism — No minted randomness

**What to check**

* The whole pipeline is deterministic:

  * canonicalization is deterministic,
  * optimizer seeds fixed (if stochastic methods ever used; our default is deterministic),
  * snapping tie-break is a single frozen rule (lexicographic over color index).

**How**

* Run the same task twice; hash the output grid; hashes must match.

---

## J) Property-based invariance tests (cheap, high yield)

**What to check**

* For randomly generated small grids, properties hold universally:

  * Π idempotence,
  * FREE invariance of invariant extractor under D4,
  * Γ equalities maintained by solver outputs on synthetic mini tasks,
  * CVXPY DCP build never violates convexity.

**How**

* Use Hypothesis strategies to generate small integer grids and D4 actions; assert properties in quick tests. ([Hypothesis][2])

---

## K) Minimal receipt list per PR / run

1. **Π:** idempotence ✅
2. **FREE:** D4 invariance of invariants & D-terms ✅
3. **Γ:** all linear seam equalities satisfied ✅
4. **FY (convex):** DCP valid; duality gap ~ 0 on trains ✅ ([CVXPY][4])
5. **Paid (manifold):** Pymanopt gradient/Hessian checks pass ✅ ([Pymanopt][8])
6. **Orthogonality:** ( \mathcal{L}_{J\nabla E} D \approx 0 ) if any continuous free (E) used ✅
7. **Agreement:** cloth vs convex outputs match after snap ✅
8. **Determinism:** identical outputs on repeated runs ✅

If any fails, the change **mints a difference** or leaves a **remainder**—reject per `00-vision-universe.md`.

---

## References (the docs these receipts lean on)

* **ARC JSON format** (train/test, grids of ints). ([GitHub][1])
* **CVXPY DCP rules & atoms** (convexity checks; duality gap/status). ([CVXPY][4])
* **Pymanopt problem & tools** (auto grad/Hess; consistency checks). ([Pymanopt][5])
* **Geomstats information geometry** (Fisher-Rao metric context). ([Geomstats][6])
* **Hypothesis** (property-based testing). ([Hypothesis][2])
* **Landauer principle** (ledger sanity). ([PMC][7])
* **Stokes/Schur intuition** (gluing on interfaces). ([Virtual Math][3])

---

*End of checklist.*

[1]: https://github.com/fchollet/ARC-AGI?utm_source=chatgpt.com "fchollet/ARC-AGI: The Abstraction and Reasoning Corpus"
[2]: https://hypothesis.readthedocs.io/?utm_source=chatgpt.com "Hypothesis 6.142.5 documentation"
[3]: https://virtualmath1.stanford.edu/~conrad/diffgeomPage/handouts/stokesthm.pdf?utm_source=chatgpt.com "Math 396. Stokes' Theorem on Riemannian manifolds"
[4]: https://www.cvxpy.org/version/1.4/tutorial/dcp/index.html?utm_source=chatgpt.com "Disciplined Convex Programming"
[5]: https://pymanopt.org/docs/stable/problem.html?utm_source=chatgpt.com "Problem — Pymanopt stable (2.2.1) documentation"
[6]: https://geomstats.github.io/notebooks/08_practical_methods__information_geometry.html?utm_source=chatgpt.com "Information geometry — Geomstats latest documentation"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7514250/?utm_source=chatgpt.com "The Landauer Principle: Re-Formulation of the Second ..."
[8]: https://pymanopt.org/docs/stable/tools.html?utm_source=chatgpt.com "Tools — Pymanopt stable (2.2.1) documentation"
