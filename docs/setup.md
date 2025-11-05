# What ARC is (so our modeling lines up)

* Tasks are tiny colored grids in **JSON**; each file has `"train"` I/O pairs and one `"test"` pair with only input. The job is to infer a rule from a few examples and apply it once. ([GitHub][1])
* The benchmark stresses **few-shot, compositional reasoning**, not a catalog of hand-coded ops. ([ARC Prize][2])

---

# Libraries we’ll glue (and exactly what each does)

**Paid (Hessian / natural-gradient)**

* **Geomstats**: manifolds + information geometry; we use it for simplex products and access to Riemannian metrics (Fisher/Bregman flavor). ([GitHub][3])
* **Pymanopt**: optimisation *on* a manifold with autodiff backends; we use it to minimise our potential (D) on the product of simplexes (one simplex per pixel’s color distribution). ([pymanopt.org][4])

**Free (isometries / Hamiltonian)**

* **Diffrax** (JAX): includes **symplectic ODE solvers** (Hamiltonian “free” flows); we mainly use it for completeness and to keep the cloth honest, though ARC’s free moves are typically discrete D4 transforms we apply directly. ([docs.kidger.site][5])

**One-shot convex modeling (parallel baseline)**

* **CVXPY**: we express the same (D) and seams (\Gamma) as a convex program and solve once; great for rapid checks and small grids. ([cvxpy.org][6])

*(Why both?)* “Universe-aligned” means natural-gradient on the cloth + exact isometries + lawful gluing. The CVXPY path is the mathematically equivalent convex solve; it’s handy in v1 to prove we didn’t mess up geometry.

---

# Modeling ARC on the cloth (what our code implements)

* **Z**: the test grid as a **product of simplexes** ( \Delta^{C-1} ) at each cell; we keep it continuous during solve, then snap to one-hot.
* **G_free**: D4 rotations/flips (and allowed shifts) act as exact isometries; we treat pose selection as **free** (no cost).
* **Γ**: seams where local laws must agree (row/patch/component overlaps) are linear equalities.
* **D** (one potential): convex terms inferred from the train pairs: per-color histogram/area, tiling periodicity, mirror/concat constraints, block substitution consistency, component mapping, etc. We minimise (D) under Γ, once.

This matches ARC’s intent (few examples, compositional generalisation) and its JSON format. ([ARC Prize][7])

---

# Repo skeleton (drop-in)

```
arc_cloth/
  README.md
  pyproject.toml
  arc_cloth/
    __init__.py
    io/
      __init__.py
      arc_loader.py          # read JSON tasks, validate shapes/palette
      canonicalize.py        # pose/palette/origin canonicalizer (D4 etc.)
    model/
      __init__.py
      state.py               # Z: product-simplex representation for grids
      invariants.py          # infer counts, periods, mirrors, block-codes from trains
      potential.py           # build D: compose convex terms from invariants
      interfaces.py          # build Γ: linear equalities for seams/overlaps
      free_moves.py          # D4 isometries; optional Hamiltonian generators
    solvers/
      __init__.py
      cloth_ng.py            # paid: natural-gradient on product-simplex (Geomstats+Pymanopt)
      convex_one_shot.py     # CVXPY formulation+solve of same D,Γ
      snap.py                # project probs -> one-hot, tie-breaker
    receipts/
      __init__.py
      checks.py              # Π idempotence, isometry checks, reproduce-train, Γ consistency
    runner/
      run_task.py            # CLI entry; picks universe or convex path; saves PNG/JSON
  tests/
    test_io.py
    test_invariants.py
    test_potential.py
    test_end_to_end.py
  examples/
    notebook_arc_walkthrough.ipynb
```

---

# Runner skeleton (short, readable, “Claude-fillable”)

```python
# arc_cloth/runner/run_task.py
from arc_cloth.io.arc_loader import load_task
from arc_cloth.io.canonicalize import canonicalize_task
from arc_cloth.model.invariants import infer_invariants
from arc_cloth.model.potential import build_potential
from arc_cloth.model.interfaces import build_interfaces
from arc_cloth.solvers.cloth_ng import solve_on_cloth
from arc_cloth.solvers.convex_one_shot import solve_convex
from arc_cloth.solvers.snap import snap_to_one_hot
from arc_cloth.receipts.checks import check_isometries, check_train_reproduce

def solve_task(task_json_path, mode="cloth", tie_break="lex"):
    task = load_task(task_json_path)            # trains[], test_in
    task_c = canonicalize_task(task)            # Π: D4/pose/palette/origin
    inv = infer_invariants(task_c["train"])     # counts, periods, mirrors, block-codes...
    D = build_potential(inv)                    # compose convex terms into one potential
    Gamma = build_interfaces(task_c["train"])   # seams/overlaps constraints

    check_isometries(task_c)                    # FREE: D4 isometries preserve invariants

    if mode == "cloth":
        X_prob = solve_on_cloth(task_c["test_in"], D, Gamma)  # natural-gradient descent on Z
    elif mode == "convex":
        X_prob = solve_convex(task_c["test_in"], D, Gamma)    # CVXPY one-shot program
    else:
        raise ValueError("mode must be 'cloth' or 'convex'.")

    X = snap_to_one_hot(X_prob, tie_break=tie_break)          # discretize to colors
    check_train_reproduce(task_c["train"], D, Gamma)          # FY-tightness on trains (sanity)
    return X
```

---

# The “meat” Claude will fill (exact contracts)

## `io/arc_loader.py`

* **Input**: path to ARC JSON; **Output**: dict with `train` = list of `{input, output}`, `test` = list with one `{input}`. Validate rectangular grids, palette (\subseteq{0..9}). ARC format reference. ([GitHub][1])

## `io/canonicalize.py`  *(Π closure)*

* D4 pose selection (e.g., lexicographically minimal bytes), origin shift to top-left, palette remap to canonical order. (Free isometries; no cost.) ARC guide: native tasks are JSON grids. ([ARC Prize][7])

## `model/invariants.py`

* From the train pairs, compute **candidates**:

  * per-color counts/histograms between input→output,
  * horizontal/vertical **periods** via autocorrelation,
  * **mirror/concat** flags via equality tests,
  * **block substitution** candidates via small patch dictionaries,
  * **component mapping** via connected components alignment.
* Return a structured `Invariants` object (booleans, integers, small matrices).

*(ARC is intentionally built so a few examples surface these simple invariants.)* ([ARC Prize][2])

## `model/potential.py`  *(build one convex (D))*

* Compose weighted terms (each convex):

  * color mismatch: (\ell_1) or (\ell_2) between expected and actual per-color counts,
  * tiling violations: penalties for deviating from period (p),
  * mirror/concat equivalence: linear equalities or squared penalties,
  * block substitution consistency: linear maps over bit-planes,
  * optional TV-like smoothness for lines/borders when present.
* Expose **two backends**:

  * **Manifold** backend → returns callable `value/grad/hessian` for Pymanopt/Geomstats. ([pymanopt.org][4])
  * **CVXPY** backend → returns an expression tree using CVXPY atoms. ([cvxpy.org][6])

## `model/interfaces.py`  *(Γ)*

* Build linear equalities for overlaps (row/patch/component seams) so local laws glue uniquely (sheaf-style consistency).

## `solvers/cloth_ng.py`  *(paid = natural-gradient on Z)*

* Construct a **product manifold** ( \mathcal{M} = \prod_{i=1}^{H\cdot W} \Delta^{C-1} ) (simplex per pixel).
* Use **Geomstats** to represent the manifold & metric; pass to **Pymanopt** to minimise (D) under Γ (as penalties or projected steps). ([GitHub][3])
* Stop when (\Delta D) < tol; return probability tensor (X_{\text{prob}}).

## `solvers/convex_one_shot.py`  *(parallel baseline)*

* Build CVXPY variable (X\in[0,1]^{H\times W\times C}) with (\sum_c X_{ijc}=1).
* Add the same (D) terms and Γ constraints as convex expressions; call a cone/QP solver; return (X_{\text{prob}}). ([cvxpy.org][6])

## `solvers/snap.py`

* Snap each simplex to one-hot using a **single frozen tie-break** (lexicographic). This handles rare ambiguities deterministically.

## `receipts/checks.py`  *(lightweight, no extra work)*

* Π idempotence (canonicalize twice = once),
* FREE: D4 isometries preserve (D) terms,
* FY sanity: the same (D)+Γ reproduces each train output.

---

# Why this keeps effort low (and success high)

* You are **not** writing a zoo of operators, only:

  * an invariant extractor (~300–500 LOC),
  * a potential composer (~300–600 LOC),
  * two thin solver wrappers (~300–500 LOC combined),
  * simple I/O + canonicalizer (~300 LOC).

Call it ~**1.2k–2.0k LOC** for v1, which is realistic for Claude Code to crank through.

* It is faithful to ARC’s spec and ethos: JSON grids; few examples; one global solve. ([GitHub][1])
* It is faithful to the cloth: **paid** = natural-gradient on a manifold; **free** = exact isometries (D4/pose) or optional symplectic flows; **gluing** = linear seams. (Diffrax has symplectic solvers if you later want continuous free flows, but for ARC, D4 is direct.) ([docs.kidger.site][5])

---

## Quick start checklist (for Claude Code)

1. Wire `arc_loader.py` to parse one task JSON per the ARC repo format. ([GitHub][1])
2. Implement `canonicalize_task` with D4 pose search (lexicographic min).
3. Fill `infer_invariants` with: color counts, periods, mirror/concat checks, and a tiny patch dictionary learner.
4. Implement `build_potential` in **both** backends:

   * Pymanopt callable (value/grad/hessian),
   * CVXPY expression (atoms only). ([cvxpy.org][8])
5. Implement `build_interfaces` as linear equalities on overlaps.
6. Add `cloth_ng.solve_on_cloth` (Geomstats manifolds → Pymanopt minimise). ([pymanopt.org][4])
7. Add `convex_one_shot.solve_convex` (CVXPY; then project to one-hot). ([cvxpy.org][6])
8. Tie-break snap; write `run_task.py` CLI and a small notebook under `examples/`.

That’s it. You get a “universe-aligned” solver and a convex mirror-check in one repo, with minimal moving parts.

## References 
[1]: https://github.com/fchollet/ARC-AGI?utm_source=chatgpt.com "fchollet/ARC-AGI: The Abstraction and Reasoning Corpus"
[2]: https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025?utm_source=chatgpt.com "Announcing ARC-AGI-2 and ARC Prize 2025"
[3]: https://github.com/geomstats/geomstats?utm_source=chatgpt.com "geomstats/geomstats: Computations and statistics on ..."
[4]: https://pymanopt.org/?utm_source=chatgpt.com "Pymanopt"
[5]: https://docs.kidger.site/diffrax/?utm_source=chatgpt.com "Diffrax"
[6]: https://www.cvxpy.org/?utm_source=chatgpt.com "cvxpy"
[7]: https://arcprize.org/guide?utm_source=chatgpt.com "ARC Prize - Guide"
[8]: https://www.cvxpy.org/version/1.2/api_reference/cvxpy.atoms.html?utm_source=chatgpt.com "Atoms — CVXPY 1.2 documentation"
