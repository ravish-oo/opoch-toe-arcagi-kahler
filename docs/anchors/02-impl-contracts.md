# 02 — Implementation Contracts (ARC on the Kähler Cloth)

> This is the only source the code must match. It fixes APIs, shapes, and library glue so a single dev + Claude Code can implement v1 in 1–2 days. No CI, no team overhead, no extras.

---

## 0) Ground truth about ARC (so our I/O is correct)

* An ARC task is JSON with keys:

  * `"train"`: list of `{ "input": grid, "output": grid }`
  * `"test"`: list with one `{ "input": grid }`
* A “grid” is a rectangular matrix (list of lists) of **integers 0–9**, size 1×1 to 30×30. ([GitHub][1])
* ARC Prize guide: few train pairs then a single test input; JSON grids of ints. ([ARC Prize][2])

---

## 1) Environment + libraries (pin to avoid yak-shaves)

* **Python**: 3.11+
* **Core libs**:

  * **Geomstats** ≥ latest stable docs (manifolds/info geometry) ([Geomstats][3])
  * **Pymanopt** ≥ 2.2 (manifold optimisation; cost/grad/Hess backends) ([Pymanopt][4])
  * **CVXPY** ≥ 1.4 (DCP-checked convex modeling) ([CVXPY][5])
  * **Diffrax** (JAX) for optional symplectic free-flow demo; not required for D4 isometries. ([Kidger Docs][6])
  * **NumPy**/**SciPy** for LA; **Pillow/Matplotlib** for PNGs (optional)

> Note: Diffrax/JAX is dense-LA oriented; fine for ARC’s tiny grids. For very sparse Jacobians, acknowledge dense bias. ([GitHub][7])

---

## 2) High-level data shapes (fixed across modules)

Let test grid shape be **H×W**, colors **C≤10**.

* **Probability tensor** for the *unknown* output during solve:

  * `X_prob`: `float64` array of shape **[H, W, C]**, with per-cell simplex constraints:
    `X_prob[i,j,:] >= 0`, `X_prob[i,j,:].sum() == 1`.
* **One-hot output** after snap:

  * `X_onehot`: `uint8` or `int64` array **[H, W]**, each cell in `[0..C-1]` (palette indices).
* **Flattening** convention for CVXPY / Γ:

  * `vec(X_prob)` flattens as row-major over `(i,j,c)` → vector **[H*W*C]**.

---

## 3) Module contracts (functions, signatures, semantics)

### 3.1 I/O + Canonicalization (Π)

**`arc_cloth/io/arc_loader.py`**

```python
def load_task(path: str) -> dict:
    """
    Returns: {
      "train": [{"input": List[List[int]], "output": List[List[int]]}, ...],
      "test":  [{"input": List[List[int]]}]
    }
    Validates:
      - grids rectangular
      - entries in 0..9
      - sizes within 1..30
    """
```

(Format matches ARC repo/guide.) ([GitHub][1])

**`arc_cloth/io/canonicalize.py`**

```python
def canonicalize_task(task: dict) -> dict:
    """
    Applies Π (free isometries) uniformly:
      - choose a canonical D4 pose for every grid (lexicographic min of bytes)
      - remap palette to contiguous [0..C-1] in order of first appearance
      - top-left origin normalization (no-op for ARC bounds)
    Returns new task dict with same keys.
    Invariants: applying twice == once (Π idempotence).
    """
```

* D4 (rot/flip) are exact isometries; they **must not** alter invariants/D. ([GitHub][1])

---

### 3.2 Invariant extraction → terms for **D**

**`arc_cloth/model/invariants.py`**

```python
from typing import TypedDict, List, Tuple, Dict

class Invariants(TypedDict, total=False):
    colors: List[int]                   # palette indices 0..C-1
    color_counts: List[int]             # expected per-color totals
    period_h: int | None                # horizontal period
    period_v: int | None                # vertical period
    mirror_v: bool                      # vertical mirror flag
    mirror_h: bool                      # horizontal mirror flag
    concat_axes: List[str]              # e.g., ["h","v"] bands equalities
    block_size: Tuple[int,int] | None   # patch size for substitution
    patch_codebook: Dict[bytes, bytes]  # input->output patch mapping (k×k bytes)
    components: List[List[Tuple[int,int]]]  # connected components coords
```

```python
def infer_invariants(train: list[dict]) -> Invariants:
    """
    Derive candidates from all train pairs:
      - per-color counts on outputs
      - periods via autocorr on rows/cols
      - mirror/concat via exact equality checks
      - block substitution: learn small codebook for k×k patches when detected
      - components: 4-connected components and their output colors/placements
    Deterministic; no stochastic fitting.
    """
```

*(ARC intent: infer minimal priors from few examples.)* ([ARC Prize][2])

---

### 3.3 Build **D** (paid potential) — two backends must agree

We expose **two** parallel builders over the same `Invariants`:

#### 3.3.a CVXPY backend (one-shot convex mirror)

**`arc_cloth/model/potential.py`**

```python
import cvxpy as cp

class ConvexProgram:
    X: cp.Variable        # shape (H, W, C), simplex per cell
    objective: cp.Expression
    constraints: list[cp.Constraint]

def build_potential_cvxpy(inv: Invariants, H: int, W: int, C: int) -> ConvexProgram:
    """
    Creates CVXPY variable X in [0,1]^(H×W×C) with sum_c X[i,j,c]==1,
    adds convex terms per 'inv' and linear equality seams Γ,
    returns (X, objective=sum(weights*terms), constraints).
    DCP compliance required; use only CVXPY atoms / linear ops.
    """
```

* DCP must hold (CVXPY checks convexity). ([CVXPY][5])
* Typical atoms: `sum`, `norm1`, `square`, `abs`, indicators via constraints. ([CVXPY][8])

#### 3.3.b Manifold backend (natural-gradient on product of simplexes)

**`arc_cloth/model/potential.py`**

```python
import numpy as np
from typing import Callable

class ClothPotential:
    value: Callable[[np.ndarray], float]              # (H,W,C) -> scalar
    grad:  Callable[[np.ndarray], np.ndarray]         # (H,W,C) -> (H,W,C)
    hess_vec: Callable[[np.ndarray, np.ndarray], np.ndarray] | None  # optional hvp

def build_potential_cloth(inv: Invariants, H: int, W: int, C: int) -> ClothPotential:
    """
    Returns value/grad/(optional hvp) implementing the same D as the CVXPY build.
    All terms convex; gradients Lipschitz on interior of the simplex product.
    """
```

* Will be passed to **Pymanopt** with a **product-simplex** manifold from **Geomstats** (info geometry context). Geomstats supplies manifold abstractions; Pymanopt handles Riemannian optimisation given cost/grad/HVP decorators. ([Geomstats][3])

---

### 3.4 Interfaces **Γ** (lawful gluing)

**`arc_cloth/model/interfaces.py`**

```python
class Interfaces:
    # As linear equalities A vec(X) = b
    A: np.ndarray     # shape [M, H*W*C]
    b: np.ndarray     # shape [M]

def build_interfaces(inv: Invariants, H: int, W: int, C: int) -> Interfaces:
    """
    Build linear equalities enforcing overlaps/seams:
      - tiling overlaps (periodic equality)
      - mirror seams (paired indices equal)
      - concat band equalities
      - component-wise consistency equalities (single color across component)
    """
```

* In CVXPY this becomes `constraints += [A @ vec(X) == b]`.
* In manifold descent, either:

  * enforce as hard projection after each step (projected manifold step), or
  * add as large-weight penalties in D (keep convexity).

---

### 3.5 Solvers (two code paths that must agree)

#### 3.5.a Universe-aligned cloth descent (paid = natural gradient)

**`arc_cloth/solvers/cloth_ng.py`**

```python
from geomstats.geometry.product_manifold import ProductManifold
# or custom construction: Product of (H*W) probability simplexes

def solve_on_cloth(
    test_input: list[list[int]],
    potential: ClothPotential,
    interfaces: Interfaces,
    palette_size: int,
    *,
    max_iters: int = 2000,
    tol: float = 1e-8,
    step_rule: str = "armijo"
) -> np.ndarray:
    """
    Construct manifold M = Π_{i=1..H*W} Δ^{C-1}.
    Initialize X_prob at uniform (or train-informed prior).
    Run Pymanopt optimizer (e.g., ConjugateGradient/RGD) on M to minimize 'potential'
    subject to Γ (hard projection or penalty).
    Stop at ||grad|| below tol or ΔD < tol. Return X_prob shape (H,W,C).
    """
```

* Pymanopt contract: cost/grad/hessian-vector products may be supplied or AD-decorated; backends available (autodiff decorators). ([Pymanopt][4])
* Optimizers: Nonlinear CG/steepest-descent on manifolds per docs. ([Pymanopt][9])
* Geomstats: manifold constructs; information geometry background. ([Geomstats][3])

#### 3.5.b Convex one-shot mirror (DCP program)

**`arc_cloth/solvers/convex_one_shot.py`**

```python
import cvxpy as cp

def solve_convex(
    H: int, W: int, C: int,
    prog: ConvexProgram,
    solver: str = "ECOS",
    tol: float = 1e-8
) -> np.ndarray:
    """
    Solve: minimize(prog.objective) s.t. simplex + Γ constraints.
    Returns X_prob (H,W,C) as np.ndarray in [0,1] with per-cell sums == 1.
    """
```

* Must be DCP-valid; see CVXPY DCP tutorial + atoms list. ([CVXPY][5])

---

### 3.6 Discretization (snap) + tie-break

**`arc_cloth/solvers/snap.py`**

```python
def snap_to_one_hot(X_prob: np.ndarray, tie_break: str = "lex") -> np.ndarray:
    """
    For each cell (i,j), take argmax over c.
    If tie: break deterministically by lex order on (color index).
    Returns int array [H,W] with values in 0..C-1.
    """
```

---

### 3.7 Receipts (machine-checks, not logs)

**`arc_cloth/receipts/checks.py`**

```python
def check_pi_idempotent(task: dict) -> None:
    """ assert canonicalize(canonicalize(task)) == canonicalize(task) """

def check_free_isometries(task_c: dict, inv: Invariants) -> None:
    """ assert D-terms & derived invariants invariant under D4 transforms """

def check_train_reproduce(train: list[dict], builder_cvxpy, builder_cloth) -> None:
    """
    For each train pair:
      - with builders produce D, Γ for that pair
      - evaluate D at ground-truth output; expect near zero
      - (optional) solve convex and ensure duality gap ~ 0
    """
```

* DCP and atom use per CVXPY user guide. ([CVXPY][10])

---

## 4) Exact form of v1 **D** terms (stable math + code hooks)

Each term must appear in **both** backends (CVXPY + ClothPotential).

1. **Color histogram**

   * Detect: per-color totals on train outputs.
   * CVXPY: `sum(abs(sum(X[:,:,c]) - n_hat[c]))` (or squared). DCP-valid. ([CVXPY][5])
   * Cloth: value/grad over `X_prob`; gradient is per-cell constant for that color.

2. **Horizontal/vertical periodicity**

   * Detect: period p via row/col autocorrelation.
   * CVXPY: `sum(norm1(X[i,j,:] - X[i,j+p,:]))` (index wrap or valid window).
   * Cloth: sum of L1/L2 penalties between paired simplex rows.

3. **Mirror constraints**

   * Detect: equality across seam.
   * Γ: hard linear equalities `X[i,j,:] == X[i,j*,:]` (preferred); or squared penalty.

4. **Concatenation**

   * Detect: shifted band equality.
   * Γ or penalty: tie band slices linearly.

5. **Block substitution (k×k)**

   * Detect: dictionary map from input to output patches.
   * CVXPY: mixture weights per codeword with simplex constraint, linear reconstruction; or hard codeword selection for tiny dictionaries.
   * Cloth: same as convex, continuous relaxation preferred.

6. **Component consistency**

   * Detect: 4-conn components; each maps to a consistent color or transform.
   * CVXPY/Cloth: enforce equal color distribution inside a component mask (linear averages).

> All terms must keep convexity (CVXPY DCP) and have well-defined gradients (Cloth). Use atoms and rules per docs. ([CVXPY][5])

---

## 5) Optional: Free flows beyond D4 (demo only)

ARC’s free moves are discrete D4 isometries, applied directly. For completeness, keep a tiny Hamiltonian stub using **Diffrax**:

**`arc_cloth/model/free_moves.py`**

```python
def apply_d4(grid: np.ndarray, op: str) -> np.ndarray:
    """ op in {'rot90','rot180','rot270','flip_h','flip_v',...}; pure isometry. """

# optional demo
from diffrax import KahanLi8, diffeqsolve, ODETerm

def symplectic_demo_step(y0: np.ndarray, H_func, t1: float):
    """
    NOT used in ARC v1; keeps cloth honest. Uses Diffrax symplectic solver.
    """
```

(Diffrax: JAX ODE suite incl. symplectic, reversible solvers.) ([Kidger Docs][6])

---

## 6) Runner API (one place to orchestrate)

**`arc_cloth/runner/run_task.py`**

```python
def solve_task(task_json_path: str, mode: str = "cloth", tie_break: str = "lex") -> np.ndarray:
    """
    mode in {"cloth","convex"}.
    Steps:
      load -> canonicalize -> infer_invariants -> build D & Γ -> solve -> snap
    Returns one-hot grid [H,W] with palette indices.
    """
```

* “cloth” path = Geomstats (product simplex) + Pymanopt (RGD/NCG). ([Geomstats][3])
* “convex” path = CVXPY DCP program. ([CVXPY][5])
* Both should agree on well-posed tasks.

---

## 7) Tests we **must** include (no CI, just local)

* **Receipts tests** (in `tests/`):

  * Π idempotence on a few tasks.
  * FREE invariance: invariants & D terms unchanged under D4.
  * Train reproduction: D≈0 at ground-truth outputs (and duality gap~0 in CVXPY).
  * Agreement: cloth vs convex outputs equal after snap on sample tasks.

* **Property tests** (manual or Hypothesis): D4 invariance over random small grids is preserved by canonicalizer and counts detector.

(CVXPY DCP and atoms ensure convexity programmatically. ([CVXPY][5]))

---

## 8) Non-goals (cut the weight)

* No CI, no multi-env matrix, no packaging polish beyond `pyproject.toml`.
* No operator zoo; no per-task branching; no stochastic search/tuning.
* No MILP/ILP in v1; keep pure convex + manifold descent.

---

## 9) Implementation order (so Claude doesn’t wander)

1. `load_task`, `canonicalize_task` (Π; D4 pose/palette) — test Π idempotence.
2. `infer_invariants` (counts, period, mirror/concat, minimal block map).
3. `build_potential_cvxpy` **and** `build_potential_cloth` for the same terms.
4. `build_interfaces` Γ (linear equalities).
5. `solve_convex` then `solve_on_cloth`; add `snap_to_one_hot`.
6. Receipts tests + 2–3 public tasks end-to-end.

---

## 10) References (the only ones we rely on)

* ARC format & intent: ARC repo; ARC Prize guide. ([GitHub][1])
* Geomstats (manifolds/info geometry) & paper. ([Geomstats][3])
* Pymanopt (autodiff + problem/optimizers docs; paper). ([Pymanopt][4])
* CVXPY DCP rules & atoms (convexity guarantees). ([CVXPY][5])
* Diffrax (JAX ODE; symplectic/reversible solver list). ([Kidger Docs][6])

---

### One-liner you can hand to Claude Code

> Implement exactly the functions and signatures in `02-impl-contracts.md`; use Geomstats+Pymanopt for the “cloth” path, CVXPY for the “convex” path, and D4 isometries for free moves. Keep DCP-validity, manifold descent on the product of simplexes, linear Γ equalities, and the receipts tests. Anything else is out (see 00/01 docs).

[1]: https://github.com/fchollet/ARC-AGI?utm_source=chatgpt.com "fchollet/ARC-AGI: The Abstraction and Reasoning Corpus"
[2]: https://arcprize.org/guide?utm_source=chatgpt.com "ARC Prize - Guide"
[3]: https://geomstats.github.io/geomstats.github.io/index.html?utm_source=chatgpt.com "Geomstats — Geomstats latest documentation"
[4]: https://pymanopt.org/docs/latest/autodiff.html?utm_source=chatgpt.com "Automatic Differentiation"
[5]: https://www.cvxpy.org/version/1.4/tutorial/dcp/index.html?utm_source=chatgpt.com "Disciplined Convex Programming"
[6]: https://docs.kidger.site/diffrax/?utm_source=chatgpt.com "Diffrax"
[7]: https://github.com/patrick-kidger/diffrax/issues/545?utm_source=chatgpt.com "sparse jacobian solve · Issue #545 · patrick-kidger/diffrax"
[8]: https://www.cvxpy.org/version/1.2/api_reference/cvxpy.atoms.html?utm_source=chatgpt.com "Atoms — CVXPY 1.2 documentation"
[9]: https://pymanopt.org/docs/latest/optimizers.html?utm_source=chatgpt.com "Optimization — Pymanopt latest (2.2.2.dev1+ga1f52e7) ..."
[10]: https://www.cvxpy.org/version/1.2/tutorial/index.html?utm_source=chatgpt.com "User Guide — CVXPY 1.2 documentation"
