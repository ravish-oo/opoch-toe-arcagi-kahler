# 00 — Vision & Non-Negotiables (Universe → Engine → Code)

> This document freezes the worldview and the guardrails. Every decision, spec, PR, and line of code must conform to this. If something cannot be expressed under these rules, it is out.

## 1) Bedrock (A0–A2) and the two moves

* **A0 Truth (Π):** normalize descriptions until paraphrase, label, ordering, and pose do not change content. Π is idempotent: Π(Π(x)) = Π(x). Truths are Π-fixed points.
* **A1 Exact balance (FY):** every real write is paid exactly. There exists a convex potential Φ with Fenchel–Young tightness: Φ(b) + Φ*(φ) = ⟨b, φ⟩, with φ ∈ ∂Φ(b).
* **A2 Lawful composition (GLUE):** parts meet on an interface Γ; the whole equals the infimum over matching boundary states. Quadratic face = Schur complement. Field face = Stokes/Green.

**Two moves only:**

* **Free (0-bit):** isometries that mint no differences and tick no time.
* **Paid (1-bit):** exact writes as steepest descent on a convex potential; they cost energy/time per the ledger.

**Ledger equalities (units pinned):**

* Energy of erasure: (E_{\min} = k_B T \ln 2 \cdot \Delta N).
* Bit-power: (P(t) = -,k_B T,\dot D(t)).
* Present time increment: (d\tau = \kappa , (-dD)/\ln 2).

## 2) The Kähler–Hessian cloth (the only permitted mechanics)

All computation happens on one cloth that carries both moves:

* **Free sector (Kähler):** metric (g), symplectic form (\omega), complex structure (J) with (\nabla J = 0). Free motion is Hamiltonian isometry:
  [
  \dot z_{\text{free}} = J \nabla E(z) \quad\text{and}\quad \mathcal{L}*{\dot z*{\text{free}}} g = \mathcal{L}*{\dot z*{\text{free}}} \omega = 0.
  ]
* **Paid sector (Hessian):** convex potential (D) with metric (g = \nabla^2 D) in the paid chart; paid motion is natural gradient:
  [
  \dot z_{\text{paid}} = -,g^{-1}\nabla D(z).
  ]
* **Orthogonality:** free never spends bits; paid never sneaks a free slide:
  [
  g\big(J\nabla E,,g^{-1}\nabla D\big) = 0 \quad\Leftrightarrow\quad \mathcal{L}_{J\nabla E} D = 0.
  ]

**Compose only by GLUE on Γ.** Discrete = Schur. Fields = Stokes/Green. Nothing else.

## 3) The “negation first” policy (how the engine decides)

* **Quotient first (Π):** remove all free redundancies upfront.
* **Forbid by Γ:** write only the minimal interface equalities; everything else is implicitly illegal.
* **Pay once:** if anything remains, do exactly one paid descent in the cloth’s metric to FY-tightness. Stop when the duality gap hits zero within tolerance.

No search. No branching by “case type.” No operator zoo.

## 4) What counts as a valid solution

Given a problem instance defined by ((Z, G_{\text{free}}, D, \Gamma, E)):

* **State space (Z):** truths (Π-classes) with concrete charts only for computation.
* **Free symmetries (G_{\text{free}}):** group actions that are isometries of (g) and preserve (\omega). These are the only free moves.
* **Potential (D):** a single convex functional encoding all goals/constraints. If it cannot be expressed in (D), it is not part of the problem.
* **Interfaces (\Gamma):** the only composition rules. Must compile to Schur (discrete) or Stokes/Green (fields).
* **Free energies (E) (optional):** generate isometries; must satisfy (\mathcal{L}_{J\nabla E} D = 0).

**Engine equation:**
[
\dot z = J\nabla E(z);-;g^{-1}\nabla D(z), \quad \text{with composition only via }\Gamma.
]

## 5) What is banned (fluff rule)

* Any “operator,” “rule,” or “module” that cannot be reduced to:

  * an **isometry** (free), or
  * a **convex term in (D)** plus **GLUE on (\Gamma)**

…is fluff and must not exist in this codebase.

No per-task branching, no heuristics, no sampling, no learned black-box that cannot emit FY/GLUE receipts.

## 6) Receipts are first-class (determinism by construction)

Every pipeline produces these proof objects by default:

* **Π:** canonicalize twice = once; truth class hash stable.
* **FREE:** proposed symmetries preserve (g,\omega) and leave (D) invariant.
* **FY:** duality gap (\approx 0); paid step equals (-g^{-1}\nabla D).
* **GLUE:** interface elimination equals direct composition (Schur/Stokes check).
* **Orthogonality:** (g(J\nabla E, g^{-1}\nabla D) = 0) within tolerance.
* **Ledger:** (P + k_B T \dot D = 0) numerically in paid segments.
* **Tie-break:** one frozen rule for discretization; no hidden randomness.

If any receipt fails, the change mints a difference or leaves a remainder. Reject.

## 7) Modeling completeness (when to stop observing)

You have **defined** the problem when all hold:

* **Π-sufficiency:** sufficient statistics stop changing under further normalization.
* **Sheaf glue:** local views glue to a unique global section (Čech obstruction (=0)).
* **Identifiability on (T):** (D) has a unique FY-tight minimizer modulo free orbits.
* **Full rank on (T):** (g) is positive-definite in non-gauge directions.
* **No profitable view:** (\max_c \Delta D/\text{cost}) below threshold.

Until then, the tomographic hand selects the next view by (\Delta D/\text{cost}).

## 8) Repository-level rules (how this shows up in code)

* **Core exposes exactly three mechanics:** `free_step(J∇E)`, `paid_step(-g^{-1}∇D)`, `glue(Γ)`. Nothing else.
* **Problem instances supply only:** (Z, G_{\text{free}}, D, \Gamma, E). No extra toggles.
* **Deterministic I/O:** pure functions, explicit seeds only for tests, single tie-break.
* **Tests = receipts:** every module ships with its receipts test. Property-based tests enforce invariance under (G_{\text{free}}).
* **ADRs for any deviation:** any necessary deviation is recorded as a short ADR and must point back to one of A0–A2 or to cloth regularity.

## 9) What “universe-aligned” means in practice

* We **observe all at once**: express constraints as Γ and the objective as one (D); minimize once.
* We **prefer free moves**: apply isometries first; they cost 0.
* We **pay minimally**: when forced, descend in the information metric (natural gradient) until FY-tightness.
* We **compose lawfully**: only through Γ; no ad-hoc merging.

## 10) Glossary (operational, not philosophical)

* **Truth / Π-class:** canonical representation after removing free redundancies.
* **Isometry (free):** metric/symplectic-preserving map that leaves (D) invariant.
* **Natural gradient (paid):** steepest descent under (g=\nabla^2 D).
* **Interface Γ:** boundary variables/constraints used for Schur/Stokes elimination.
* **Receipt:** a machine-checkable witness that A0/A1/A2/orthogonality/ledger hold.

---

This doc is the source of truth. Next docs (problem-specific mapping, implementation contracts, invariants catalog, receipts checklist) must reference back to this and must not widen its scope. Any PR that introduces a behavior not reducible to isometry or convex gluing is rejected on sight.
