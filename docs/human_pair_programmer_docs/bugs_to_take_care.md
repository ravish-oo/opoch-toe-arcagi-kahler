# Lattice bug
##After runnning check for lattice found this
---
so got this for lattice part:
 Lattice Wiring Verification Complete

  I've verified lattice wiring across all 1000 tasks. Here are the findings:

  ✅ LATTICE IS PROPERLY WIRED

  On all 773 SUCCESS tasks:
  - T3_lattice appears in emitters_evaluated
  - Lattice is being called and generating receipts
  - Emitters frozen order maintained: T1_witness → T2_unity → T3_lattice

  ❌ ONE CRITICAL BUG FOUND: S_lat mask construction

  6 tasks have periods detected but zero pruning:

  Task ID  | Period | residue_scope_bits | T3_prunes | Status
  ---------|--------|-------------------|-----------|--------
  310f3251 | (4, 4) | 23                | 0         | ❌ BUG
  48131b3c | (4, 4) | 12                | 0         | ❌ BUG
  c92b942c | (3, 3) | 81                | 0         | ❌ BUG
  caa06a1f | (6, 6) | 324               | 0         | ❌ BUG
  f5b8619d | (4, 4) | 56                | 0         | ❌ BUG
  f823c43c | (3, 3) | 144               | 0         | ❌ BUG

  Diagnosis:
  - Lattice correctly detected global periods
  - Lattice emitted admits for agreeing residue classes
  - BUT: S_lat masks are outside the consensus region (intersection of S_out_list)
  - Result: Admits never intersect with actual domains during LFP → zero pruning

  Root cause:
  Per WO-09 spec, S_lat should only emit admits over pixels in ∩ S_out_list (where ALL trainings have pixels). Current implementation likely uses union or
  ignores S_out masks.

  Fix location: src/arcbit/emitters/lattice.py in _emit_residue_admits()

  Fix: Only set S_lat bits for pixels (r,c) where:
  1. r % p_r == i AND c % p_c == j (residue class membership)
  2. ALL trainings have pixel at (r,c) (consensus region)

  ⚠️ MINOR: Missing emitters_evaluated on UNSAT/ERROR (not a bug)

  - 227 tasks (88 UNSAT + 139 ERROR) have no emitters_evaluated field
  - Lattice receipt EXISTS on these tasks (lattice is running)
  - Expected: emitters_evaluated only populated during LFP pass
  - UNSAT/ERROR exit before LFP → field never set

  This is expected behavior, not a wiring bug.

  📊 Lattice Activity Summary

  767 tasks (76.7%): Wired correctly, ran, found no global periods ✅6 tasks (0.6%): Wired correctly, found periods, mask bug prevents pruning ❌227 tasks 
  (22.7%): UNSAT/ERROR - no diagnostic data ⚠️
---
does it mean only 6 tasks are affected by it? or more? give me 1-3 line human comprehensible info - like how many tasks are affected if this bug remains? like can this also affect other tasks like succes/not yet?
---
## gpt said this
Short and clear:

* The lattice bug (using **union** instead of **intersection** of training scopes to build `S_lat`) shows up **only** on the 6 tasks where lattice actually detected periods. Those 6 would have pruned (and likely improved accuracy) but didn’t. Fixing it will affect **those 6 tasks** directly.

* For the other ~767 “SUCCESS” tasks where lattice found **no global periods**, `S_lat` was empty by design, so this bug doesn’t touch them.

* It won’t change UNSAT/ERROR counts directly (those exit before LFP), but fixing `S_lat` may turn a few borderline NOT_YET cases (in truly periodic tasks) into higher-accuracy or even perfect matches.
===
# LFP bugs
## How to find

---

## B. Decide “NOT_YET” vs “BUG” for the 72% (receipts-only checks)

**Where:** Use the final LFP receipts and either the current `selection` or add the tiny set-logic.

For each “SUCCESS / NOT_YET” task:

1. **Singleton check**
   Add to `lfp` (done in Step 0): `lfp_singleton_pixels`, `lfp_multi_pixels`, `lfp_empty_pixels`.

* If `lfp_singleton_pixels == R_out*C_out` **and** GT shape matches **and** `Y_out ≠ GT` → **BUG** (forced but wrong). Investigate σ/transport for this task.
* If `lfp_multi_pixels > 0` → **candidate under-constraint**. Go to (2).

2. **Hard-fact intersection check (C_must)** *(one pixel sample per class is enough)*
   For a handful of multi-valued pixels (p):

* Compute ( C_{\text{must}}(p) = \bigcap_{\ell \in {T1,T2,T3}: S^\ell[p]=1} \text{Admits}^\ell(p)).

  * If (|C_{\text{must}}(p)| == 1) and yet (D^*[p]) (final domain) has multiple bits → **LFP bug** (you didn’t intersect all hard facts at that pixel).
  * If (|C_{\text{must}}(p)| \ge 2) or equals the **family union** (e.g., only T1 spoke and admitted colors {a,b}) → **true under-constraint** (no bug; no other hard fact restricts that pixel).
  * If (|C_{\text{must}}(p)| == 0) and task is “SUCCESS” → you must be deferring one hard fact; it’s still under-constraint, not a bug.

3. **Lattice contribution**
   On a periodic task (where `global_validated=true` and `residue_scope_bits>0`), check that the multi-pixels are **outside** S_lat; inside S_lat they should be **singletons** after T3. If not, it’s a lattice wiring bug (mask not intersected in LFP).

**Outcome:**

* Any task that fails (1) (all singletons but wrong) or (2) (C_must=1 but D* multi) is a **bug**.
* All others with multi-pixels are **NOT_YET**—provably under-constrained by the currently wired families.

---
## sweep runs on h1-7
  📊 Impact: 572 BUGS Found (74.0% of SUCCESS tasks)

  Out of 773 SUCCESS tasks analyzed:

  - 43 bugs (5.6%): Singleton wrong (σ/transport bugs - already known)
  - 528 bugs (68.3%): ❌ LFP intersection bug (|C_must|==1 but |D*|>1)
  - 1 bug (0.1%): Lattice wiring bug (inside S_lat but still multi)
  - 145 tasks (18.8%): NOT_YET (true under-constraint, |C_must|>=2)
  - 3 tasks (0.4%): PERFECT

