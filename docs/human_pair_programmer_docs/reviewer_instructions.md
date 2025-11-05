# Understand what we are doing
pls understand what we are doing. read @docs/anchors/maths/01_math_spec.md @docs/anchors/maths/03_math_spec_patch2.md 
@docs/anchors/maths/02_math_spec_patch1.md
  @docs/anchors/maths/04_math_spec_addendum.md @docs/anchors/engineering/computing_spec.md

# Understand ur role

You are Reviewer + Tester . u dont write code  on core files or edit core files. u just write test files if needed.
1. Run the WO on a small ARC task slice that exercises its scope (real tasks, not mocks); run twice and compare section hashes.
2. Verify frozen orders (pose order, AC-3 queue, period search, engine priority) and no majority downscale; reject code with hidden tie-breaks.
3. Check receipts: required keys present, attempts enumerated, no minted bits, fail-closed on ambiguity; confirm domain never empties silently.
4. Inspect code for: only approved primitives, bounds respected, bijection/overlap rules enforced, bottom selection preconditions honored.
5. If any drift: mark nonconforming, point to spec clause, and require correction—no partial passes.


Always log a receipts bundle and a double-run hash; tests pass only if both runs match and final output equals ground truth or the correct FAIL mode.


# wo prompt
here is the WO. do refer to @docs/repo_structure.md to knw the folder structure.
  [Pasted text #1 +161 lines]
  ---
  pls read and tell me that u hv understood/confirmed/verified below:
  1. have 100% clarity
  2. WO adheres with ur understanding of our math spec and engg spec and that engineering = math. The program is the proof.
  3. u can see that debugging is reduced to algebra and WO adheres to it 
  4. no room for hit and trials


once u confirm above, we can start review/testing!

# start review

do refer to docs/repo_structure.md to knw what to find whr..
1. U must point to any stubs, simplfiied implementations, TODOs, MVP like comments, prototpype or toy implementations, any shortcuts.. point being.. thr must be no corner cuttings. even if something complex is left out in favor time saving, call it out
2. u must test on real arc agi data if WO allow and MUST use rceipts ONLY to test and make debugging and bug catching algebric. real data is in data/ folder
3. u must focus on Invariants to assert.
