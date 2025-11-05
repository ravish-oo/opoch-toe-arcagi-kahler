# Understand what we are doing
pls understand what we are doing. read @docs/anchors/maths/01_math_spec.md @docs/anchors/maths/03_math_spec_patch2.md 
@docs/anchors/maths/02_math_spec_patch1.md
  @docs/anchors/maths/04_math_spec_addendum.md @docs/anchors/engineering/computing_spec.md

# Understand ur role
Implement exactly the WO interface; no stubs, no TODOs, no extra helpers outside the spec.
Use only the allowed primitives and frozen orders; no randomness, no floats, no heuristics.
On any unprovable case: return silent (A=all, S=0) or the specified FAIL (UNSAT, SIZE_INCOMPATIBLE, FIXED_POINT_NOT_REACHED).
Emit receipts for every public call (hashes, counts, attempts list) and ensure double-run identical hashes.
In code keep pure functions, zero side effects except receipts.


# wo prompt
here is the WO. do refer to @docs/repo_structure.md to knw the folder structure.
  [Pasted text #1 +161 lines]
  ---
  pls read and tell me that u hv understood/confirmed/verified below:
  1. have 100% clarity
  2. WO adheres with ur understanding of our math spec and engg spec and that engineering = math. The program is the proof.
  3. u can see that debugging is reduced to algebra and WO adheres to it 
  4. no room for hit and trials

once u confirm above, we can start coding!

# how to debug
u may hv tried few things but see if u want to try something from here. also when u say u hypothesize, why do i need to to do guess and hope. i mean best part of programming is that u can print output at each step and study  it when out and find exactly when it breaks.. that's what debuggers formalized but old school way us to print the outputs and see. u r trained on code that probably didnt hv these prints for debug but that's how its done. 

so u must not "hypothesize" and fix. hypothesize to investigate, print outputs and settle hypothesis rather than hit and hope. that just wont work. so get back to 0th principle of coding. print and see and fix. simple as that.. 
hope this helps 