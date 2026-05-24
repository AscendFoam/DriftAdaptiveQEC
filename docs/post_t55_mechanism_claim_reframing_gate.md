# Post-T55 Mechanism Claim Reframing Gate

## 1. Status Recap: T36 Through T55

### 1.1 Task chain

| Task | Type | Key mechanism contribution |
| --- | --- | --- |
| T36 | Read-only single-seed diagnosis | Narrowed seed=20260429 to residual-amplitude / teacher-delta regime instability hypothesis; ruled out response lag, correction saturation, and dead teacher branch |
| T38 | Single-seed trace-export probe | 4798 trace rows; upgraded diagnosis to trace-supported combined committed-`b` instability; identified large `teacher_b` + large `delta_b` as the clearest trace-level explanation |
| T46 | Docs-only plan gate | Defined 6-seed locked pack, phased approach (Phase A: trace-only, Phase B: intervention), and intervention matrix (I1–I6) |
| T54 | Multi-seed trace-only probe | 57,586 trace rows across 6 seeds; confirmed committed-`b` instability is broadly repeated with qualifications; introduced three seed categories (quiet, classic, universal) |
| T55 | Multi-seed I1 intervention probe | 48 HIL sessions; I1 lower-clip (0.12 → 0.06) is mixed: harms 4/6, helps 2/6; "high committed-`b` is harmful" not supported as general explanation |

### 1.2 Evidence trajectory in one paragraph

T36 established that seed=20260429 is a borderline case where Gv5 does not clearly outperform Full, and proposed combined committed-`b` instability as the leading hypothesis. T38 confirmed this at trace level for the same seed. T46 planned a phased multi-seed approach to test generalization and intervention. T54 showed the instability pattern is broadly present (5/6 seeds) but mostly helps Gv5 rather than harms it. T55 directly tested the most natural intervention — lowering the residual clip to reduce committed-`b` amplitude — and found it harms 4/6 seeds. The simple mechanism narrative "high committed-`b` is the problem, reducing it will help" is not supported by the intervention evidence.

## 2. Claim Table

| # | Claim or hypothesis | Strongest supporting task(s) | Strongest contradicting task(s) | Current status | Exact wording boundary |
| --- | --- | --- | --- | --- | --- |
| M1 | seed=20260429 shows combined committed-`b` instability in Gated v5 | T36 (summary diagnosis), T38 (4798 trace rows) | None — no task contradicts this specific single-seed observation | **retain** | "Trace-supported single-seed diagnostic evidence for combined committed-`b` instability on seed=20260429"; do not extend to causal proof or multi-seed generalization of harm |
| M2 | Committed-`b` instability is the primary cause of Gv5 degradation | T36, T38 (for seed=20260429 specifically) | T54 (instability mostly helps Gv5 across seeds), T55 (lowering committed-`b` harms 4/6 seeds) | **weaken** | Safe wording: "committed-`b` instability correlates with Gv5's advantage in most seeds, and with marginal degradation in seed=20260429 specifically"; do not write "primary cause of degradation" without seed-qualification and without acknowledging that the same instability mostly helps |
| M3 | The instability pattern generalizes beyond seed=20260429 | T54 (5/6 seeds show instability) | None — no task contradicts generalization | **retain** | "Broadly repeated with qualifications: 5/6 seeds show committed-`b` instability in Gv5, with three categories (quiet/classic/universal)"; do not write "uniformly confirmed" or "causal mechanism proven across seeds" |
| M4 | High committed-`b` amplitude is harmful and should be reduced | T36 (proposed as hypothesis), T38 (observed large committed-`b` on the borderline seed) | T54 (high committed-`b` correlates with Gv5 winning in 4/5 classic seeds), T55 (lowering clip → harms 4/6 seeds) | **retire** | Do not use "high committed-`b` is harmful" or "committed-`b` should be reduced" as a general claim; the I1 intervention directly contradicts this as a general explanation. Where the hypothesis is referenced historically, label it as "pre-T55 working hypothesis, now contradicted by I1 intervention evidence" |
| M5 | Reducing residual clip from 0.12 to 0.06 will stably improve Gv5 outcomes | T46 (planned as I1) | T55 (I1 harms 4/6, helps 2/6) | **retire** | I1 is empirically refuted as a general improvement; do not propose clip reduction as a solution; where I1 is referenced, state "I1 tested, mixed result (harms 4/6, helps 2/6)" |
| M6 | The committed-`b` instability is exclusive to Gated v5 | T36, T38 (Full was stable for seed=20260429), T54 (4 classic seeds show Full stable, Gv5 unstable) | T54 (seed=20260510 shows both modes unstable — "universal" category) | **weaken** | "Committed-`b` instability is a systematic property of Gv5 in 4/6 seeds and is absent from Full in 4/6 seeds; in 1/6 seeds (20260510) both modes are unstable; in 1/6 seeds (20260425) neither mode is unstable"; do not write "exclusive to Gv5" without the universal-seed qualification |
| M7 | Teacher delta channels (`teacher_delta_b_q`, `teacher_delta_b_p`) drive the instability | T36 (observed regime shift in teacher-delta channels on seed=20260429) | None — no intervention has directly tested I3 (teacher-delta attenuation) | **still-open** | "T36 identified teacher-delta regime shift on seed=20260429; this remains an open hypothesis but has not been tested by intervention; I3 from T46 plan was not executed"; do not write "teacher deltas are proven as the upstream cause" or "teacher deltas are ruled out" |
| M8 | The committed-`b` instability needs to be fixed/mitigated | T36, T38 (working assumption that instability → degradation) | T54 (instability mostly helps), T55 (fixing it harms most seeds) | **reframe** | "The committed-`b` instability in Gv5 appears to be a performance mechanism (correlating with better outcomes) rather than a defect; the question should be reframed from 'how to fix the instability' to 'under what conditions the instability is harmful and whether any intervention is needed'"; do not assume instability = defect |
| M9 | The three seed categories (quiet / classic / universal) are stable regimes | T54 (introduced classification based on 6-seed evidence) | T55 (seed=20260430 is "classic" but I1 helps, violating the simple prediction that classic seeds would be harmed by clip reduction) | **weaken** | "Three descriptive categories capture the 6-seed trace evidence but do not cleanly predict intervention outcomes; seed=20260430 violates the category-intervention mapping"; do not write "seed categories predict intervention response" |
| M10 | A second intervention (I2 or I3) is justified as the next execution lane | T46 (planned I1–I3 as phased intervention matrix) | T55 (I1 result suggests parameter-sweep interventions produce mixed results) | **still-open** | "I2 (lower scale) is partially confounded with I1 and may produce similar mixed results; I3 (teacher-delta attenuation) targets a different mechanism channel and remains theoretically justified, but T55's overall picture suggests cautious expectations"; see Section 4 for gate decision |

## 3. What T55 Changed in the Mechanism Story

### 3.1 The narrative shift

Before T55, the mechanism story was:

> "Gated v5's committed-`b` instability is the mechanism behind seed=20260429's borderline performance. If we can reduce the instability (e.g., by lowering the residual clip), we should see improvement."

After T55, the mechanism story is:

> "Gated v5's committed-`b` instability is broadly present across seeds and mostly correlates with better outcomes. Lowering the clip to reduce instability removes Gv5's advantage in 4/6 seeds. The instability is more likely a performance mechanism than a defect. Seed=20260429 remains a borderline case, but the mechanism story is more complex than a simple amplitude problem."

### 3.2 Specific changes by evidence source

| Evidence source | Pre-T55 interpretation | Post-T55 interpretation |
| --- | --- | --- |
| T36 mechanism matrix: "committed combined-`b` instability" | Leading explanation for degradation | Still valid as the observed pattern, but the interpretation of "instability = degradation" is not general |
| T38 trace: high-amplitude delta-b and committed-b in Gv5 | Evidence that Gv5's residual path is unstable | Still valid as the observed pattern, but "unstable" does not mean "harmful" — it mostly helps |
| T54: instability in 5/6 seeds | Generalization of the degradation mechanism | Generalization of the instability pattern, but not of degradation — the instability helps in most seeds |
| T54: three seed categories | May guide targeted interventions | Descriptive categories that do not cleanly predict intervention outcomes (seed=20260430 violates) |
| T55: I1 harms 4/6 | (not yet tested) | Directly contradicts "reduce committed-`b` to help"; suggests the instability is a feature, not a bug |

### 3.3 What did NOT change

1. The T36/T38 observation that seed=20260429 is a borderline case where Gv5 does not clearly beat Full remains valid.
2. The T54 observation that committed-`b` instability is broadly present across the 6-seed pack remains valid.
3. The T54 three-category classification (quiet/classic/universal) remains a valid descriptive summary of the trace evidence.
4. The frozen-set formal benchmark evidence (T24/T25) is unaffected.
5. C4 remains `partial` — it was not upgraded or downgraded, but its interpretation has shifted.

## 4. Second Intervention Lane Decision

### 4.1 Assessment

The T55 I1 intervention produced mixed results. The natural next intervention candidates from T46 are:

- **I2 (lower `residual_scale_b` from 1.0 to 0.5):** Partially confounded with I1 (both target output amplitude). T55 suggests amplitude reduction is the wrong direction. I2 would likely produce similar mixed results.
- **I3 (teacher-delta attenuation):** Targets a different mechanism channel (input rather than output). Theoretically distinct from I1. But T55's overall picture suggests that the project should reconsider whether the instability needs fixing at all, before investing in another intervention execution.

### 4.2 Verdict: `deferred pending better question`

A second intervention lane is **not recommended as an immediate next execution task**. The reason is not that I2/I3 are technically infeasible, but that the question they would answer — "how to fix the committed-`b` instability" — is no longer the right question. T55 shows the instability is mostly helpful.

Before running I2 or I3, the project should first resolve:

1. Whether the mechanism question should be reframed from "how to fix instability" to "under what conditions does instability help vs. harm, and is the borderline seed=20260429 case worth optimizing at the expense of the other 4 classic seeds?"
2. Whether the paper narrative needs the instability to be "fixed" at all, or whether the current evidence (Gv5 wins 5/6 seeds, I1 harms 4/6) already supports a paper story about why Gv5 works well.

If a future task re-frames the question and still concludes that an intervention test is needed, I3 (teacher-delta attenuation) is the more informative choice because it tests a different mechanism channel than I1.

## 5. T47 Recommendation

### 5.1 Can T47 proceed?

**T47 (paper ablation result-pack and material ledger) can proceed, but only under explicit mechanism-hedge wording.**

### 5.2 Required mechanism-hedge boundary for T47

If T47 proceeds, it must:

1. **Not present the committed-`b` instability as a solved or mitigated mechanism.** The current evidence supports "broadly present diagnostic pattern" but not "causal mechanism with validated intervention."
2. **Not present I1 as a successful or failed intervention in isolation.** I1 is one data point (clip reduction from 0.12 to 0.06) that produced mixed results; it does not close or open the mechanism story.
3. **Present the Gv5 instability as a feature with qualifications.** The paper should acknowledge that committed-`b` instability is broadly present and correlates with Gv5's advantage in most seeds, with seed=20260429 as a borderline case.
4. **Retain diagnostic hedging language** consistent with C4 = `partial`:
   - Safe: "trace-supported diagnostic evidence," "observed pattern across 6 seeds," "mixed intervention results"
   - Unsafe: "mechanism proven," "root cause identified," "causal evidence," "validated intervention"
5. **Not reopen the frozen-set benchmark boundary.** T47 is a material-packaging task, not a benchmark expansion task.

### 5.3 T47 should not proceed as unconditional next work

T47 should not be treated as "the mechanism story is closed, now package results." T47 should be treated as "package what we know honestly, with appropriate hedging around the still-open mechanism questions."

## 6. Explicit Non-Claims

This gate document does not claim:

1. that the mechanism story is closed — C4 remains `partial`
2. that T55 proves the instability is always helpful — the effect is seed-dependent
3. that no future intervention could help — I3 was not tested, and a different question framing might justify it
4. that the frozen-set benchmark boundary is being reopened
5. that `.tflite` runtime, real-board validation, or training reproducibility are affected
6. that T47 should proceed unconditionally — it must carry mechanism-hedge wording
7. that the three seed categories (quiet/classic/universal) are exhaustive or final
8. that this gate upgrades any result into causal proof, mechanism closure, or paper-grade benchmark evidence
9. that a second intervention lane is permanently closed — it is deferred, not rejected
