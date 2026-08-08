# Reviewer response: post-selection and break-even boundaries

- Task: `T7.3.4`
- Verdict: `PASS_POSTSELECTION_DIAGNOSTIC_AND_BREAKEVEN_NOT_ESTABLISHED`
- Package readiness: `draft_with_placeholders`
- Gates/mutations: `24/24` / `24/24`

## Point-by-point response

We agree that post-selection and break-even are easy to overstate, and we have separated three distinct quantities. First, no Phase-6D primary metric uses post-selection: all 79,872 registered rounds and all 13 drift families remain in the denominator, and the proposed decoder retains its zero-improvement NO-GO result against static-mixture exact MLD.

Second, the historical post-selection result is an offline diagnostic, not online correction. Its threshold is fitted on 294,912 training samples and evaluated on 1,572,864 disjoint samples using an observed static-MAP confidence score. At the 90% target, conditional decision error decreases from 0.013785 to 0.001242 at 0.899108 acceptance. However, accepted failures plus a unit rejection penalty gives total cost 0.102009, compared with raw error 0.013785. All eight targets improve conditional error and all eight become worse at unit rejection cost. The diagnostic is therefore reported with acceptance, rejection and cost and is ineligible for the primary LER or break-even claim.

Third, the finite-model result is only a 300-us wall-clock operational boundary: a sustained/cumulative crossover of leakage-inclusive CPTNI average-fidelity curves against matched encoded idle. It uses neither an exponential fit nor a lifetime ratio. The low-cutoff counterexample is retained, the active short-time rate is unqualified, matched idle is not the best passive physical-qubit encoding, and twelve physical/control cost fields remain null. Consequently, paper-defined simulation-derived coherence gain, full-cost break-even and experimental break-even are all NOT_ESTABLISHED, with no reported gain value.

Sivak's 2.27±0.07 coherence gain is a literature-reported physical-system result under its own device, best-passive denominator and fitted lifetime protocol. It cannot be transferred to our simulator or RTL. The manuscript therefore uses the fully qualified term finite-cutoff wall-clock operational boundary only for the historical result and explicitly states that the current work contains no measured logical lifetime or physical break-even result.

## Frozen taxonomy

| Quantity | Status |
| --- | --- |
| 300-us finite-cutoff wall-clock operational boundary | `ESTABLISHED_WITHIN_300US_FINITE_CUTOFF_MODEL` |
| Full-cost operational boundary | `NOT_ESTABLISHED` |
| Simulation-derived coherence gain | `NOT_ESTABLISHED`; value=`None` |
| Postselected break-even | `NOT_ESTABLISHED` |
| Experimental break-even | `NOT_ESTABLISHED` |

## Manuscript checklist

- State that every Phase-6D round remains in the primary denominator.
- Keep post-selection diagnostic, acceptance and rejection cost in one paragraph/table.
- Use the full finite-cutoff wall-clock operational-boundary qualifier.
- Keep coherence gain, full-cost and physical break-even NOT_ESTABLISHED.
- Attribute Sivak 2.27±0.07 only to the cited physical system.

## Missing author input

- `ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING`

## 中文核对

主指标不使用 post-selection；历史 post-selection 只作离线诊断且显式计入 rejection。当前只建立 300 us finite-cutoff matched-idle operational boundary；coherence gain、full-cost/postselected/experimental break-even 均未建立。
