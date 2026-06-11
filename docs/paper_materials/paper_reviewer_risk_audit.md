# Paper Reviewer Risk Audit

## Scope Note

This audit assumes an adversarial reviewer. The purpose is to identify where the current paper state is vulnerable if drafting drifts beyond the evidence ledger in `docs/paper_materials/paper_claim_evidence_ledger.md`.

## Novelty Challenge Points

| ID | Likely reviewer objection | Current trigger | Concrete reference | Wording-only mitigation | Evidence-upgrade-needed mitigation |
| --- | --- | --- | --- | --- | --- |
| N1 | "This reads like a recovery report, not a novel method paper." | The strongest completed evidence is revalidation plus bounded recovery, not a new broad benchmark result. | `C1`, `C2`, `C3`, `C11` blocked, `R5`, `R9` | Frame the manuscript as an evidence-bounded systems/method recovery with explicit benchmark boundary rather than a new-state-of-the-art claim. | Expand beyond the frozen-set benchmark and add fresh comparator coverage before claiming broader method novelty. |
| N2 | "Winning four frozen scenarios is too narrow to establish a generally stronger decoder." | `hybrid_residual_b` win is real but limited to the frozen set. | `C3`, `T1`, `C11` blocked | Use "wins all four frozen scenarios under the current formal protocol" rather than "outperforms baselines broadly." | Run a broader scenario set and predeclare protocol expansion. |
| N3 | "The statcalib component is not yet a results-bearing novelty contribution." | Current evidence stops at interface contract and focused tests. | `C9` supported, `C10` blocked, `R24` | Keep statcalib in the paper as a boundary/status subsection, not a headline contribution. | Integrate statcalib into comparator experiments and report measured behavior. |

## Evidence-Grade Challenge Points

| ID | Likely reviewer objection | Current trigger | Concrete reference | Wording-only mitigation | Evidence-upgrade-needed mitigation |
| --- | --- | --- | --- | --- | --- |
| E1 | "This is not hardware validation; it is software-only." | The validated path is mock-backed software HIL. | `C1` supported, `C8` blocked, `T4`, `R13`, `R14` | Write "mock-backed software HIL revalidation" everywhere. Remove any wording that implies board execution evidence. | Execute the real-board smoke/readiness plan and collect board-side logs and outputs. |
| E2 | "Deployment claims are unsupported because `.tflite` runtime is not working." | Entry points/stubs exist, but true runtime validation does not. | `C7` blocked, `T4`, `R12` | Restrict deployment language to runtime-path readiness gaps. | Restore true `.tflite` export/runtime and capture bounded validation evidence. |
| E3 | "The benchmark is too frozen and too small to support paper-grade empirical claims." | T24/T25 evidence is formal but intentionally frozen-set only. | `C2`, `C3`, `C11` blocked, `T1`, `T2`, `R5`, `R9` | Explicitly label the benchmark as frozen-set formal software revalidation. | Expand benchmark breadth, scenario diversity, and comparator coverage under a declared protocol. |
| E4 | "Mechanism claims are under-supported." | The diagnosis is trace-supported but only on one seed. | `C4` partial, `F1`, `R10` | Use "single-seed trace-supported diagnosis" and avoid "mechanism proven" language. | Add multi-seed traces, intervention/mitigation experiments, or causal tests. |
| E5 | "The training chain evidence is too thin for reproducibility claims." | Only one clean CPU-only real smoke exists. | `C5` supported, `C6` blocked, `T3`, `R11` | Present it as a successful clean-environment smoke, not as reproducibility validation. | Add repeated runs, OS variation, and GPU/runtime coverage. |

## Overclaim Wording Traps

| ID | Wording trap | Why it is unsafe | Concrete reference | Safe replacement |
| --- | --- | --- | --- | --- |
| W1 | "hardware validated" | Upgrades software HIL into board evidence. | `C1` vs `C8`, `T4` | "mock-backed software HIL revalidated" |
| W2 | "deployment-ready" | Implies true runtime and board evidence that do not exist. | `C7`, `C8`, `R12`, `R13`, `R14` | "deployment/readiness boundary documented" |
| W3 | "reproducible training pipeline" | One smoke run does not establish reproducibility. | `C5` vs `C6`, `T3`, `R11` | "one clean-environment CPU-only training smoke completed" |
| W4 | "mechanism explained" or "root cause proven" | The present evidence is diagnostic, not causal proof. | `C4`, `F1`, `R10` | "single-seed trace-supported diagnosis suggests..." |
| W5 | "comprehensive benchmark" | The benchmark is intentionally frozen-set only. | `C2`, `C3`, `C11`, `T2` | "frozen-set formal software revalidation benchmark" |
| W6 | "integrated calibrated comparator" | statcalib has not yet been validated in-system. | `C9` vs `C10`, `T5`, `R24` | "statcalib interface contract and focused tests" |

## Reproducibility and Deployment Challenge Points

| ID | Likely reviewer objection | Current trigger | Concrete reference | Wording-only mitigation | Evidence-upgrade-needed mitigation |
| --- | --- | --- | --- | --- | --- |
| R1 | "Can another group reproduce the training result on a different host?" | Current proof is one Windows/Python 3.12 CPU-only smoke. | `C5`, `C6`, `T3`, `R11` | State exact host boundary in the methods and limitations sections. | Add repeatability matrix across machines, OSes, and optionally GPU backends. |
| R2 | "Where is the deployment path if runtime conversion is blocked?" | `.tflite` runtime remains unavailable on the current machine. | `C7`, `T4`, `R12` | State that deployment discussion is a readiness gap analysis, not runtime validation. | Produce validated `.tflite` export/inference evidence. |
| R3 | "Why discuss board execution if there are no board results?" | Real-board work is plan/readiness only. | `C8`, `T4`, `R13`, `R14` | Keep board content in limitations/future work, not results. | Run bounded board smoke and compare outputs to software reference. |

## Ablation and Mechanism Challenge Points

| ID | Likely reviewer objection | Current trigger | Concrete reference | Wording-only mitigation | Evidence-upgrade-needed mitigation |
| --- | --- | --- | --- | --- | --- |
| A1 | "Why should the reviewer trust the residual pathway explanation?" | The current diagnosis is one-seed and trace-derived. | `C4`, `F1`, `R10` | Present it as a hypothesis strengthened by trace evidence, not a settled explanation. | Add targeted ablations or counterfactual intervention experiments. |
| A2 | "Where are broader ablations across decoder families or correction regimes?" | Current results emphasize the frozen-set ranking, not a full ablation suite. | `C3`, `C11` blocked, `R20` | Keep ablation language narrow and scenario-bound. | Add broader ablation tables and correction-regime analysis. |
| A3 | "How do known structural zeros affect interpretation?" | Correction saturation / structural-zero risk remains open. | `R20` | Mention the unresolved structural limitation in threats to validity. | Add dedicated analysis or mitigation experiments for the saturation regime. |

## Section-by-Section Reviewer Hotspots

| Section | Reviewer hotspot | What to watch |
| --- | --- | --- |
| Abstract | Evidence inflation | Do not include hardware, runtime, reproducibility, or broad superiority language. |
| Introduction | Contribution inflation | Keep contributions bounded to `C1`, `C2`, `C3`, `C5`, `C9`, plus `C4` with caveat. |
| Method | System-boundary ambiguity | Distinguish software HIL, training smoke, deployment readiness, and board readiness as separate evidence grades. |
| Experiments | Benchmark-scope drift | Do not let frozen-set protocol wording drift into "comprehensive evaluation." |
| Results | Mechanism overread | `F1` can support diagnosis wording only. |
| Limitations | Under-reporting blockers | `C6`, `C7`, `C8`, `C10`, `C11` should all be surfaced explicitly. |
| Conclusion | Future-work disguised as completion | Avoid phrases like "nearly deployment-ready" or "effectively reproducible." |

## Minimum Safe Paper Positioning

If the draft must be defensible today, its strongest safe positioning is:

- a bounded recovery and revalidation paper for a CNN-assisted decoding pipeline
- validated at the mock-backed software-HIL and frozen-set benchmark level
- with one clean-environment CPU-only training smoke
- with explicit disclosure that deployment/runtime, real-board, broader benchmark, and integrated statcalib evidence are not yet complete

## Do-Not-Publish-As-Claimed List

The following paper framings are not supportable with current evidence:

- real-board HIL validation paper
- deployment-complete `.tflite` inference paper
- cross-platform reproducible training paper
- broad comparator/ablation superiority paper
- integrated statcalib evaluation paper
