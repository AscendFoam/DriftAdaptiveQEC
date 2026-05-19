# Claim Evidence Table

Freeze date: `2026-05-19`

This is the recovery-mode claim-by-claim ledger. It cross-references `docs/paper_claim_evidence_ledger.md` (C1-C11) and adds recovery-mode-specific claims (RRC12-RRC15) about the recovery process itself.

Status labels: `supported` / `partial` / `blocked`

## 1. Paper-Facing Claims (C1-C11 from paper ledger)

| ID | Claim | Status | Concrete evidence path | Missing link / blocker | Paper section impact |
| --- | --- | --- | --- | --- | --- |
| C1 | A bounded P3 software HIL path has been recovered and revalidated. | `supported` | `docs/03_hil_p4_boundary_audit.md`; `docs/P3_software_hil_bootstrap.md`; two HIL runs with SHA256-identical outputs | Not real-board, not `.tflite` runtime | System framing |
| C2 | The historical frozen P4 comparison set has been re-run under locked protocol. | `supported` | `docs/P4_benchmark_formal_protocol.md`; `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`; `comparison.csv`; T24/T25 reviews | Still frozen-set only (4 scenarios x 5 modes x 2 repeats) | Benchmark section |
| C3 | `hybrid_residual_b` wins all four frozen scenarios. | `supported` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | Cannot generalize beyond frozen set | Results claim |
| C4 | `seed=20260429` failure has mechanism evidence. | `partial` | `docs/seed20260429_failure_diagnosis.md`; `docs/seed20260429_trace_export_diagnosis.md`; `runs/T38_seed20260429_trace_probe_20260513/trace_export/trace_rows.csv` | Single-seed only; no causal intervention or multi-seed confirmation | Mechanism discussion only |
| C5 | One clean Windows/Python 3.12 CPU-only environment can execute one real training smoke. | `supported` | `requirements-train-cpu-win-py312.txt`; `docs/training_chain_cpu_cleanenv_bootstrap.md`; `docs/training_chain_cpu_cleanenv_train_smoke.md`; T40-isolated `train_report.json` | Only one bounded smoke | Narrow reproducibility wording |
| C6 | Training chain is fully reproducible across OS/GPU/host. | `blocked` | `docs/training_chain_portable_dependency_lock_plan.md`; `docs/training_chain_cpu_cleanenv_train_smoke.md` | R11: no cross-host, cross-OS, GPU, or repeated-run evidence | Blocks broader reproducibility claim |
| C7 | True `.tflite` export/runtime is restored. | `blocked` | `docs/TFLite_runtime_bootstrap.md`; `cnn_fpga/model/export.py` (stub fallback path exists) | R12: `tensorflow` / `tflite_runtime` not installed | Blocks deployment claim |
| C8 | Real-board HIL has been validated. | `blocked` | `docs/real_board_hil_readiness.md`; `docs/real_board_smoke_execution_plan.md` | R13/R14: no device, no board logs, AXI/DMA contract unverified | Blocks hardware claim |
| C9 | `statcalib` has a separate interface contract and focused tests. | `supported` | `cnn_fpga/decoder/statcalib.py`; `tests/test_statcalib_interface.py`; T26/T30 reviews | No slow-loop integration, no benchmark evidence | Status subsection only |
| C10 | `statcalib` is a formal integrated comparator. | `blocked` | `cnn_fpga/decoder/statcalib.py` | R24: interface-only; `slow_loop_runtime.py` does not import `statcalib` | Blocks comparator claim |
| C11 | Current evidence constitutes paper-grade expanded benchmark evidence. | `blocked` | `docs/review/T25_p4_formal_evidence_gate_review.md` | R5/R9/R10/R11/R12/R13: fragmented and bounded | Blocks broad empirical positioning |

## 2. Recovery-Mode-Specific Claims

| ID | Claim | Status | Evidence path | Missing link |
| --- | --- | --- | --- | --- |
| RRC12 | T44 alone can raise the project to strong-submission standard. | `blocked` | T44 task package; `docs/08_risks_and_open_questions.md` | T44 freezes truth only; does not create missing evidence |
| RRC13 | Currently visible pending tasks (T32, T37) are sufficient for paper closure. | `blocked` | `docs/04_task_board.md` | Missing benchmark broadening, mechanism closure, reproducibility, ablation/material pack |
| RRC14 | `docs/reference/延伸改进思路.md` is part of the current mainline. | `blocked` | `docs/reference/延伸改进思路.md` | This is a future extension reference; not mainline truth |
| RRC15 | Paper prose expansion may resume after T44. | `blocked` | `docs/review/T43_review.md`; `docs/reference/科研纠偏意见.md` | User explicitly required evidence-first; prose must wait for mainline blockers |

## 3. Cross-Reference to Existing Risk Register

| Risk ID | Affects claims | Current status |
| --- | --- | --- |
| R5 | C2, C3, C11 | Open: frozen-set only |
| R8 | C1 | Open: bounded software path wording |
| R9 | C2, C3, C11 | Open: frozen-set protocol boundary |
| R10 | C4 | Open but narrowed: single-seed trace evidence |
| R11 | C5, C6 | Open but narrowed: one clean-env smoke |
| R12 | C7 | Open: runtime unavailable |
| R13, R14 | C8 | Open: no device, no logs |
| R20 | C3 | Open but narrowed: structurally sound metric, genuine zero in tested range |
| R23 | — | Open: no focused tests for aggregation/report writer |
| R24 | C9, C10 | Open: interface-only statcalib |
| R25 | C1-C11 | Open: narrative ran ahead of evidence; now paused |

## 4. Summary Statistics

- `supported`: C1, C2, C3, C5, C9 (5 claims)
- `partial`: C4 (1 claim)
- `blocked`: C6, C7, C8, C10, C11, RRC12, RRC13, RRC14, RRC15 (9 claims)

The ratio of blocked to supported claims means the paper can proceed with framing and bounded prose drafting, but cannot make strong empirical claims beyond the frozen-set ranking result until mainline blockers are resolved.
