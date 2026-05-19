# Figure And Result Ledger

Freeze date: `2026-05-19`

This ledger catalogs every figure, table, and result intended for the paper, with source, data path, seed/scenario linkage, and readiness status.

Status labels: `ready` / `partial` / `missing` / `blocked`

## 1. Paper Figures (from C1-C11 claim ledger)

| ID | Planned figure | Status | Source data path(s) | Seed/scenario linkage | Blocker |
| --- | --- | --- | --- | --- | --- |
| F1 | `seed=20260429` mechanism-diagnosis figure: per-window `teacher_b`, predicted `delta_b`, committed `b`, window outcome | `partial` | `runs/T38_seed20260429_trace_probe_20260513/trace_export/trace_rows.csv`<br>`runs/T38_seed20260429_trace_probe_20260513/trace_export/paired_repeat_comparison.csv` | `seed=20260429` only; `hybrid_residual_b` mode | R10: single-seed only; usable as bounded one-seed diagnosis figure |
| F2 | Benchmark evidence-boundary diagram: P3 software HIL, T24 formal revalidation, TFLite boundary, real-board readiness | `ready` | `docs/03_hil_p4_boundary_audit.md`<br>`docs/P4_benchmark_formal_protocol.md`<br>`docs/TFLite_runtime_bootstrap.md`<br>`docs/real_board_hil_readiness.md` | N/A (schematic) | None; safe because it is a boundary diagram, not a result-upgrade figure |
| F3 | Training portability/reproducibility figure implying cross-platform or GPU readiness | `blocked` | `docs/training_chain_portable_dependency_lock_plan.md`<br>`docs/training_chain_cpu_cleanenv_train_smoke.md` | One clean CPU-only smoke | R11: current evidence supports a boundary table, not a broad portability figure |

## 2. Paper Tables (from C1-C11 claim ledger)

| ID | Planned table | Status | Source data path(s) | Seed/scenario linkage | Blocker |
| --- | --- | --- | --- | --- | --- |
| T1 | Frozen-set benchmark ranking summary | `ready` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`<br>`summary.json` | 4 scenarios x 5 modes x 2 repeats; paired seeds | Must stay labeled `mock-backed software HIL formal software revalidation` |
| T2 | Benchmark boundary / evidence-level table | `ready` | `docs/03_hil_p4_boundary_audit.md`<br>`docs/P4_benchmark_formal_protocol.md`<br>`docs/review/T25_p4_formal_evidence_gate_review.md` | N/A (status table) | None; explicit boundary table |
| T3 | Training reproducibility boundary table | `ready` | `docs/training_chain_portable_dependency_lock_plan.md`<br>`docs/training_chain_cpu_cleanenv_bootstrap.md`<br>`docs/training_chain_cpu_cleanenv_train_smoke.md` | One clean CPU-only smoke | Safe only as a boundary table |
| T4 | Deployment / readiness boundary table | `ready` | `docs/03_hil_p4_boundary_audit.md`<br>`docs/TFLite_runtime_bootstrap.md`<br>`docs/real_board_hil_readiness.md` | N/A (readiness table) | None; non-claim table |
| T5 | `statcalib` comparator status table | `partial` | `cnn_fpga/decoder/statcalib.py`<br>`tests/test_statcalib_interface.py` | N/A (interface only) | R24: benchmark-evidence cells blocked |

## 3. Additional Paper Figures/Tables Not in Original Ledger

| ID | Item | Status | Source path(s) | Blocker |
| --- | --- | --- | --- | --- |
| FR1 | System architecture figure: fast loop / slow loop / param bank / HIL boundary | `partial` | `physics/`<br>`cnn_fpga/runtime/`<br>`cnn_fpga/hwio/` | Publication-grade figure asset not frozen; needs explicit generation script |
| FR4 | Scenario-wise benchmark summary figure (LER curves or bar charts per scenario) | `partial` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | Figure-generation script/path not frozen in recovery docs |
| FR6 | Multi-seed mechanism/intervention figure | `missing` | None | Needs new bounded task and multi-seed evidence |
| FR7 | Feature/teacher ablation result table | `missing` | No frozen recovery-grade result pack | Needs ablation result-pack task (T47) |
| FR8 | Statcalib comparator result table | `missing` | `cnn_fpga/decoder/statcalib.py` (interface only) | No integrated comparator run; needs slow-loop integration task |
| FR10 | True `.tflite` runtime latency/accuracy figure | `blocked` | `docs/TFLite_runtime_bootstrap.md` | R12: runtime unavailable |
| FR11 | Real-board smoke evidence figure/table | `blocked` | `docs/real_board_hil_readiness.md` | R13/R14: no device, no logs |
| FR12 | Latency / commit / violation summary table | `partial` | T24 `comparison.csv` and `summary.json` | Exact table shape and regeneration step not frozen |

## 4. Summary Statistics

- `ready`: F2, T1, T2, T3, T4 (5 items)
- `partial`: F1, T5, FR1, FR4, FR12 (5 items)
- `missing`: FR6, FR7, FR8 (3 items)
- `blocked`: F3, FR10, FR11 (3 items)

## 5. Direct Answer to Recovery-Mode Question

T44 plus currently visible pending tasks (T45-T52) do not yet cover all high-value paper materials.

The largest gaps are:

1. **FR6**: multi-seed mechanism/intervention figure (missing)
2. **FR7**: feature-ablation result table (missing)
3. **FR8**: statcalib comparator result table (missing)
4. **FR1/FR4/FR12**: stable regeneration paths for architecture figure, scenario-wise summary, and latency table (partial)

These gaps require bounded execution tasks before the paper has a complete figure/material pack.

## 6. Extension Classification

Items inspired by `docs/reference/延伸改进思路.md` should currently be treated as:
- optional enhancement candidates for later figure/result growth
- not required evidence for the current minimum mainline thesis

They only become mainline material after separate task approval and evidence generation.
