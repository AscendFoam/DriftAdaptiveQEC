# Code Truth Audit

Freeze date: `2026-05-19`

This document audits where the code really does what the docs say, where the docs are ahead of the code, where placeholders/fallbacks still exist, and where historical results are being reused correctly.

## 1. Where Code Matches Docs

### 1.1 Mock-Backed Software HIL

- Docs say: P3 software HIL uses `mock` + `model_artifact` + `artifact_npz` + `inproc`.
- Code: `run_hil_suite.py` constructs mock noise provider when `hil.backend == "mock"`. Two SHA256-identical HIL runs confirm deterministic replay.
- Verdict: **consistent**

### 1.2 Deterministic RNG Chain

- Docs say: `RealisticSyndromeMeasurement` uses explicit `rng` injection; fast loop separates measurement noise RNG from error RNG.
- Code: `syndrome_measurement.py` line 84 accepts `rng: Optional[np.random.Generator]`. `fast_loop_emulator.py` lines 188-198 passes dedicated `self._measurement_rng = np.random.default_rng(...)`. All randomness goes through `_normal()` / `_random()` dispatching to `self._rng` when set.
- Verdict: **consistent**

### 1.3 P4 Benchmark as HIL Wrapper

- Docs say: P4 benchmark wraps the same HIL session stack.
- Code: `run_p4_multiscenario_benchmark.py` line 553 calls `run_hil_session(run_cfg, repeat_run_dir)` directly.
- Verdict: **consistent**

### 1.4 TFLite Stub Fallback

- Docs say: `export.py` tries true `.tflite` export first, falls back to `tflite_stub_v1` on failure. `inference_service.py` distinguishes `tflite_service` vs `tflite_stub_service`.
- Code: `export.py` lines 156-160 try-catch with fallback to `_export_tflite_stub()`. `inference_service.py` line 302 reports `source="tflite_stub_service"`, line 341 reports `source="tflite_service"`.
- Verdict: **consistent**

### 1.5 Statcalib Interface-Only

- Docs say: `statcalib` is interface contract only, not integrated into slow loop.
- Code: `cnn_fpga/decoder/statcalib.py` is 289 lines of dataclasses, constants, validation helpers, and convenience constructors. No algorithmic calibration logic. `slow_loop_runtime.py` (974 lines) does not import or reference `statcalib`.
- Verdict: **consistent**

### 1.6 Teacher Diagnostics Semantics

- Docs say: T28 repaired missing-vs-zero writer semantics. Output distinguishes `not_applicable`, `not_generated`, observed zero.
- Code: `slow_loop_runtime.py` `_teacher_branch_diagnostics()` (lines 293-307) emits `not_applicable` for modes without scalar teacher features, `not_generated` with reason for hybrid modes. Missing diagnostics remain empty/null, not coerced to `0.0`.
- Verdict: **consistent**

### 1.7 P4 Report Header

- Docs say: T29 fixed duplicate markdown report header.
- Code: `_write_report()` has one header row; static shape check confirmed `header_rows=1`, `column_counts=[12, 12, 12]`.
- Verdict: **consistent**

## 2. Where Docs Are Ahead of Code

### 2.1 board_backend.py "Placeholder" Label vs Reality

- Docs say: `board_backend.py` is a "placeholder real-board backend."
- Code: File is 308 lines with structurally complete `MemoryMappedRegisterIO`, `MemoryMappedDMARegion`, and `BoardFPGA` classes using `mmap`, AXI register maps, and DMA readout. The header docstring still says "Placeholder real-board backend using memory-mapped AXI/DMA interfaces."
- Assessment: **The code is more complete than the docstring suggests.** The "placeholder" label is a legacy misnomer. The structural code for memory-mapped register I/O, DMA regions, and board FPGA is implemented. However, since no real hardware has been connected, the evidence boundary is correctly at readiness-plan level.
- Verdict: **doc label is stale; code is more complete than docs claim; evidence boundary is correctly honest**
- Recommended follow-up: update the docstring to reflect implementation status while preserving the "not hardware-validated" evidence boundary.

### 2.2 Paper Background/Related Work Prose

- `docs/paper_background_related_work_draft.md` exists as bounded prose draft.
- No code, config, or experiment changes were made in T42/T43.
- The prose references supported claims correctly but has internal drafting annotations (`[stable conclusion 9.1 item 7]`, `[supported claim C3]`) that need cleanup before manuscript assembly.
- Verdict: **docs are ahead of evidence in the sense that prose exists for claims that are still partial/blocked, but the prose correctly marks them; no silent upgrade**

## 3. Where Placeholders or Fallbacks Still Exist

| Component | Placeholder/fallback type | Code evidence | Current boundary |
| --- | --- | --- | --- |
| `board_backend.py` | "Placeholder" label (stale); structurally complete but no real device connection | 308 lines of AXI/DMA code; docstring says "Placeholder" | R13/R14: readiness-plan only |
| `export.py` stub fallback | Automatic fallback to `tflite_stub_v1` when TensorFlow unavailable | Lines 156-160 try-catch | R12: true runtime not available |
| `inference_service.py` stub path | `tflite_stub_service` source when loading `.tflite.json` | Line 302 stub vs line 341 true | R12: stub ≠ deployment |
| `statcalib.py` | Interface-only contract; no algorithmic implementation | 289 lines of dataclasses/constants | R24: interface-only |
| Aggregation/report writer | No focused unit/static tests for format or null-semantics regression | No `test_*` file for report writer | R23: deferred |

## 4. Where Historical Results Are Reused Correctly

| Historical result | Reuse in current docs | Correct? |
| --- | --- | --- |
| T24 frozen-set `comparison.csv` | Referenced in C2, C3, T1 figure/table | Yes: frozen-set boundary explicitly stated |
| T12 SHA256-identical HIL runs | Referenced in C1 and deterministic recovery evidence | Yes: byte-identical reproduction confirmed |
| T38 `trace_rows.csv` (4798 rows) | Referenced in C4 mechanism diagnosis | Yes: single-seed boundary explicitly stated |
| T40 `train_report.json` | Referenced in C5 training smoke | Yes: isolated output path, clean-env boundary stated |
| T15 development bounded run | Referenced in handoff as predecessor to T24 | Yes: clearly labeled `development_smoke`, not formal benchmark |

## 5. Marker Comments Audit

- Searched `cnn_fpga/` and `physics/` for `# TODO`, `# FIXME`, `# HACK`, `# PLACEHOLDER`.
- **Zero instances found.**
- The only "placeholder" wording is in `board_backend.py` docstrings (legacy label).
- The only "stub" wording is in the intentional `tflite_stub_v1` mechanism.
- Verdict: **clean; no hidden shortcuts**

## 6. Main Audit Verdict

Current docs are mostly honest relative to code truth. The notable exception is the `board_backend.py` "placeholder" label, which understates the actual code completeness.

The larger problem is not doc/code contradiction. The larger problem is that several paper-relevant evidence lanes still stop at:

- `placeholder label` / `stub` / `plan` / `readiness`
- `single-seed diagnosis`
- `one bounded smoke`
- `interface-only contract`
- `structural zero in metric`

That means code truth currently supports a bounded software-HIL method/system paper, but not a strong deployment paper, a broad benchmark paper, or a hardware-validated paper.
