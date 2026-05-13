# T30 Independent Adversarial Review

## Verdict

`PASS`

## Reviewer

Independent adversarial reviewer, 2026-05-13.

## Scope Confirmation

T30 is a bounded interface contract and minimal implementation package. Reviewer confirms:

| Check | Result |
|-------|--------|
| `param_mapper.py` unchanged | Confirmed: `git diff` empty |
| `slow_loop_runtime.py` unchanged | Confirmed: `git diff` empty |
| `run_p4_multiscenario_benchmark.py` unchanged | Confirmed: not in diff |
| No new run directory | Confirmed: `runs/p4_benchmark/` latest entry predates T30 |
| No frozen-set expansion | Confirmed |
| No config changes | Confirmed |

## Files Changed (T30 Scope Only)

| File | Change | Within Allowed Set |
|------|--------|--------------------|
| `cnn_fpga/decoder/statcalib.py` | New | Yes |
| `tests/test_statcalib_interface.py` | New | Yes (task package allows focused test under `tests/`) |
| `docs/statcalib_feasibility_gate.md` | Modified (Minimal Comparator Interface section tightened) | Yes (task package allows "only if clarifying the interface contract") |
| `docs/review/T30_review.md` | New | Yes |
| `docs/for_human/T30_explanation.md` | New | Yes |
| `docs/tasks/Phase2/T30_statcalib_interface_contract.md` | Modified (Worker Output + Verification Record appended) | Yes |

Note: the working tree also contains governance doc changes from T26 Captain closeout (`docs/00~08`, task board, handoff). Those are not T30 Worker output and must be reviewed separately.

## Task Completion Check

### Required output from task package

| Required Output | Present | Assessment |
|-----------------|---------|------------|
| `StatCalibInput` exact contract | Yes | 10 fields with types, `__post_init__` validation, immutable frozen dataclass |
| `StatCalibOutput` exact contract | Yes | 8 fields, status/reason validation, `to_runtime_params()` conversion, factory methods |
| Status values and reason strings | Yes | 4 status values, 5 reason values, all module-level constants |
| Provenance / source fields | Yes | Both Input and Output carry `source`, `provenance`, `metadata` |
| Conversion boundary to `DecoderRuntimeParams` | Yes | `to_runtime_params()` only works for `generated` status; raises `ValueError` otherwise |
| Separate `cnn_fpga/decoder/statcalib.py` | Yes | New file, imports only `param_bank.DecoderRuntimeParams` and numpy |
| No changes to current mapper behavior | Confirmed | `param_mapper.py` diff is empty |
| `docs/review/T30_review.md` | Yes | Worker self-review present |
| `docs/for_human/T30_explanation.md` | Yes | Concise explanation present |
| Task package Worker Output / Verification Record | Yes | Appended to task package |

## Code Quality Review

### `cnn_fpga/decoder/statcalib.py`

**Positive observations:**

1. **Frozen dataclasses** enforce immutability — appropriate for a contract module.
2. **`__post_init__` validation** coerces types and checks invariants (shape, finiteness, non-empty strings). This is honest boundary validation, not decoration.
3. **`_coerce_vector2` / `_coerce_matrix22`** handle shape and finiteness checks. The `.copy()` calls prevent aliasing.
4. **`from_delta_b()`** computes `K = prior.K.copy()` (identity K passthrough) and `b = prior.b + delta_b`. This is the minimal sensible "residual-b" baseline behavior. Provenance is merged from both input and caller.
5. **Status/reason separation** is explicit: `not_generated` vs `not_applicable` vs `diagnostic_error` are distinct, preserving T28 semantics.
6. **`to_runtime_params()`** guards on `status == generated` — non-generated output cannot silently produce runtime params.
7. **No import of `ParamMapper`** — the module is genuinely separate.

**Observations (not issues):**

- The module imports `DecoderRuntimeParams` from `param_bank`, which is the correct dependency direction (statcalib depends on the runtime param contract, not on the mapper).
- `to_dict()` methods are present for serialization/auditability. Not strictly required for the interface contract but not unreasonable.

### `tests/test_statcalib_interface.py`

**6 tests covering:**

| Test | What it verifies |
|------|-----------------|
| `test_generated_output_converts_to_runtime_params` | Full generated path: `from_delta_b()` → runtime params conversion, K/b values, metadata fields |
| `test_not_generated_output_preserves_null_semantics` | `not_generated` output has `None` K/b/delta_b, `to_runtime_params()` raises |
| `test_not_applicable_output_is_separate_from_not_generated` | `not_applicable` is a distinct status |
| `test_invalid_reason_is_rejected` | Unknown reason string rejected at construction |
| `test_invalid_input_prediction_shape_is_rejected` | Missing required teacher_prediction keys rejected |
| `test_non_generated_output_cannot_carry_runtime_arrays` | Non-generated output with K/b arrays rejected at construction |

All 6 tests pass. Coverage is appropriate for an interface contract module.

## Pseudo-Implementation / Mock / Stub / Hardcode Check

- No mock backends. No stub logic. No placeholder return values.
- `from_delta_b()` does real computation: `K = prior.K.copy()`, `b = prior.b + delta_b`.
- The module is a genuine contract with real validation, not a skeleton.

The only "minimal" aspect is that `from_delta_b()` uses identity K passthrough — but this is the correct baseline behavior for a residual-b comparator that only modifies `b`.

## Over-Engineering Check

Not over-engineered. The module has:
- Frozen dataclasses with validation (appropriate for an API boundary)
- 4 status values + 5 reason values (appropriate for T28 status semantics)
- Factory methods for common construction paths
- `to_dict()` serialization

No unnecessary abstractions, no class hierarchies, no strategy patterns, no configuration objects. The complexity matches the contract's role as a type-safe API boundary.

## Benchmark / Protocol Integrity Check

- Frozen T24 benchmark set: unchanged.
- `ParamMapper.map_prediction()`: unchanged.
- Benchmark runner: unchanged.
- No statcalib added to ranked set.
- No config, scenario, baseline, seed, or repeat policy changes.

## Documentation Honesty Check

- Gate doc (`docs/statcalib_feasibility_gate.md`): "Explicit Non-Claims" section still says "This document does not claim statcalib exists" and "does not claim statcalib has been validated." The interface contract module now exists, but it is not validated beyond interface-level tests. The intent of these non-claims remains correct.
- Task package Worker Output: honest about scope and verification.
- For-human doc: correctly explains that this is interface contract work, not benchmark validation.

## DecoderRuntimeParams Compatibility

Confirmed:
- `DecoderRuntimeParams` requires `K: np.ndarray` (shape `(2,2)`) and `b: np.ndarray` (shape `(2,)`).
- `statcalib.py` uses `_coerce_matrix22` and `_coerce_vector2` to enforce exactly these shapes.
- `to_runtime_params()` constructs `DecoderRuntimeParams` with correct shapes and metadata.

## Blocking Issues

None.

## Non-Blocking Issues

N1: `docs/statcalib_feasibility_gate.md` "Explicit Non-Claims" says "No source code, config, run, or artifact was changed." T30 has since added `statcalib.py`. This statement was accurate for T26 but is now stale. Should be annotated or updated in the next Captain closeout.

N2: `tests/` directory has no `__init__.py`. This works for `python -m unittest` but may cause issues with IDE test discovery or `pytest`. Not blocking for T30 but should be addressed if the test directory grows.

N3: Running the tests created `tests/__pycache__/test_statcalib_interface.cpython-312.pyc`. This is a known repo-noise side effect per T19/T28 tracked-cache governance. Should not be committed as task output.

N4: `from_delta_b()` uses `prior.b + delta_b` as the baseline `b` computation. This is the simplest possible residual-b logic. Any follow-up task adding real calibration logic should verify this is the intended baseline and not accidentally bake in a "residual-b only" assumption that limits future calibration approaches.

## Missing Tests

- No edge-case test for `diagnostic_error` status construction or conversion behavior.
- No test for `to_dict()` round-trip (serialization correctness).

These are minor gaps for an interface contract module. The 6 existing tests cover the critical paths (generated, not_generated, not_applicable, invalid inputs). Edge-case coverage can be added in a follow-up task if needed.

## Suspicious Implementation Details

None found.

## Assessment of Worker Self-Review

The Worker's self-review (previously in this file) was honest and covered the key points. It correctly identified that SlowLoopRuntime integration was out of scope and that the unittest choice was acceptable. Minor gap: it did not mention the gate doc update or the `__pycache__` side effect.

## Recommended Next Action

1. Captain accepts T30 as `PASS`.
2. If a follow-up task is approved, it should:
   - Wire this contract into a separate statcalib comparator lane in the slow-loop runtime
   - Keep the frozen benchmark set unchanged
   - Add focused integration smoke before any broader benchmark
3. N1 (gate doc staleness) should be cleaned up during Captain closeout.
4. R10 / R20 / R23 remain open and should be tracked in any follow-up task's risk assessment.
5. T36 (seed failure diagnosis) remains a separate priority.
