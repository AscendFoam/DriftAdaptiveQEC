# Reviewer response: experiment relevance without quantum hardware

- Task: `T7.3.3`
- Verdict: `PASS_EXPERIMENT_RELEVANCE_WITHOUT_HARDWARE_OVERCLAIM`
- Package readiness: `draft_with_placeholders`
- Gates/mutations: `24/24` / `24/24`

## Point-by-point response

We agree that the phrase experiment-related is ambiguous when no quantum processor or physical FPGA measurement is part of the study. We therefore use experiment-informed or experiment-facing, always paired with the evidence modality, and do not describe this work as experimental GKP quantum error correction.

The revised evidence ladder separates seven levels. Literature-reported physical results, including AQEC lifetime gains and Sivak-style break-even, remain facts about the cited systems. Two official-code CPD threshold cells are numerical reproductions in a different surface-lattice task. Our own decoder and AQEC results are project-native simulations; the legacy HIL path uses a mock FPGA backend. None of those levels is a board or quantum-hardware measurement.

The hardware contribution is narrower but substantive. The exact single-mode production top passes 17 formal gates, kills 21 targeted mutants, and matches an independent reference over ten 100,000-cycle CXXRTL families with zero full-vector mismatch, undefined action or silent overflow. Three open-source place-and-route seeds meet the 27-MHz contract, with whole-harness Fmax between 36.794 and 37.869 MHz. These are pre-board digital qualification results, not measured latency, jitter, deadline or power.

The dedicated board gate remains blocked: every physical measurement field is null, the historical UART candidate was neither programmed nor measured, and the optional real-GKP-data and control-chain tasks remain Todo. Accordingly, AQEC or Sivak evidence cannot validate our simulator, decoder, RTL or board. The present experiment-facing relevance comes from causal observation contracts, fault paths, deadlines and deployment interfaces designed for a future control chain, not from claiming that such an experiment has already been performed.

We have made this distinction explicit in the title, abstract, Methods, Results and Limitations. A future promotion to physical evidence will require a board-identified, bitstream-bound measurement pack and, separately, licensed real GKP syndrome data with protocol metadata and valid labels or tomography.

## Evidence ladder

| Level | Current status | Allowed | Forbidden |
| --- | --- | --- | --- |
| `LITERATURE_FACT` | `AVAILABLE_CONTEXT_ONLY` | attribute reported values to the cited physical system | project measurement or reproduction |
| `OFFICIAL_CODE_REPRODUCTION` | `AVAILABLE_TWO_CPD_CELLS` | report source-qualified numerical reproduction within its task signature | device or project-native hardware evidence |
| `PROJECT_NATIVE_SIMULATION` | `AVAILABLE_MIXED_POSITIVE_NEGATIVE` | report simulator-scoped LER, lifetime and fault-path results | physical lifetime, device uncertainty or break-even |
| `MOCK_SOFTWARE_HIL` | `AVAILABLE_LEGACY_MOCK_ONLY` | report software orchestration and mock event semantics | real-board HIL or physical timing |
| `PREBOARD_DIGITAL_QUALIFICATION` | `AVAILABLE_EXACT_SINGLE_MODE` | formal/CXXRTL/P&R-qualified deterministic atomic fail-closed architecture | measured board latency/power, multimode RTL or fastest FPGA |
| `PHYSICAL_BOARD_MEASUREMENT` | `BLOCKED_ALL_FIELDS_NULL` | state blocker and recovery conditions | board correctness, latency, jitter, deadline, resources, power or speed |
| `QUANTUM_HARDWARE_OR_REAL_GKP_DATA` | `ABSENT_OPTIONAL_PHASE8_TODO` | state optional future route and required permissions/metadata | cavity/transmon, real-syndrome, frame-update or active-feedback result |

## Manuscript checklist

- Use experiment-informed/facing only with an adjacent evidence modality.
- Keep the title free of experimental-GKP or device-demonstration language.
- Keep simulation, mock HIL, formal/CXXRTL, P&R estimate and physical measurement in separate rows.
- Keep every physical-board field null until a board-identified raw measurement pack exists.
- Keep AQEC/Sivak physical results attributed to their systems and optional Phase 8 explicitly future-facing.

## Missing author input

- `ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING`

## 中文核对

本回答把“实验相关”收紧为“实验启发/面向实验接口”，并逐层区分文献事实、官方代码复现、项目原生仿真、mock 软件 HIL、预板 RTL、真板测量与真实 GKP/量子硬件。AQEC/Sivak 的物理证据不迁移；当前真板字段全部为空，Phase 8 仍为可选 Todo。
