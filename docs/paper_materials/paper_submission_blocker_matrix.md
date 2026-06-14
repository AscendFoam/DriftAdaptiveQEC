# T85 Submission Blocker Matrix

本矩阵回答的不是“当前 note 能不能继续人工润色”，而是“如果下一步进入 submission-facing assembly，哪些 surface 仍必须被显式排除、降级或单独开任务”。

| blocker_id | blocker_type | affected_surface | why_not_ready | next_bounded_task |
| --- | --- | --- | --- | --- |
| `SB01` | `assembly_not_executed_yet` | submission-facing pack itself | `T85` 只做 preflight、residual wording-lag 清扫与 blocker 明确化；当前仓库还没有一张专门把主文/附录/补充/排除项装配成 submission-facing bundle 的 bounded assembly 任务。 | `bounded_submission_pack_assembly_task` |
| `SB02` | `hardware_host_missing` | board-level execution / latency / resource surface | 当前仍无 `Linux + FPGA` host、openable device path 与真实板级测量，因此任何 execution/timing/resource row 都不能进入 submission-facing completion 叙事。 | `future_host_real_board_gate_or_smoke_task` |
| `SB03` | `runtime_portability_not_closed` | default-env `.tflite` / deployment portability surface | 当前只有 isolated current-host true `.tflite` runtime 窄路径，没有 default-env recovered、HIL closure 或 deployment portability closure。 | `default_env_or_portability_audit_task` |
| `SB04` | `reproducibility_not_closed` | full training reproducibility surface | 当前只有 canonical chain intact + one clean CPU-only bounded rerun，不支持 repeated-run、cross-host、GPU/CUDA 或 Linux portability closure。 | `repeated_run_or_cross_host_repro_task` |
| `SB05` | `extension_lane_no_promotion` | `FR8/statcalib` promoted-comparator surface | `statcalib` 当前仍是 supplement-side extension lane，且保留 no-promotion / no unique clean threshold；submission-facing assembly 不能把它压成成熟主线 comparator。 | `comparator_promotion_gate_task` |
| `SB06` | `expanded_benchmark_not_opened` | stronger oracle baselines / broader drift families / paper-grade expanded benchmark surface | 当前主线 formal anchor 仍是 `T24` frozen set；更广 benchmark 族和更强 baseline 还没有进入新的 protocol-then-execution route。 | `benchmark_expansion_protocol_then_execution_task` |
