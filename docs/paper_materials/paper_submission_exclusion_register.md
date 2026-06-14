# T86 Submission Exclusion Register

本表只记录当前 submission-facing package 中必须显式排除的 surface。它不是“以后也永远不能写”，而是“在当前证据层级下不能进入这轮 package claim”。

| exclusion_id | blocked_surface | why_excluded_now | do_not_claim_wording | future_unblock_task |
| --- | --- | --- | --- | --- |
| `EX01` | real-board execution / timing / resource rows | 当前仍无 `Linux + FPGA` host、openable device path 与真实板级测量；现有 `T49/T71/T72` 只支持 read-only gate / regeneration / provenance。 | `real-board execution succeeded` / `hardware validated` / `board-level latency is measured` | `future_host_real_board_gate_or_smoke_task` |
| `EX02` | default-env / cross-host `.tflite` portability story | 当前只有 isolated current-host true `.tflite` runtime 窄路径；没有 default-env recovered、cross-host portability 或 HIL closure。 | `default environment recovered` / `deployment closure is complete` / `portable runtime path is established` | `default_env_or_portability_audit_task` |
| `EX03` | full training reproducibility closure | 当前只有 canonical chain intact + one clean CPU-only bounded rerun，不支持 repeated-run、cross-host、GPU/CUDA 或 Linux portability closure。 | `full reproducibility is closed` / `training is fully portable across hosts` | `repeated_run_or_cross_host_repro_task` |
| `EX04` | `statcalib` mature comparator promotion | `FR8/statcalib` 仍是 supplement-side extension lane，且保持 no-promotion / no unique clean threshold。 | `statcalib is the promoted final comparator` / `a unique clean threshold has been established` | `comparator_promotion_gate_task` |
| `EX05` | expanded benchmark / stronger oracle baseline story | 当前主线 formal anchor 仍是 `T24` frozen set；更广 drift family 与更强 baseline 还未进入新的 protocol-then-execution route。 | `paper-grade expanded benchmark is complete` / `the ranking is proven beyond the frozen set` | `benchmark_expansion_protocol_then_execution_task` |
| `EX06` | unified portability/deployment closure figure or prose | 训练、isolated `.tflite` runtime、real-board gate 仍是分层边界证据，当前没有诚实的统一闭环图或统一闭环叙事。 | `a unified portability pipeline is now in place from training to deployment` | `cross_surface_portability_closure_task` |
