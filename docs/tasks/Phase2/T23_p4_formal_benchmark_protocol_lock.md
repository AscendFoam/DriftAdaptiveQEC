# T23: P4 Formal Benchmark Protocol Lock

Task ID: `T23`

Goal: 在不运行 benchmark 的前提下，把下一轮 P4 formal benchmark 的 protocol、证据缺口、计算预算和 go/no-go 条件锁定清楚，为后续有界执行任务做准备。

Why now: `T21` milestone gate 已确认当前 `T15` 仍只是 `development_smoke`，不是 formal benchmark；`T22` 已完成真板 smoke execution plan，但真板条件仍未满足。若最终目标是形成可投稿论文，最近一步应先补最关键的软件 benchmark 证据口径，而不是直接写论文路线或启动长跑。

Allowed files:

- `docs/tasks/Phase2/T23_p4_formal_benchmark_protocol_lock.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/protocols/benchmark/P4_benchmark_development_protocol.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Inputs to read:

- `docs/02_experiment_plan.md`
- `docs/reference/进一步的深度研究结果.md`
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`
- `docs/protocols/benchmark/P4_benchmark_development_protocol.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/review/T15_frozen_smoke_review.md`
- `docs/review/T16_p4_evidence_gate_review.md`
- `docs/review/T21_phase2_milestone_review.md`
- `docs/08_risks_and_open_questions.md`
- `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

Expected output:

- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- 必须包含：
  - formal / development / recovery 三种证据等级边界
  - proposed formal matrix: scenarios / modes / repeats / seeds
  - baseline inclusion and exclusion rules
  - paired-seed and statistical reporting rules
  - compute budget and expected run-time risk
  - output evidence pack requirements
  - explicit assessment of deep-research recommendations:
    - strong classical / soft-information / calibration / learned baseline classes
    - static, drift, random-walk, sinusoidal, burst/reset scenario families
    - training-seed vs evaluation-seed separation
    - confidence interval or stopping rule
    - latency / commit / rollback / fallback metrics
    - whether a statcalib baseline must be implemented before formal execution
    - why true `.tflite` runtime is prioritized before real-board smoke for deployment claims
  - exact go/no-go criteria for a later execution task
  - if formal execution is not yet ready, name the prerequisite task type instead of forcing a run
  - explicit statement that no formal benchmark was run in T23

Forbidden scope:

- 不运行 P4 benchmark
- 不运行训练、`.tflite` runtime、cleanup 或硬件命令
- 不改源码
- 不改 benchmark 口径、baseline 集合或 ParamMapper 语义
- 不改历史 `runs/` / `artifacts/` 结果
- 不把 `T15` development run 写成 formal benchmark
- 不新增 teacher-representation 长跑或新模型主线

Verification:

- 只读审计。
- 用 `rg` / `Get-Content` 核对 config、benchmark 入口和已有 review 结论。
- 文档中必须明确写出 `T23 did not run benchmark`。
- 输出必须能让 Captain 决定后续是否开 `T24` execution task。

Docs to update:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Reviewer type: `adversarial`
