# 投稿稿 Phase A `linear_ramp` formal chunk 启动检查记录

日期：2026-07-06

## 检查对象

本记录只检查投稿稿 Phase A repeat-expanded benchmark 的 `linear_ramp` formal chunk 是否具备可执行入口。目标缺口是：当前 repeat-expanded interval 证据只完成 `static_bias_theta` 单一场景；投稿稿若要把 Phase A 写成跨场景 repeat-expanded 结果，还需要补齐其余预声明场景的 formal paired repeats、汇总脚本和 paired-interval 分析。

## 当前结论

- `runs/paper_submission_phase_a/formal_linear_ramp_ukf_hybrid_r12_00_04_20260706` 目前只看到空的 stdout/stderr 日志，没有 `summary.json`、`progress.jsonl` 或 repeat status；不能作为任何性能、统计或复现证据。
- `runs/paper_submission_phase_a/formal_linear_ramp_ukf_hybrid_r12_00_04_20260706_codex2` 是一次后台启动尝试残留，只包含 launch metadata 和空日志；不能作为 benchmark 证据。
- 前台启动检查 `runs/paper_submission_phase_a/formal_linear_ramp_startup_check_20260706` 使用 `--repeat-start 0 --repeat-stop 0`，验证了 runner/import/config/filter 入口可以完成，但 `raw_rows=0`、`comparison_rows=0`、`missing_runs=24`；这是零 repeat 的启动检查，不是实验结果。
- 本轮没有产生新的 LER、paired delta、CI、p-value、holdout drift、fixed-point parity、latency/resource 或硬件测量结果，因此不应更新投稿稿主文结果表、图件或统计表述。

## 启动尝试与问题

- `Start-Process` 后台启动在当前 PowerShell 环境中触发 `PATH` / `Path` 环境变量重复键问题，未形成有效 benchmark 输出。
- .NET `ProcessStartInfo` 环境变量清理路径在当前运行时不可用，未形成有效 benchmark 输出。
- `cmd.exe /c` 后台启动返回了进程 id，但目标目录没有 runner 输出、进度或 summary，不能视为执行成功。
- 前台零 repeat 启动检查完成，说明命令入口本身可用；真正的 formal chunk 仍需长时间前台或受控后台执行。

## 后续有效命令

下面命令是下一步可执行的 `linear_ramp` formal chunk 形状；运行前应使用新的 run directory，完成后再运行 Phase A summary 和 paired-interval 汇总脚本。

```powershell
C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml --scenario linear_ramp --mode ukf --mode hybrid_residual_b --repeats 12 --paired-seeds --run-dir runs\paper_submission_phase_a\formal_linear_ramp_ukf_hybrid_r12_00_04_<timestamp> --repeat-start 0 --repeat-stop 4
```

## 可写边界

- 可以写：`linear_ramp` formal repeat chunk 尚未形成可用结果；当前缺口是跨场景 repeat-expanded interval gate 的一部分；零 repeat 启动检查只证明入口可达。
- 不可写：`linear_ramp` 已完成 formal repeats；all-scenario repeat-expanded gate 已通过；Hybrid 在全场景 repeat-expanded setting 下具有统计确认优势；已有硬件、真实 FPGA timing/resource、finite-energy physical-channel fidelity 或 real-board evidence。

## 本轮验证

- 列出 `formal_linear_ramp_ukf_hybrid_r12_00_04_20260706` 与 `formal_linear_ramp_ukf_hybrid_r12_00_04_20260706_codex2` 目录内容，确认无可用 summary/progress 结果。
- 读取 `formal_linear_ramp_startup_check_20260706/summary.json`，确认 `raw_rows=0`、`comparison_rows=0`、`missing_runs=24`。
- 未运行 Phase A summary / interval collectors，避免把零 repeat 启动检查误解为结果更新。
