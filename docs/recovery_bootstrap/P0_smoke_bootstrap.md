# P0 Smoke Bootstrap

## 1. 目的

本文件只服务恢复期最小验证闭环，不负责完整训练、HIL 或 P4 运行说明。

目标是让后续会话不用重新猜环境，就能复用当前已验证通过的 P0 smoke 路径。

## 2. 先装哪份依赖

恢复期当前推荐先安装根目录的：

- `requirements-recovery.txt`

安装命令：

```powershell
python -m pip install -r requirements-recovery.txt
```

它当前只覆盖：

1. `benchmark/compare_full_vs_simplified_ler.py --no-plot`
2. `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
3. `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml ...`

它当前不覆盖：

1. `DLEnv` 训练链
2. `.tflite` runtime / export
3. `real_board` HIL backend
4. 去掉 `--no-plot` 后所需的 `matplotlib`

## 3. 当前推荐解释器分工

### 最小 smoke 解释器

- `C:\ProgramData\anaconda3\python.exe`

用途：

- P0 最小 benchmark
- 轻量脚本与结果读取
- 恢复期最小验证

原因：

- 已确认具备 `numpy + yaml`
- 已成功跑通当前最小 P0 smoke

### 训练候选解释器

- `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`

用途：

- 后续 torch 训练
- 更重的模型实验

备注：

- 这是 legacy 开发常用环境
- 当前也已确认具备 `numpy + yaml + torch`
- 但恢复期最小 smoke 不依赖它

## 4. 当前已验证的最小 smoke 命令

```powershell
& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda
```

## 5. 预期输出

运行成功后应得到：

- `runs/smoke_test_anaconda/n10_r2_s0.250_ler_curve_compare.csv`
- `runs/smoke_test_anaconda/n10_r2_s0.250_summary.json`

当前已验证的关键结果为：

- `full_final_ler_mean = 0.150000`
- `simplified_final_ler_mean = 0.000000`
- `final_gap_mean = 0.150000`

## 6. 复核命令

### 读 summary

```powershell
Get-Content -Raw -Encoding UTF8 "runs/smoke_test_anaconda/n10_r2_s0.250_summary.json"
```

### 读 CSV

```powershell
Get-Content -Raw -Encoding UTF8 "runs/smoke_test_anaconda/n10_r2_s0.250_ler_curve_compare.csv"
```

## 7. 当前不处理的内容

本文件不负责：

1. `run_hil_suite.py` 的当前可运行性
2. `run_p4_multiscenario_benchmark.py` 的当前可运行性
3. `.tflite` 独立环境恢复
4. `DLEnv` 训练性能与稳定性结论

这些内容应在 `T3+` 后续任务中继续处理。
