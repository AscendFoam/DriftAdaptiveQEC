# T5.4.5 训练 horizon 与长时递归外推验证

## 结论

本任务得到 `QUALIFIED_LONG_RECURRENCE_PASS_PHYSICAL_GAIN_NOT_ESTABLISHED`。这表示 frozen GRU-10
teacher 与 4-state students 在已登记的 `g/e` 输入流上真实执行到 `10^6` cycles 后，状态、动作、float32
shadow 和 reset recovery 均通过；它不表示已经执行 `10^6` cycles 的 Fock-space logical channel，也不建立
long-horizon physical gain、physical-memory LER、leakage robustness、device 或 hardware claim。

## 预注册设计

- 训练 horizon：2、5、10、32 cycles；2/5/10 各重新拟合 3 个 4-state restarts，validation-only 选择；
  32-cycle 点复用 T4.4.3 strict-split production student，不用本任务 evaluation 重选。
- 部署 horizon：`10^3/10^5/10^6` cycles，即每条流真实执行 `2,000,000` 次 half-cycle recurrence。
- 输入流：stationary nominal、persistent regime、range shift 各 2 个新 seeds，加 all-g/all-e 两条边界流，
  共 8 条；每种精度执行 16,000,000 teacher updates 和 64,000,000 student updates。
- performance：在每个 horizon 的线性/对数冻结检查点和 reset 窗口计算 teacher-action imitation；共有
  13,631 个状态检查点。dense head 只在检查点运行，但 GRU/student state 没有跳步、重复短序列或解析替代。
- reset：在每个部署终点前 256 half-cycles 强制清空 teacher hidden 或恢复 student initial state，使用相同未来
  outcomes 与 uninterrupted counterfactual 比较，要求连续 8 点恢复且不超过 128 half-cycles。

## 训练 horizon 结果

| 训练 horizon | validation 选择 restart | validation MSE | 独立 32-cycle MSE |
| ---: | ---: | ---: | ---: |
| 2 | 2 | `6.3951e-7` | `9.9540e-5` |
| 5 | 1 | `3.3231e-6` | `1.4585e-5` |
| 10 | 2 | `5.0707e-6` | `9.4858e-6` |
| 32 | 0（T4.4.3 frozen） | `5.6485e-6` | `6.0831e-6` |

2-cycle 模型在自己的短 validation 前缀最好，却在独立 32-cycle evaluation 上超过 `5e-5` 阈值；这是一项
正式负结果，说明短 horizon 的低 validation loss 不能替代外推检查。selection 始终不读取 evaluation。

## 长时动作保持与最坏流

| 模型 | `10^3` mean / worst MSE | `10^5` mean / worst MSE | `10^6` mean / worst MSE |
| --- | ---: | ---: | ---: |
| fresh 2-cycle | `1.2564e-4 / 8.1028e-4` | `1.2426e-4 / 8.1414e-4` | `1.2754e-4 / 8.1423e-4` |
| fresh 5-cycle | `1.1993e-5 / 5.8882e-5` | `1.1708e-5 / 5.9146e-5` | `1.2298e-5 / 5.9152e-5` |
| fresh 10-cycle | `7.4826e-6 / 2.3433e-5` | `7.1217e-6 / 2.3534e-5` | `7.8447e-6 / 2.3536e-5` |
| production 32-cycle | `4.7240e-6 / 7.6121e-6` | `4.4112e-6 / 6.6024e-6` | `5.0431e-6 / 6.3251e-6` |

2/5-cycle 模型的最坏流均为 all-e boundary，且 5-cycle worst 超过阈值；没有用跨流均值隐藏。正式
retention gate 只对与 parent claim 相关的 10/32-cycle models 生效，并同时要求 mean 与 worst stream
`<=5e-5`。

## 状态、精度与 reset

- teacher 全步最大 hidden absolute value 为 `0.382517`，小于 GRU `tanh` 解析界 1；
- 每个 student 的实际全步 state maximum 均不超过 `max(|initial|, |saturation_g|, |saturation_e|)` 解析凸包界；
- 所有采样动作的最大 bound-normalized magnitude 为 `0.266154`，小于 hard action box 1；
- float32/float64 最大动作差为 `1.2630e-6`，低于 `1e-4` 门；
- 120 个 model×stream×horizon reset interventions 全部恢复，最慢为 20 half-cycles；terminal 差值由真实
  counterfactual 计算，最大约为数值舍入量级，不预设为零。

## 非 demo 检查与产物

`tests/test_horizon_extrapolation_validation.py` 验证 GRUCell 与批量 GRU 的逐步等价、Numba scan 与导出
student 公式等价、全步 state bound、worst-stream retention、reset 干预、checkpoint/Source Data hash，并对
缩短 stream、evaluation 选模、破坏 state bound、隐藏 worst stream、伪造 reset recovery 和升级 physical
gain 六类 mutation fail closed。正式产物为：

- `docs/t5_4_5_horizon_extrapolation_validation.json`；
- `docs/t5_4_5_horizon_extrapolation_candidates.pt`；
- `docs/t5_4_5_horizon_extrapolation_validation_source_data.csv`（521 rows）；
- `cnn_fpga/benchmark/horizon_extrapolation_validation.py`。

21/21 machine gates 与 24 项 direct/artifact/mutation tests 通过。下一步 T5.4.6 必须独立随机化 gate bias、
readout confusion、leakage/reset、dephasing、drift 与 timing/dynamics；本任务的 observed `g/e` recurrence
稳定性不能替代 model-mismatch robustness。
