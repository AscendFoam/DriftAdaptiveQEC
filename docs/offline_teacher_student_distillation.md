# T4.1.5 离线 teacher / 在线 student 分离

## 1. 任务结论

T4.1.5 已把 T2.3.7 的 5-agent 冻结 NMF policy ensemble 限定为离线 teacher，并将其在
`g/e` synthetic trajectories 上的 15 维 raw control residual 蒸馏为低维指数递推 student。在线模块只接收
`cycle_index`、observed outcome 与 validity/CRC/parameter-fresh/deadline health flags；不导入 PyTorch、
`physics` simulator 或 teacher object。任何 leakage、无效、CRC 错误、参数 stale 或 deadline miss 都强制
输出零 raw residual，从而回到 nominal sBs 安全参数，而不是输出零物理控制。

当前结果证明的是 frozen teacher action 的离线模仿，不是 teacher 的物理 lifetime/fidelity/control gain
已被 student 保留。该后续结论必须由 T4.4.4 在相同物理轨迹上同时比较 standard、MF、teacher、递推
student 与 control oracle 后才能建立。

## 2. Teacher provenance 与数据隔离

teacher 恢复自 `docs/t2_3_7_nmf_directional_ranking_checkpoints.pt`，并以 T2.3.7 manifest 和 5 个 model
SHA-256 逐一核验。teacher 可以使用 model-aware simulator 训练所得权重，但其 checkpoint、module 和
state dict 都不进入在线 artifact。

| split | seed | trajectories | half-cycles | Source Data rows | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| training | 20261501 | 256 | 20 | 5,120 | 拟合 75 个 student trainables |
| validation | 20261502 | 256 | 20 | 5,120 | 选 restart/checkpoint、检查 plateau |
| evaluation | 20261503 | 256 | 20 | 5,120 | 冻结评测，不参与训练或选择 |

三个 seed 和数据 hash 完全分离。student artifact 绑定 training/validation hash，但有意不写入 evaluation
hash，避免把 evaluation 伪装成可部署参数来源。15,360 行 CSV 保存每个 prefix 的 observed outcome、
teacher/student/latest-only/zero-safe residual 和 split provenance。

## 3. Student 结构与训练

student 对 15 个 control residual 分量分别保存 initial state，以及 `g/e` 两类 outcome 的 saturation 和
decay：

\[
r_{t+1,j}=a_{m,j}r_{t,j}+(1-a_{m,j})r^{\infty}_{m,j},\qquad m\in\{g,e\}.
\]

因此共有 `15 + 2x15 + 2x15 = 75` 个可训练参数。artifact 另保存未训练的 leakage safe saturation/decay，
总计 105 scalars；在线状态为 15 scalars。健康 step 精确资源 proxy 为 15 multiplications、30 additions、
21 comparisons，float32 参数存储为 420 bytes。latency cycles、RTL 和 board 字段保持 `null/false`，这些
数值不是综合或实板证据。

训练使用 3 个 restart、最多 1200 epochs、600 epoch 后降低学习率，仅由 validation MSE 选择 checkpoint。
三个 restart 的最佳 epoch 为 `1040/1180/1180`，validation MSE 分别为
`1.322464e-6/1.322565e-6/1.323419e-6`；最后一个验证窗口的相对改善为
`0.000125/0.000254/0.000491`，均低于预注册 plateau tolerance `0.005`。所有 75 trainables 都收到梯度。
由于最佳点接近上限，只能称达到当前 tolerance 下的 validation plateau，不能称 optimizer/global optimum。

## 4. 冻结 evaluation 结果

| comparator | evaluation MSE | RMSE | imitation gain retention vs zero-safe |
| --- | ---: | ---: | ---: |
| distilled recurrence student | `1.453624e-6` | `0.00120566` | `0.999724` |
| latest-only | `1.404389e-4` | `0.0118507` | `0.973328` |
| zero-residual safe baseline | `5.265504e-3` | `0.0725638` | reference |

这里的 “imitation gain retention” 只定义为 teacher residual MSE 相对 zero-safe MSE 的缩小比例；它不等于
逻辑寿命、state fidelity、`p(g)` 或控制收益保留。evaluation 上 teacher/student 相同 prefix 的最大不一致
均为零，online step 与 offline batch bit-exact，重复在线 replay 确定性一致。

## 5. 安全路径与在线边界

- 正常路径只消费 observed `g/e` 与 health flags；online API 显式拒绝 teacher model object。
- leakage、invalid、CRC failure、stale parameters 或 deadline miss 立即进入零 residual baseline。
- artifact 由 canonical payload hash 绑定，不含 `state_dict`、teacher object 或 evaluation hash。
- host 慢状态与 parameter bank 仍由 T4.1.3 合同管理；本 student 不绕过 atomic update，也不把 teacher 放入
  fast path。
- leakage 分支是 fail-closed 的显式安全行为，但当前 two-level teacher 数据没有训练 leakage control。

## 6. 反简化审计与 claim 边界

初版 600 epochs 的三个 restart 全部在训练上限选中，无法区分“趋于平台”与“只是提前停止”。生产版本将
上限扩到 1200，并保存完整 validation tail、best epoch 和 plateau gate；仍保留最佳点接近上限的限制。
测试还覆盖数据 split/hash、5-agent checkpoint provenance、75 参数梯度、same-prefix causality、batch/
online exact replay、health/leakage fallback、teacher-object rejection、在线模块依赖扫描和 artifact tamper。

允许表述：5-agent frozen offline teacher 已被蒸馏为 deterministic 105-scalar observed-outcome recurrence
candidate，并有显式 zero-residual safety fallback。

禁止表述：teacher 进入在线 runtime、物理/lifetime/control gain 已保留、student 已学习 leakage、fixed-point
等价、RTL/FPGA timing、综合或实板测量。

## 7. 产物与复现

- `cnn_fpga/control/teacher_student.py`
- `cnn_fpga/benchmark/offline_teacher_student_distillation.py`
- `docs/t4_1_5_teacher_student_validation.json`
- `docs/t4_1_5_teacher_student_source_data.csv`
- `docs/t4_1_5_distilled_student_checkpoint.json`
- `tests/test_teacher_student.py`
- `tests/test_offline_teacher_student_distillation.py`

生产 artifact 需在安装 PyTorch/CUDA 的 DLEnv 中复现：

```powershell
C:\ProgramData\anaconda3\envs\DLEnv\python.exe -m cnn_fpga.benchmark.offline_teacher_student_distillation
```
