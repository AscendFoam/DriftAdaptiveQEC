# T5.4.6 随机模型失配验证

## 结论

正式 campaign 通过 `19/19` 个机器门。此前冻结的 4-state student 在本任务预注册的随机失配分布下保持
qualified branch；判决输出为 `qualified_student_retention`。该结论只表示 student 相对同 cell teacher 的
短时物理控制增益保持，不表示 teacher/student 对 nominal 性能无退化，更不表示装置或长时物理 memory 稳健。

## 设计

四条证据 lane 保持独立，不生成跨 lane 总分或排名：

1. **finite-cutoff physical control**：cutoff 12、10 cycles、batch 16、float64；每个 cell 对
   standard/teacher/student 执行 nominal 与 mismatch 配对轨迹；
2. **readout confusion**：8 个随机 4×3 stochastic matrices；协议轨迹检查 g/e 后果，独立分类采样检查
   g/e/f/higher 四行均被真实访问并校准；
3. **leakage/reset**：8 个随机 higher-leakage injection rates 与 8 个随机 reset-failure rates，复用
   persistent hidden-state kernel，并保存校准试验分母；
4. **drift/dynamics**：8 个 parent-disjoint random parameter vectors，在 chirped sinusoid、stateful random
   telegraph 与 ramp-burst 三类新动力学上只评估冻结的 standard/static/window/EWMA/Kalman/oracle。

物理 lane 共 32 cells：

- 8 个 full 15-dimensional gate-bias vectors；
- 8 个 cavity phase-diffusion cells，`T_phi` 从 `[80,1000] us` 对数均匀采样；
- 8 个保持 half-cycle 总时长 `5000 ns` 的 phase-allocation 与 lifetime/T1/T2 cells；
- 8 个同时施加上述扰动的 compound cells。

全部 64 个 mismatch vectors 与 parents 的 1,351 个已登记 seeds 无重叠。正式输出含 96 个物理
strategy-cell rows、32 个 retention rows 与 273-row Source Data。

## 分支判决

阈值在 evaluation 前固定：

- qualifying teacher gain fraction `>=0.75`；
- student/teacher gain-retention median `>=0.80`；
- retention Q1 与 compound median `>=0.50`；
- teacher/student absolute score-gap p95 `<=0.05`；
- qualifying teacher gain `>1e-4`。

观测为 32/32 qualifying cells、positive fraction `1.0`、retention median/Q1/minimum
`0.998101/0.990413/0.897630`、compound median `0.995598`、absolute-gap p95 `0.003185`。验证器从
原始三策略 mismatch scores 重新构造 retention rows、summary 和 branch；stored booleans、阈值或 fallback
字段不能自证通过。

## 负结果与边界

保留显著 absolute degradation。比如 gate-bias family 的 teacher worst nominal-minus-mismatch score
degradation 为 `0.424155`，compound family 为 `0.395654`；因此“student 跟随 teacher”不能改写成
“controller 对失配不敏感”。随机分布是项目 sensitivity distribution，不是设备校准 posterior。

readout 的 f/higher 行由独立 full-matrix categorical audit 检查，而不是宣称 multilevel master-equation
trajectory；leakage/reset 是 effective persistent kernel；drift lane 是 syndrome-decision benchmark，不能与
finite-Fock control score拼接。physical-memory LER、long-horizon physical gain、device calibration、RTL、
FPGA/board 与实验 claim 全部保持关闭。

## 复现

```powershell
$env:PYTHONPATH='.'
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' `
  -m cnn_fpga.benchmark.randomized_model_mismatch --device cuda
```

产物：

- `docs/t5_4_6_randomized_model_mismatch.json`
- `docs/t5_4_6_randomized_model_mismatch_source_data.csv`
- `tests/test_randomized_model_mismatch.py`

