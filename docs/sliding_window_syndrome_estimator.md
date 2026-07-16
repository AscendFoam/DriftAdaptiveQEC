# T3.2.3 sliding-window syndrome estimator

## 1. 结论

T3.2.3 实现了 causal uniform sliding-window periodic syndrome estimator，并在完全相同的
`384 new observations / 1 update / window` 预算下扫描 `384、480、576、768、1152、1536`
六个 history lengths。窗口长度只由 3 个 training seeds 上的 observation-only next-window
periodic-moment score 选择；8 个 evaluation seeds 不参与选择。

训练选择 `384`，evaluation aggregate 的事后最低 LER 也为 `384`。因此本任务的主要结论是：
在当前 4 类连续 wrapped-Gaussian 场景混合上，增加 uniform history 没有产生可推广的全局收益；
T3.2.2 的 latest-window 行就是当前 training-selected sliding-window baseline。不能把局部场景中较长
窗口的较低 point estimate 写成可部署的 scenario-oracle selector。

选择后的 384 窗相对 formal static MAP 的 aggregate LER gain 为
`0.009740 [0.009529,0.009951]`，Student-t 95% CI 以 evaluation seed 为 cluster；四场景 CI
下界均正，full-state oracle 仍严格更低。由于 selected 与 latest 是同一 384 窗，二者差和 CI
严格为 `0 [0,0]`，没有虚构 sliding-window 对 latest-window 的新优势。

## 2. 与 T3.2.2 的语义关系

T3.2.2 已有 `latest_window_periodic_moment_map`，每次用上一批 384 residuals。若 T3.2.3 再写一个
固定 384 批量估计器，只是重复实现。这里新增的是可审计的 uniform history-length family：每次仍只
接收 384 个新 residuals、更新一次，但可保留上一个或多个窗口的圆特征。384 候选精确退化为
T3.2.2 latest-window；大于 384 的候选形成重叠滑窗。

## 3. 增量充分统计量

每 96 个 residuals 计算 `q、p、q+p、q-p` 四个 joint circular feature sums。窗口更新时：

1. 加入当前 384 observations 的 4 个新 feature chunks；
2. 从双端队列移除超出 history length 的旧 chunks；
3. 用累计 feature sum / retained samples 恢复周期均值和完整 `2 x 2` covariance；
4. 当前 evaluation 必须先完成，当前 observation 后加入，保持 one-window delay。

实现不保存 raw syndrome history。384 到 1536 候选分别只保存 `20、24、28、36、52、68`
个 complex values（含 4 个 accumulator）；每 observation 的代理始终为 2 个 complex
exponentials 和 2 个 complex products。长窗增加 storage，不增加 observation/update bandwidth，
也不把每次更新退化成对全部原始历史重算。

## 4. 训练选择与公平性

- training seeds：`20260911--20260913`；evaluation seeds：`20260931--20260938`；
- 4 scenarios x 48 windows；每场景先用 state-0 的 1536 residuals 填满所有候选；
- 每个 evaluation window：1024 paired evaluation samples、384 independent observations；
- 六候选、standard、formal static 和 oracle 共用同一 evaluation trace；
- 训练 score 使用独立的 next-window observation buffer，不读 state/truth/LER；
- 每个候选均接收全部 observations、每窗更新一次，唯一变量是 retained history length；
- Source Data 按 4 scenario x 8 seeds 保存 32 条 unique trace hashes。

训练 score 随窗口为：

| window samples | training score |
| ---: | ---: |
| 384 | 0.162924 |
| 480 | 0.169436 |
| 576 | 0.169750 |
| 768 | 0.173736 |
| 1152 | 0.223002 |
| 1536 | 0.288516 |

最小值在 384 下边界，这不是“搜索充分的内部 optimum”；它是长 uniform history 在当前 mixture
上被证否的边界结果，故报告中保留完整网格和边界状态。

## 5. Evaluation 结果

aggregate candidate LER：

| window samples | aggregate LER |
| ---: | ---: |
| 384 | 0.004123 |
| 480 | 0.004150 |
| 576 | 0.004147 |
| 768 | 0.004166 |
| 1152 | 0.004377 |
| 1536 | 0.004668 |

逐场景诊断表如下。`diagnostic best` 是 evaluation 后分析，不进入 selector：

| scenario | static | selected 384 | oracle | diagnostic best |
| --- | ---: | ---: | ---: | --- |
| linear mean | 0.006635 | 0.000923 | 0.000852 | 480: 0.000908 |
| variance/correlation ramp | 0.012121 | 0.004423 | 0.004166 | 1152: 0.004300 |
| sinusoidal joint | 0.019447 | 0.004336 | 0.003700 | 384: 0.004336 |
| smooth mixed | 0.017250 | 0.006811 | 0.006378 | 576: 0.006762 |

该表说明 bias/variance trade-off 确实存在，但不同场景的事后最佳窗口不同。由于在线方法没有
scenario truth，不能取每行最小值组成“新 baseline”；后续 HMM/change-point 或 learned estimator
如需动态调 history，必须以相同 observed inputs、384/1 budget 和独立 selection protocol 比较。

## 6. 反简化检查

1. 保留 6 个候选而不是只做 384 demo；至少 5 个候选形成真实 overlapping history。
2. 逐更新把增量 feature estimator 与 raw concatenated batch recomputation 对齐到 `4e-16` 量级；
   384 候选与 latest batch estimator 精确对齐。
3. 直接检查对象不保存二维 raw residual array，只保存 4-complex feature chunks。
4. 非整数/重复/跳号 window ID、错误 stride、短 calibration、NaN、越界 residual、无序/重复/少于
   4 个候选全部 fail closed。
5. online `update` signature 只有 `residuals/window_id`，没有 truth、state、drift 或 displacement。
6. evaluation 后最佳只标 `diagnostic_only`，不反向覆盖 training selection。
7. 置信区间以 8 个 evaluation seeds 聚类；不把 1536 windows 或 157 万样本伪装为独立重复。
8. 新 comparison 进入 major registry 后，T3.1.1--T3.2.2 七份 source-bound artifacts 全规模重生成。

## 7. Claim 与硬件边界

允许：在当前 registered continuous wrapped-Gaussian scenario mixture 与 384/1 budget 上，
training-selected uniform sliding window 退化为 latest 384；较长 uniform history 没有 aggregate gain，
但存在 scenario-specific bias/variance trade-off。

禁止：universal optimal window、evaluation-tuned selector、CNN superiority、loss/outlier/leakage 识别、
finite-energy protocol fidelity、device calibration 或 FPGA synthesis/measurement。所有
`LUT/FF/BRAM/DSP/Fmax` 字段为 `null`，本任务只提供 incremental storage/operation proxy。

## 8. 可复现入口

```powershell
python -m cnn_fpga.benchmark.sliding_window_syndrome_estimator
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/test_sliding_window_syndrome.py tests/test_sliding_window_syndrome_estimator.py
```

机器证据：

- `docs/t3_2_3_sliding_window_validation.json`
- `docs/t3_2_3_sliding_window_source_data.csv`

