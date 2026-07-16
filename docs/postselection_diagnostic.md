# T3.2.4 post-selection 诊断上界与成本核算

## 1. 结论

T3.2.4 只把 post-selection 用作离线诊断：检查 static-MAP posterior confidence 是否携带 logical
failure 信息，并在每个 coverage 点同时报告 survival、rejection、conditional LER、accepted failures
per input、truth-only upper bound、random rejection 和 rejection-penalty cost。它不是在线 decoder，
结果不得进入主纠错增益或 break-even claim。

在 294,912 个 training samples 上校准阈值，并在 4 类 continuous wrapped-Gaussian scenarios、
8 evaluation seeds、1,572,864 samples 上评估。90% 目标 survival 的 aggregate realized survival 为
`0.899108`，raw LER `0.013785`，accepted conditional LER `0.001242`，捕获约 `92.44%` failures。
四场景 score AUROC CI 下界均大于 `0.95`，raw-minus-conditional LER CI 下界均为正，证明 syndrome-
derived posterior confidence 确实含 failure 信息。

但 post-selection 不是免费纠错。90% 点的 aggregate accepted failures/input 为 `0.001106`；加入
rejection penalty `0.25/0.50/1.00` 后总成本为 `0.026329/0.051552/0.101997`，均可远高于 raw
LER。逐场景 break-even rejection penalty 只有约 `0.083--0.142`，均值 `0.1223`。因此只能报告
coverage--risk--cost 前沿，不能只抄 conditional LER。

## 2. 三条严格分离的 lane

### 2.1 Observed-score policy

static periodic MAP 只消费 centered `residual_q/p`，得到四 logical cosets posterior。score 定义为：

```text
risk(observation) = 1 - max_c p(c | observed syndrome)
```

score 函数不接收 truth、displacement、drift state 或 failure label。target survival 对应的 threshold
只由 training scores 的 quantile 冻结，evaluation 不重新调阈值。

### 2.2 Random-rejection reference

相同 realized survival 下，random rejection 的期望 conditional LER 等于 raw LER，accepted
failures/input 等于 `survival * raw LER`。它防止把“样本变少”本身误写成 score 有信息。

### 2.3 Truth-only diagnostic upper bound

在与 observed policy 完全相同的 accepted count 下，truth upper 先拒绝真实 failures，再拒绝 correct
samples。其 accepted failures 为：

```text
max(0, total_failures - rejected_count)
```

truth 只进入 evaluator，不进入 observed score 或 threshold。`99.5%` survival 区域保留非零 upper，
较低 survival 区域进入零-error upper，避免只展示平凡的零值。

## 3. Threshold 与规模

training/evaluation seeds 分别为 `20261011--13` 与 `20261031--38`，严格不相交。training-only
thresholds：

| target survival | score threshold |
| ---: | ---: |
| 0.995 | 0.411640 |
| 0.990 | 0.328197 |
| 0.980 | 0.199261 |
| 0.950 | 0.049318 |
| 0.900 | 0.010223 |
| 0.800 | 0.001549 |
| 0.700 | 0.000442 |
| 0.500 | 0.000069 |

每条 evaluation trace 含 48 x 1024 samples；4 scenarios x 8 seeds x 8 thresholds 形成 256 行
Source Data。阈值随 target survival 严格单调，realized survival 与 training target 偏差均小于 0.10。

## 4. 90% 主诊断点

| scenario | survival | raw LER | conditional LER | failure capture | break-even rejection penalty | cost at penalty 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear mean | 0.923810 | 0.006582 | 0.000278 | 0.9610 | 0.0830 | 0.076447 |
| variance/correlation ramp | 0.920792 | 0.011556 | 0.001262 | 0.8995 | 0.1313 | 0.080371 |
| sinusoidal joint | 0.863592 | 0.019633 | 0.001746 | 0.9231 | 0.1329 | 0.137917 |
| smooth mixed | 0.888240 | 0.017370 | 0.001684 | 0.9139 | 0.1420 | 0.113256 |

truth upper 在 90% 点为零，是因为 rejection fraction 已超过 failure prevalence；它不表示可免费
实现零错误。99.5% 点的 truth-upper conditional LER 分别为
`0.003637/0.007566/0.012210/0.011310`，证明 coverage matching 的 upper frontier 非平凡。

## 5. 成本定义

对 rejection penalty `c`：

```text
total_cost(c) = accepted_failures / total_inputs + c * rejected_inputs / total_inputs
```

`c=0` 是免费拒绝的乐观极限，只作诊断；`c=1` 把每次 rejection 视为一次失败，所有 rows 的成本均
不低于 raw LER。break-even penalty 等于 rejected set 的 failure rate；只有实际 rejection 代价低于
它时，该 coverage 点的总成本才低于 raw failure cost。该 penalty 尚未映射到实验 repeats、time、
active pulses 或 state-preparation cost，T5.3.4 才能做完整物理成本。

## 6. 反简化检查

1. coverage 从 99.5% 到 50% 共 8 点，不只报一个好看的低 conditional LER。
2. 阈值由 294,912 training scores 冻结；evaluation 不参与 threshold selection。
3. 256 行 Source Data 保存 survival/rejection、raw/conditional/unconditional failures、failure capture、
   random reference、truth upper、break-even penalty 与四种 total cost。
4. truth upper 使用相同 realized coverage，不能免费拒绝任意数量样本。
5. 增加 99%/99.5% 高 survival 区，机器门要求 truth upper 同时有非零和零区域。
6. random rejection conditional LER 严格等于 raw；unit rejection cost 不允许伪净收益。
7. Student-t CI 以 8 个 evaluation seeds 为 cluster，不把 157 万 samples 当独立重复。
8. score API 只接收 posterior；shape/normalization/NaN/negative probability、demo training array、非法
   coverage/penalty、空 acceptance 与 AUROC 单类别均 fail closed。

## 7. Claim 边界

允许：offline evidence that static-MAP posterior confidence contains failure information，并完整报告
survival/rejection/penalty cost 与 truth diagnostic upper。

禁止：online correction gain、postselected break-even、free rejection、truth-score deployment、
device-calibrated anomaly detector 或 FPGA synthesis/measurement。post-selected conditional LER 不进入
主算法排名。

## 8. 可复现入口

```powershell
python -m cnn_fpga.benchmark.postselection_diagnostic
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/test_postselection_diagnostic.py tests/test_postselection_diagnostic_benchmark.py
```

机器证据：

- `docs/t3_2_4_postselection_validation.json`
- `docs/t3_2_4_postselection_source_data.csv`
