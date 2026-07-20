# T5.2.3 leakage 与 reset failure 独立因果注入

**日期：** 2026-07-16  
**状态：** PASS（protocol-native effective simulation）  
**实现：** `cnn_fpga/benchmark/leakage_reset_causal.py`  
**机器证据：** `docs/t5_2_3_leakage_reset_causal.json`  
**Source Data：** `docs/t5_2_3_leakage_reset_source_data.csv`

## 1. 为什么不能直接复用旧结果

T2.0.6/T5.1.2 已有 leakage occupancy、long-lag correlation 和离线去除长 leakage trajectory 后的 tail shrink，
但它们是 component/post-selection diagnostics：没有 online detection delay、healthy-state false alarm、reset-failure-only
intervention、action availability 或 raw reset recovery cost。因此本任务没有把旧 component PASS 改名，而是复用
T2.0.3 的 full hidden `g/e/f/higher` -> observed `g/e/leakage` -> conditional-reset kernel，新增两条互斥 causal lanes：

1. `higher_leakage_injection`：只改变 g/e 下进入 `higher` 的 injection probability；
2. `higher_reset_failure`：只改变 observed leakage 后 `higher -> g` reset 的 failure probability。

## 2. 正式设计与观测边界

每条 lane 有 6 个 rate、8 个新 seed clusters、每 seed 256 条 trajectory、128-cycle burn-in 和 512-cycle
evaluation；总计 96 个 family×seed×rate cells。一个 cycle 保留 X/Z 两个 constituent step，ideal g/e 按
`K_gg/K_ge/K_eg/K_ee` 平衡。family/seed 内跨 rate 使用 common random numbers，CI 以 whole-seed cluster
做 20,000-repeat bootstrap。

冻结 classifier：healthy g/e 的 leakage false alarm probability 为 `2e-4`，hidden f/higher 的 leakage detection
probability 为 `0.95`。这些是 sensitivity assumptions，不是装置标定。leakage family 固定 higher reset failure
为 `0.9`，reset family 固定 higher injection 为 `0.002`；机器门会从实际 episode/opportunity 和
failure/attempt 反算这两个概率，防止固定通道暗漂移。

deployable lane 只保留 observed X/Z、conditional reset action 和 observed leakage run。hidden state 只用于：

- injection onset 到首次 observed leakage 的 detection delay；
- hidden g/e 时出现 leakage observation 的 false alarm；
- hidden higher 时未报 leakage 的 false negative；
- truth-scored safe availability 与 persistence。

## 3. Estimand 定义

- detection delay：从 hidden higher injection onset 到首次 observed leakage 的 constituent steps；1 step 是半个 X/Z cycle；
- false alarm：hidden pre-readout 为 g/e、observed 为 leakage 的 conditional rate；
- correlation tail：observed leakage 在 lags `1/2/4/8/16/32` steps 的 pooled correlation/covariance，不删除 trajectory；
- declared availability：observed 未报 leakage；safe availability：hidden healthy 且 observed 未报 leakage；
- recovery cost：每 1,000 full cycles 的 reset requests、true attempts、successes 和 failures，全部分列；
- 不生成 combined availability-cost score，不计算 physical-memory LER。

## 4. Leakage-free 消融

`higher_leakage_injection, rate=0` 是显式 matched ablation：

| 指标 | 结果，95% seed-cluster CI |
| --- | ---: |
| hidden leakage occupancy | `0 [0,0]` |
| true reset attempts / 1,000 cycles | `0` |
| detection probability / delay | `null / null`，无 true episode，不填伪 0 |
| false alarm / healthy step | `2.136e-4` |
| reset requests / 1,000 cycles | `0.427`，来自 false alarms |
| safe normal-action availability | `0.999786 [0.999769,0.999800]` |
| long-lag correlation | `0.000504 [-0.000226,0.001943]`，noise floor |

这个 control 证明 detector/cost 不是靠 truth 直接触发；即使没有 hidden leakage，imperfect observed classifier
仍会产生非零误报和恢复请求。

## 5. 关键因果结果

### 5.1 Leakage injection endpoint

在 injection rate `0.004`：

- empirical hazard `0.004013 [0.003941,0.004094]`；
- hidden occupancy `0.042331 [0.041214,0.043521]`；
- mean hidden run `10.948 [10.690,11.172]` steps；
- reset attempts `80.401 / 1,000 cycles`；
- safe availability `0.957461 [0.956252,0.958577]`；
- short/long-lag correlation `0.749871/0.224123`，long-lag covariance `0.008703`。

occupancy、reset attempts/failures 全网格严格增加，safe availability 严格下降；与 leakage-free control 相比，
long-lag covariance 的增加超过冻结门。

### 5.2 Reset-failure endpoint

固定 injection `0.002` 时，reset failure 从 `0` 增到 `0.95`：

| 指标 | reset failure 0 | reset failure 0.95 |
| --- | ---: | ---: |
| empirical injection | `0.002069` | `0.002074` |
| empirical reset failure | `0` | `0.949281` |
| hidden occupancy | `0.002173` | `0.041299` |
| mean hidden run | `1.055` | `20.471` steps |
| true reset attempts / 1,000 cycles | `4.127` | `78.463` |
| safe availability | `0.997639` | `0.958522` |
| long-lag correlation | `-0.000384` | `0.404017` |

occupancy、run length、reset failures 严格增加，availability 严格下降。`0 -> 0.25` 的 long-lag point estimate
仍在约 `5e-4` noise floor，故没有伪称六点 correlation 全部严格单调；正式门要求 endpoint 增幅和从 0.25 后
严格上升。

## 6. Detection 与误报结果

所有有 true episode 的 formal cells 中 observed detection fraction 为 1；这只是本次有限 sample 的观测值，不是
population guarantee。mean delay 约 `0.053--0.055` constituent steps，p95 不超过 1 step；false-negative rate
保持在冻结 0.05 classifier assumption 附近，false-alarm rate在所有 interventions 中保持 `5e-5--4e-4` 门内。
truth 只用于评分这些量，不进入 reset decision。

## 7. 反简化验证

23/23 machine gates 和 27 项 direct/mutation tests 覆盖：

- 96 cells、12 summaries、paired stream、trace/source/parent/code/CSV hashes；
- 拒绝删 family/rate/seed、降 trajectory/horizon/bootstrap 或把 classifier 改为完美；
- empirical injection/reset rate calibration、固定通道不漂移；
- leakage-free null/false-alarm semantics、detection denominator、全因果方向；
- coherent rewrite、双通道 mutation、truth leakage、fake-zero null、global score 和 postselected-primary field；
- 2,508-row ledger 的 raw metrics 与 whole-seed uncertainty。

相邻 T2.0.3/T2.0.6/T5.1.2/T5.1.6/T5.2.2 回归为 `145 passed`。

## 8. Claim boundary

允许声称：两条 effective interventions 已独立执行；online observed detection、false alarm、correlation tail、
availability、raw reset cost 与 leakage-free ablation 已完整报告。

禁止声称：classifier/injection/reset rates 是装置标定；detection fraction 1 是零漏检保证；availability 是板卡 uptime；
correlation 是 coherent physical-memory tail；本任务建立了 logical channel、break-even、QPU/FPGA 实测或 device
reset fidelity。T5.3/T5.4/T6 才能升级相应证据。
