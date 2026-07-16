# T4.1.4 多目标 loss 与严格分割 calibration

## 1. 任务结论

T4.1.4 已把慢回路训练/评测目标冻结为六项可计算 loss，并建立严格的
`training -> validation -> evaluation` 单向数据流：

1. training 仅估计连续目标和六项 objective 的 robust normalization scale；
2. validation 仅选择 regime posterior temperature/uniform mix、uncertainty scale 和带 unsafe-recall
   约束的 fallback threshold；
3. evaluation 只消费冻结参数，不参与权重、尺度或 threshold 选择；
4. 六项 loss 的 leave-one-objective-out 只作冻结输出的归因账本，不冒充重新训练后的因果消融。

当前证据是 registered synthetic offline calibration，不是物理 risk calibration、在线 student、RTL、
FPGA timing 或 board/device measurement。

## 2. 六项 loss

固定权重为：

| objective | 权重 | 定义 |
| --- | ---: | --- |
| state estimation | 0.24 | 9 个 future observed targets 的 robust-scale smooth-L1 |
| oracle gap | 0.22 | 下一 32 cycles 中 candidate 与 full-state model oracle 的 paired failure-rate gap 绝对值 |
| regime detection | 0.16 | 四态 multiclass NLL + Brier score |
| uncertainty calibration | 0.16 | 9 个 marginal 的 normalized Gaussian log score，并另报 68%/95% coverage |
| false fallback | 0.12 | false-fallback rate + `4 ×` missed-required-fallback rate |
| update cost | 0.10 | 每次 slow evaluation 固定成本、K/b churn 和 atomic stage 增量 |

总 loss 为 training-only objective scale 归一化后的加权和。权重在 evaluation 前固定；当前任务不以
evaluation 重新搜索权重。

## 3. Future target 与 oracle 对齐

输入是 T4.1.3 的 456 个 future-only outputs。每个 seed 的最后一个 output 因没有完整未来 32-cycle
horizon 被丢弃，最终得到 448 条记录。9 个 state target 来自未来 observed residual、g/e/leakage 的
periodic moments/rates，而不是把同一 history 的 estimator 输出当 target。

oracle-gap 只在未来 non-leakage cycles 上评测：candidate 使用当前输出的 mean/covariance，oracle
使用 simulator exact `DriftState`，二者对同一 residual syndrome 和同一 logical increment 配对。truth
只存在于 `offline_future_aligned_calibration_record`，不得进入 deployable history/output。

四态 offline label 使用显式优先级：跨 phase 为 `calibration_shift`；否则持续 leakage、burst、normal。
required fallback 只由持续 higher leakage、高平均 recovery burden 或显著 paired oracle-gap 触发。单个
非零 recovery depth 是正常协议活动，不能把所有样本都标为 unsafe。

## 4. 严格 split 与 calibration

| split | seeds | records | 用途 |
| --- | ---: | ---: | --- |
| training | 3 | 168 | robust state/objective scales |
| validation | 2 | 112 | calibration selection |
| evaluation | 3 | 168 | frozen report only |

三个 seed 集完全不相交，且每个 split 都包含 normal/burst/leakage/calibration-shift。calibration 参数为：

- regime temperature `4.0`；
- uniform posterior mix `0.5`；
- uncertainty scale `4.0`；
- fallback threshold `0.7`，minimum unsafe recall `0.90`。

uniform mix 是对旧 14-summary HMM bridge 过度自信的显式校准，不是重新训练 53-feature model。

## 5. Production validation 结果

8-seed replay 的 19/19 gates 通过，保存 448-row Source Data。validation 上：

- regime NLL 从 `8.563501` 降到 `1.262520`；accuracy 仍为 `0.455357`，说明改善来自概率校准而非
  分类能力提升；
- marginal 68%/95% coverage 从 `0.234127/0.479167` 改善到
  `0.779762/0.947421`；
- safety constraint 达到 required-fallback recall `1.0`。

冻结 evaluation 上 candidate/oracle paired error rate 为 `0.115988/0.101887`，六项 raw loss 为：

| objective | evaluation raw loss |
| --- | ---: |
| state estimation | 0.487799 |
| oracle gap | 0.035015 |
| regime detection | 1.868864 |
| uncertainty calibration | 1.880605 |
| false fallback | 1.000000 |
| update cost | 0.050286 |

归一化加权 total 为 `0.756840`。evaluation uncertainty coverage 为
`0.773148/0.955688`。

## 6. 负结果与反简化审计

初版 unsafe 定义曾使用“未来任一 recovery depth >= 2”，导致 448/448 全为 required fallback；该定义
已撤销，因为 recovery depth 在本协议中是常规活动。另一个初版只把实际 stage 计为 update cost，导致
stress-only evaluation 的 update objective 退化为零；现已加入每次 slow evaluation 的固定成本和 K/b
churn。

最终结果仍保留一个关键负结论：threshold `0.7` 虽达到 evaluation unsafe recall `1.0`，但
false-fallback rate 也是 `1.0`。这说明现有 T4.1.3 risk/OOD score 在安全约束下没有足够选择性，不能写成
“fallback 已校准可部署”。同样，regime calibration 改善 proper score，但没有提升 accuracy。两项缺口已
登记 R-N066，后续必须由 richer-input/student/OOD 任务解决。

## 7. 产物与复现

- `cnn_fpga/decoder/hybrid_multiobjective.py`
- `cnn_fpga/benchmark/hybrid_multiobjective_calibration.py`
- `docs/t4_1_4_hybrid_multiobjective_validation.json`
- `docs/t4_1_4_hybrid_multiobjective_source_data.csv`
- `tests/test_hybrid_multiobjective.py`
- `tests/test_hybrid_multiobjective_calibration.py`

复现：

```powershell
python -m cnn_fpga.benchmark.hybrid_multiobjective_calibration
```

