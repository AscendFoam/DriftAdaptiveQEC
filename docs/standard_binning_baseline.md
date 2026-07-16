# T3.1.1 standard-binning baseline integration

## 1. 结论

`standard_binning` 现已作为显式、无调参、可部署的 decoder 行进入当前主要逻辑解码比较，
并由注册表约束未来 T5 主比较不得遗漏。这里没有把“已有一个函数”算作完成：实现同时绑定
decision API、paired evaluator、comparison schema、逐窗账目、报告字段和 provenance artifact。

旧 P4 的 `static_linear` 是 slow-loop parameter-estimator mode，不是 fixed half-cell GKP
recovery；冻结历史结果没有被改名或重解释。

## 2. 判决合同

对每个轴写成

\[
x = k\lambda + s,\qquad s\in[-\lambda/2,\lambda/2),\qquad
\lambda=\sqrt{2\pi}.
\]

standard recovery 只读取 centered syndrome `s`，施加 `-s`，所以 decoder action 固定选择
中心 `even-even` logical coset。`k mod 2` 是 evaluator 的 hidden truth，只用于计分，绝不进入
decision signature。正边界 `+lambda/2` 归下一 cell，负边界 `-lambda/2` 留在中心 cell，严格
复用 `physics.ideal_gkp_decoder.standard_binning_1d` 的半开区间约定。

冻结 descriptor：

| 字段 | 值 |
| --- | --- |
| baseline ID | `standard_binning` |
| observed inputs | `centered_modular_syndrome_q/p` |
| hidden truth inputs | 空 |
| tunable parameters | 空 |
| decision | `logical_class=0`，fixed half-cell recovery |
| scope | ideal square-GKP syndrome-level |

## 3. 主要比较注册表

| Comparison | 类型 / 生命周期 | standard policy | 处理 |
| --- | --- | --- | --- |
| `t1_3_4_adaptive_drift_alignment` | decoder comparison / active | required | 已接入五行同 trace 比较 |
| `phase5_main_decoder_benchmark` | decoder comparison / future contract | required | schema fail closed，未来主表必须保留 |
| T2.4.3 precision sensitivity | implementation sensitivity / active | N/A | 比较同一 MAP 的位宽，不是算法排名 |
| T3.1.4 sBs branch comparison | decoder comparison / active | N/A | 独立 branch target，static/reference 角色显式登记 |
| T3.1.5 top-K sensitivity | implementation sensitivity / active | N/A | 同一 periodic MAP 的 K 截断，不是算法排名 |
| T3.2.1 memory Bayes | decoder comparison / active | required | standard + task-specific final-static + episode truth reference |
| T3.2.2 continuous adaptive MAP | decoder comparison / active | required | standard + training-average static + latest-window/EWMA/Kalman + full-state oracle |
| T3.2.3 sliding window | decoder comparison / active | required | standard + training-average static + latest-window + training-selected sliding-window + full-state oracle |
| T2.4.2 timing-fault sensitivity | timing sensitivity / active | N/A | 比较故障 schedule，不是 decoder 排名 |
| legacy P4 software-HIL | parameter-estimator / frozen legacy | N/A | 保持 `static_linear` 原语义，不冒充 standard |

注册表 validator 要求每个 required decoder comparison 恰好出现一次
`standard_binning`；漏项、重复、把 sensitivity 表伪装成 decoder comparison 或重复 comparison
ID 都会失败。

T3.2.1 集成后，所有 decoder comparison 还必须各自显式声明且恰好包含一个
`static_anchor_method_id` 和一个 `reference_anchor_method_id`。这两个角色是 task-specific：
T1.3.4/T5 使用 training-average static/full-state oracle，T3.2.1 使用 final-outcome static
Bayes/full-episode truth，T3.2.2 则重新使用 training-average static/full-state oracle，但增加三个
因果 adaptive rows；T3.2.3 保持相同 anchors，但把 latest 384 与 training-selected uniform window
并列，当前选择结果允许两者数值相同。此约束防止旧 baseline validator 把语义不同的新 comparison
强绑到同一 static/oracle 方法。

## 4. 同 trace 结果与反证

复用 T1.3.4 的 24-window、72,000-sample materialized evaluation trace；五种方法读取完全相同
的 syndrome/truth buffer，trace SHA-256 为
`975a606105be9ba2b28f81466367d38142e941515276f0556f7dde1aae770439`。

| Decoder | 总 LER | before LER | after LER |
| --- | ---: | ---: | ---: |
| Standard binning | 0.060417 | 0.005417 | 0.087917 |
| Static training-average MAP（T3.1.2 current） | 0.024792 | 0.005417 | 0.034479 |
| Window Variance + MAP | 0.022639 | 0.005375 | 0.031271 |
| EKF + MAP | 0.025319 | 0.005458 | 0.035250 |
| Full-state model oracle MAP | 0.011389 | 0.005417 | 0.014375 |

T3.1.2 current standard-minus-static 为 `+0.035625 [0.033985,0.037265]`，formal static
稳定优于 standard。T3.1.1 当时的旧 static-calibration row 为 `0.061389`，standard 相对它为
`-0.000972 [-0.001280,-0.000664]`；这项历史反证触发了 T3.1.2 的 fair baseline 修复，未被删除。
完整 current static 证据见 `docs/static_map_baseline.md`。

## 5. 反简化与验证

- decision API 不接受 displacement、cell index、drift state、noise parameter、history 或 oracle；
- 同 centered syndrome、相差奇/偶 lattice cells 的样本得到相同 action，但 evaluator truth 不同；
- `-lambda/2` 与 `+lambda/2` 边界、非法 shape、非有限值、越界 syndrome 和非法 lattice 均有负测；
- adaptive comparison 保存每窗 standard failure count，并与全局 rate 精确对账；
- production JSON 绑定 baseline、alignment 与 reference decoder 三份源码 hash；
- 10/10 machine gates PASS；focused + adjacent `92 passed`。

机器产物：`docs/t3_1_1_standard_binning_validation.json`。

## 6. Claim 边界

允许：standard binning 已作为显式 no-tuning row 接入当前 synthetic ideal square-GKP decoder
comparison；当前注册 step trace 上它略优于旧 static-calibration MAP。

禁止：把 `static_linear` 写成 standard binning；把当前差异外推为 MAP 普遍无效；把该结果写成
finite-energy/protocol-aware optimal recovery、FPGA 实测或量子硬件结果。
