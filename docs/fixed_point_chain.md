# T2.4.3 ADC/LUT/LLR/state/parameter-bank 位级模型

## 结论与范围

T2.4.3 建立了 observed-only、one-window delayed 的 periodic-Gaussian LLR decoder 位级链：

`centered syndrome -> ADC/replay code -> LUT address -> signed LLR word -> confidence threshold -> parity decision`

慢状态使用 circular mean/wrapped-normal sigma estimator，只读取当前观测 window；估计的 mean/sigma、
threshold 与 LLR table 组成 double-buffered integer bank image。正式扫描包含 42 个 precision profiles
和 4 类 bank faults，使用 8 个 paired seeds、每组 65,536 samples，共 368 个 paired runs。

这是一条 `paired_synthetic_bit_accurate_lut_decoder_not_synthesis_or_board` 软件证据链。资源字段只
是可复算的 representation bit/word-width/update-payload proxy；`FPGA LUT/BRAM/DSP/Fmax` 全部保持
`null`，不能解释为 synthesis、place-and-route 或实板数据。

## 组件与整数语义

### ADC/replay input

`UniformCodeFormat` 在 `[-lambda/2, lambda/2)` 使用 `2^B` 个 level-centre code。上下边界、非有限
输入、非法 code 都 fail closed；每个 256-sample replay window 的 payload proxy 为 `256*B` bits。

### LUT、LLR 与 threshold

- LUT address bits 扫描 `3,4,5,6,7,8`，对应 `8--256` entries；
- LLR 使用 signed `Q3.F`，`F=0,1,2,4,6,8`；
- confidence threshold 使用 signed `Q0.F`，`F=1,2,3,5,7,9`；
- `FixedPointFormat` 现在暴露 word width、signed min/max code、ties-to-even encode、decode、
  saturation 和非有限值拒绝；现有 Q4.20 仍保持 25-bit（1 sign + 4 integer + 20 fraction）语义。

### State estimate 与 update granularity

- mean/sigma 各使用 `3,4,6,8,10,12` bit bounded code；
- update period 扫描 `1,2,4,8,16,32` windows；
- update payload 精确计算为 `entries*LLR_word + 2*state_word + threshold_word`；
- double-bank storage 精确计算为上述 LUT/state/threshold 的两份 image；不做 BRAM packing 假设。

### Parameter-bank faults

每四次 update 注入一次预声明故障：

| fault | 位级行为 | 版本/提交语义 |
| --- | --- | --- |
| `lut_sign_burst` | 随机连续 1/8 LUT entries 翻转 signed sign bit | corrupted image atomic commit |
| `state_msb_flip` | mean code MSB 翻转，再按 corrupted state 重建 LUT | corrupted image atomic commit |
| `stale_commit` | 丢弃该次 staged image | version/commit 均不增长 |
| `torn_update` | new image 的奇地址替换为 old image | mixed corrupted image atomic commit |

这些是 error-effect models，不代表 CRC/CAS/ack 已实现；T4.2/T4.3 才负责 wire safety/fallback。

## Precision--resource--LER 结果

float reference 使用相同 observed-only estimator、相同一窗延迟和 analytic LLR，平均 LER 为
`0.0312269`；它没有 LUT resource claim。base quantized/joint-P12 LER 为 `0.0311923`，相对 float
差 `-3.46e-5`，95% paired CI `[-6.92e-5,-5.77e-6]`。这个极小负差是离散 regularization，
不是 fixed point 从信息论上优于 float。

### Joint profile 曲线

| profile | ADC/address/LLR-frac/threshold-frac/state/update | dual-bank bits | LER | Delta LER vs float |
| --- | --- | ---: | ---: | ---: |
| P03 | 3/3/0/1/3/32 | 80 | 0.11747 | +0.08624 |
| P04 | 4/4/1/2/4/16 | 182 | 0.10015 | +0.06892 |
| P06 | 6/5/2/3/6/8 | 416 | 0.17187 | +0.14064 |
| P08 | 8/6/4/5/8/4 | 1,068 | 0.09296 | +0.06173 |
| P12 | 12/8/8/9/12/1 | 6,212 | 0.03119 | -0.000035 |

P06 比 P03/P04 更差，证明多组件 precision 与 update staleness 会交互，不能声称 bit 数单调增加就
必然改善 LER。OAT 的 update-period LER 为 `0.03119,0.05198,0.09326,0.17655,0.11211,0.14981`，
最大 bank age 与 period 分别为 `1,2,4,8,16,32`；staleness 是本 stress law 下的主导项，但仍是
synthetic conclusion。

### Pareto 候选及反泛化边界

全体无故障 OAT/joint points 上的非支配搜索发现 `lut_address_bits=3`：8-entry LUT，double-bank
260 bits，LER `0.0292277`，相对 float 差 `-0.001999`，CI `[-0.002711,-0.001498]`。这反映当前
固定 threshold/噪声分布下的 coarse-LUT regularization；它不是通用“3-bit LUT 最优”，必须在
T5 多场景/mismatch 与 T5.5 synthesis 重新选择。

## Bank-fault paired effects

下表是同 seed、同 base precision 的 `fault - no-fault bank` LER 差：

| fault | paired Delta LER | 95% paired bootstrap CI | 结论 |
| --- | ---: | ---: | --- |
| state MSB flip | +0.04814 | [0.04675,0.04961] | severe，CI 下界通过 |
| LUT sign burst | +0.00439 | [0.00313,0.00562] | severe，CI 下界通过 |
| stale commit | +0.000223 | [-0.000156,0.000577] | 当前强度下不可判定 |
| torn update | +0.000129 | [-0.000054,0.000306] | 当前强度下不可判定 |

stale/torn 不被包装成正效应；它们只证明故障路径被逐 seed 触发，当前 effect CI 跨零。

## 资源与图形证据

- machine JSON：`docs/t2_4_3_fixed_point_validation.json`；
- 368-row Source Data：`docs/t2_4_3_precision_resource_ler.csv`；
- Python-only figure：`docs/figures/t2_4_3_precision_resource_ler.{svg,pdf,tiff,png}`；
- figure audit：`docs/t2_4_3_figure_validation.json`，5/5 gates PASS；
- SVG 保留 editable text；PDF 使用 TrueType；TIFF 600 dpi；PNG 300 dpi；
- implementation SHA256：
  `5cade4f41868da9b2efc3fb1736dda5e5867284912f6b6db308e237e2b18b29c`。

## 反简化审计

- 不是只扫一个全局 Q-format：6 个独立轴各 6 点，另有 5 个 joint profiles；
- 不是随机新 trace 比较：所有 profile/fault 复用同 seed raw error、measurement noise 和 truth；
- 不是用 hidden state 更新：circular estimator 的 API 只接受 observed centered syndrome；
- 不是浮点后贴标签：ADC、state、LLR、threshold 和 bank image 都保存整数 code，fault 在具体 bit 上发生；
- 不是总均值 gate：bank fault 逐 seed 触发，severe effect 使用 paired base CI；
- 不是资源伪综合：只报告精确 bit counts/payload width，LUT/BRAM/DSP/Fmax 为 `null`；
- 不是筛掉反例：joint 非单调、stale/torn CI 跨零、coarse regularization 全部保留。

## 复现

```powershell
python -m cnn_fpga.runtime.fixed_point_chain
python -m cnn_fpga.runtime.plot_fixed_point_chain
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\test_fixed_point_chain.py -q
```

## Claim boundary

允许写：在预注册 synthetic wrapped-Gaussian drift 下，位级软件模型给出 component-specific、
non-monotonic precision--representation--LER 曲线，并检测 bank corruption effects。禁止写：实际 FPGA
LUT/BRAM/DSP/Fmax、板端 ADC fidelity、bit upset rate、real-time hardware gain 或量子实验 LER。

