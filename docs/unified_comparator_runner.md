# T6.6.1 统一 comparator adapter 与 matched-budget runner

## 结论

T6.6.1 已建立一条可执行但**非正式结果**的统一集成资格链：六个 observed-only 方法从同一
T6.5.2 wire packet trace 重建 syndrome，legacy CNN 加载真实 checkpoint 后因输入合同和预算失败
自动降为 ablation-only，hidden-state oracle 只在物理隔离的上界表中运行。当前结果不能用于声称
Route-A 已优于 Kalman，也不能用于 FPGA 实测、T6.7 formal LER 或论文 SOTA。

## Packet 到 joint decoder 的桥

现有 fast-path 每个 cycle 只携带一个 quadrature 的 10-bit syndrome，而 joint MAP 需要二维 q/p
输入。runner 没有额外传入浮点 syndrome：

1. 每个 logical comparison sample 必须由相邻的 `phase=0(q)`、`phase=1(p)` packet 组成；
2. packet 必须同 trace、相邻 cycle、CRC roundtrip 有效；缺相位、重排或不相邻均 fail closed；
3. 每个 2048-sample 参数窗就是 2048 个 scalar packet / 1024 个 q/p pair，窗口 stride 保持
   4000 fast cycles；
4. HMM 在每个 fast cycle 读取当前相位的新值和另一相位最近一次 observed 值，每 32 cycles 更新，
   不读取当前二维浮点 side-channel 或 hidden state。

这一区分很重要：full 2D joint MAP 仍是软件 comparator，不能因为它使用相同 action schema 就被
写成当前 phase-conditioned RTL 的 bit-exact 方法。

## 实际执行的方法

| 方法 | 实际路径 | 当前资格 |
| --- | --- | --- |
| standard binning | centered modular syndrome 固定 central class；编译为常正 LLR LUT | common trace；2048 个 phase/code cases 零不一致 |
| static joint MAP | T5.1.2 training-only frozen mean/covariance + `map_decode_2d` | common trace；software joint lane |
| Window MAP | `LatestWindowPeriodicPredictor`，当前窗预测完成后才更新 | common trace |
| EWMA MAP | 冻结 `alpha=0.85`，真实 periodic-moment recursion | common trace |
| Kalman MAP | 冻结 process/measurement scale `1.5/0.75`，真实 10-state Joseph update | common trace |
| proposed Route-A | T4.1.1 validation-selected HMM + Kalman candidate + trusted static | integration-only；阈值/完整状态机未锁 |
| legacy CNN residual | 真实 21×32×32 float checkpoint，4 个 native witness samples | schema 与预算失败，ablation-only |
| hidden-state oracle | `oracle_map_2d(residual, DriftState)` | 独立 oracle 表，不参与 deployable ranking |

T6.5.2 frozen manifest 没有被原地改写。T6.6.1 另外生成 versioned adapter overlay，给八个 role
绑定 wrapper、真实算法源码和 SHA-256；这样既保留预注册合同 hash，也不再依赖“adapter pending”
叙述来证明实际执行。

## 成本核算

periodic feature frontend 在 10-bit code grid 上可用四个 phase/index LUT、两个 modular index 和整数
累加实现。runner 对全部 `1024^2` q/p code pairs 穷举两条 complex-product identity，共
`2,097,152` cases，最大复数误差约 `1.36e-15`；因此部署映射不需要 runtime transcendental 或实数
MAC。host qualification 仍执行现有 NumPy complex-moment 实现，并单独记录 wall-clock。

方法私有成本由矩阵维度计算，不使用任意“经验小数字”：

| 方法 | max real MAC / update event | float32 私有状态 | transient workspace |
| --- | ---: | ---: | ---: |
| standard | 0 | 0 B | 0 B |
| static joint | 0 | 24 B | 256 B |
| Window | 128 | 32 B | 512 B |
| EWMA | 136 | 80 B | 512 B |
| Kalman | 7,121 | 1,740 B | 2,048 B |
| Route-A worst collision | 8,047 | 5,468 B | 3,072 B |

Route-A 的 worst collision 是同一 boundary 上 `926` HMM MAC 与 `7,121` Kalman MAC 相加。不能
只报单事件上限而隐藏 HMM 高频运行：每个 2048-cycle 窗有 64 次 HMM update，因此冻结参数窗的
累计计算为 `59,264` MAC；允许 Kalman update 时为 `66,385` MAC。二者均写入逐窗 ledger。

legacy CNN 的实际 checkpoint 解析结果为每次推理 `3,489,984` MAC、`1,586,368 B` inference
state、至少 `65,536 B` transient activation，远超 8192 MAC/8 KiB/8 KiB caps。它还需要含
teacher channels 的旧 21-channel tensor，项目没有从 T6.5.2 scalar packet 到该 tensor 的无泄漏
等价变换，所以不允许进入 common-trace LER 主表。模型、manifest、test split 还必须与 T5.4.3
既有 SHA anchors 一致，不能在本任务重新生成一个“更方便”的 CNN。

## Qualification 结果与负结果

默认 trace 是 16-window calibration-shift integration witness，每方法 `16,384` logical decisions。
它用于验证 adapter、因果顺序、成本和分栏，不是 T6.5.3 的 24-cluster formal matrix。确定性 LER
如下：

| 方法 | qualification LER |
| --- | ---: |
| standard | 0.0310669 |
| static joint | 0.0231323 |
| Window | 0.0074463 |
| EWMA | 0.0075073 |
| Kalman | 0.0075684 |
| proposed Route-A integration | 0.0233765 |
| hidden-state oracle | 0.0036011 |

Route-A integration 明显落后于 Window/EWMA/Kalman，不能隐藏。原因是尚未校准的 HMM/policy 在
qualification trace 上产生 `8,384` 次 normal-Kalman、`7,793` 次 tail-trusted-static、`192` 次
uncertain-static 和 `15` 次 warmup-static；16 个参数窗只有 8 个开放 Kalman update。这个负结果正是
T6.6.2/T6.6.3 必须完成 policy/hysteresis/calibration 的理由，而不是在 T6.6.1 事后调阈值。

## 深审计

- future packet 在第 8 window 后被系统性变换；六个方法的 prefix decisions 全部逐 bit 不变，除
  固定 standard 外其余方法的 future decisions 均发生变化，排除 vacuous causality check；
- phase reorder、non-adjacent cycle、missing phase、truth key、missing CNN checkpoint、CNN budget
  overflow、oracle 进入 deployable accounting 共 7 类 mutation 全部 fail closed；
- standard rule 对两相位×1024 code 穷举零 mismatch；periodic feature grid 对 209 万 identity cases
  穷举；
- oracle 无 deployable accounting、无 common ranking；CNN 只消费 `histograms`，online 不消费
  `labels/target_params`；
- 每个 common-trace accounting record 强制 exact 6-cycle logical action、board field 为 null、
  private MAC/state/workspace 和 host update deadline 均通过 T6.5.2 validator。

## 证据边界与下一项

当前只允许声称“统一 runner 和 adapter 资格链已经可执行”。proposed policy 的 leakage/reset、
CRC/version/age、last-known-good rollback、trusted-bank switching、hysteresis recovery 和逐 action reason
完整性仍属于 T6.6.2；posterior calibration、threshold lock 与 formal-result isolation 属于 T6.6.3。

复现命令：

```powershell
C:\ProgramData\anaconda3\envs\DLEnv\python.exe scripts\export_t411_hmm_checkpoint.py
python -m cnn_fpga.benchmark.unified_comparator_runner
python -m pytest tests\test_unified_comparator_runner.py -q
```

关键产物：

- `docs/t6_6_1_unified_comparator_runner.json`
- `docs/t6_6_1_unified_comparator_runner_source_data.csv`
- `artifacts/models/route_a/t4_1_1_gaussian_hmm.json`

