# Route-A unified execution contract

## 结论

T6.5.2 已把 Route-A 的公平性要求冻结为可执行、fail-closed 的统一合同。七个
observed-only comparator 候选共享同一输入 schema、action word、MAP-LUT 数值/地址协议、
A/B bank、更新机会、资源上限和 deadline 账本；hidden-state oracle 使用独立 truth schema，
不能进入 deployable adapter。

本任务冻结的是执行合同和 validator，不是 T6.6 的 comparator 性能结果。当前 verdict 为
`PASS_UNIFIED_EXECUTION_CONTRACT_FROZEN`，17/17 gates 通过；逐方法逐合同维度共 70 个
mutation 全部 fail-fast。

## 1. 同一在线观测

所有 deployable 候选只接收 `route-a-observed-syndrome-v1`：

- trace/cycle index；
- 10-bit `syndrome_code` 和 X/Z/phase observed fields；
- 8-bit OOD score、16-bit parameter age；
- reset ack、observation-valid、deadline-ok。

字段集合必须精确相等：缺字段、额外 debug 字段、bool 冒充 integer、越界 code 均拒绝。
`truth/hidden/latent/drift_state/logical_class/regime_label/future/evaluation_label` 名称进入
deployable packet 时立即拒绝。观测字段随后进入现有 CRC-16/CCITT-FALSE packed word，并做
encode/decode 全字段 round-trip；bit flip 必须使 `input_crc_ok=false`。

## 2. Oracle 的物理隔离

oracle 使用 `route-a-isolated-simulator-truth-v1`，其中包含 latent displacement、mean、sigma、
correlation、logical X/Z 和 regime。它只有 `nondeployable_upper_bound_only` manifest：

- 不使用 deployable input schema；
- 不参与 matched resource/deadline ranking；
- 不获得 MAP bank 或 RTL 兼容性；
- oracle truth mapping 直接传给任何 deployable adapter 都会因 schema/denylist 失败。

## 3. MAP-LUT 与 6-cycle action path

共同 fast-path contract 绑定现有生产核：

| 字段 | 冻结值 |
| --- | --- |
| ADC / address / fraction | 10 / 8 / 2 bit |
| table | X/Z 两相位，每相位 257 entries |
| LLR | signed Q9.12，22 bit |
| rounding | round-to-nearest, ties-to-even |
| saturation | `[-2^21, 2^21-1]` |
| logical dual-bank payload | 22,616 bit |
| current RTL mirrored memory | 45,232 bit |
| path | 5-cycle MAP + 1-cycle registered event/action，II=1 |

“共享 LUT”指相同 code grid、interpolation、Q-format、CRC 和 bank image layout；不同方法只能在
共同 update cadence/commit 处产生其方法特定的 LLR contents。standard binning 必须在 T6.6
编译为该网格上的固定 image 并穷举等价，不能旁路 action path。

### 必须保留的 joint-MAP 边界

现有 RTL 是 phase-conditioned X/Z LLR LUT，并不是 full two-dimensional joint MAP。
因此 `static_joint_map` 当前可以作为 observed-only、budgeted software comparator，但其
`current_rtl_compatibility` 被锁为
`blocked_full_2d_joint_map_not_equivalent_to_current_phase_lut`。在 T6.6 给出无损 projection
证明或新增/验证 joint RTL 前，禁止把该行标成 current-RTL deployable。这不是删掉 strong
baseline，而是避免用相同名称掩盖不同在线能力。

## 4. Bank、cadence 与预算

- A/B bank：CRC16 wire、CRC32/SHA256 image、manifest CRC32/SHA256、`active+1` CAS、
  last-known-good rollback、6-cycle retired-bank drain；禁止 partial publish。
- fast event：每 cycle；regime window/update：32/32 cycles；parameter window/update：
  2,048 samples / 4,000 cycles；parameter update 需 one-window causal delay。
- 每次 parameter update 的 algorithm cap：8,192 MAC；private model/state：8,192 B；
  transient workspace：8,192 B；host wall-clock ceiling：5,000 us。
- 8,192 cap 是给 comparator 的共同上限，不是“每个方法已实测使用 8,192 MAC”。实际 MAC、
  state、workspace 和 wall-clock 必须在 T6.6 逐方法如实报告，低于上限不能补齐，超过则失败。

## 5. Deadline accounting

每个 deployable output 都必须记录：input cycle、valid cycle、source-to-action cycles、logical
deadline、update due、MAC、private state、workspace、host update wall-clock、host deadline 和
board deadline。

- logical action 必须精确在 `input+6`；提前/延后均为合同不一致；
- host wall-clock `>5000 us` 为失败，flag 必须与数值一致；
- update 不到期时 MAC/wall-clock 必须为零；
- 负成本、bool-as-int 和缺字段都在对象构造层拒绝；
- 真板未到时 `board_measured_deadline_miss` 必须为 `null`，不能填 `false` 冒充零 miss。

## 6. 方法表

| 方法 | 在线 privilege | update 行为 | current RTL 状态 |
| --- | --- | --- | --- |
| standard binning | observed-only | fixed rule -> common LUT image | 等待 T6.6 exhaustive equivalence |
| static joint MAP | observed-only | training-frozen | full 2D 与 phase LUT 不等价，blocked |
| Window MAP | observed-only | common parameter boundary | 等待 adapter proof |
| EWMA adaptive MAP | observed-only | common parameter boundary | 等待 adapter proof |
| Kalman adaptive MAP | observed-only | common parameter boundary | 等待 adapter proof |
| legacy CNN residual | observed-only | common parameter boundary | checkpoint + budget 通过后才候选 |
| proposed Route-A | observed-only | 32-cycle regime + 4,000-cycle parameter | 等待 policy/adapter proof |
| hidden-state oracle | isolated truth | current hidden state | prohibited / upper bound only |

## 7. 反简化验证

1. 7 个 deployable 方法 × 10 个共同字段逐一 mutation，共 70 个 mismatch 均以
   `method_contract_mismatch` 拒绝；
2. 每个 deployable 方法都单独验证 hidden truth key 拒绝；另有 missing/extra/type/range/schema
   失败分支；
3. accounting 在精确上限通过，early/late action、MAC/state/workspace overflow、host deadline、
   pre-board measurement、no-update work 和 oracle ledger 共 9 类失败；
4. 第二轮深审修复了 logical LUT payload 与 mirrored physical memory 重复计数；
5. 第二轮深审又封住 direct dataclass 构造负成本绕过，并把 standard binning 改为固定 LUT
   image 语义；
6. 13 个 focused tests 通过，机器报告 gate 独立重算，关键 semantic mutation 会使 verifier
   失败。

## 8. 证据边界与下一步

当前允许声称“统一执行合同已冻结并通过 fail-fast validator”。当前不允许声称：

- matched-budget 方法已经跑出 LER/tail 优势；
- full joint MAP 已在当前 phase LUT 上等价实现；
- legacy CNN checkpoint 已满足预算；
- Route-A policy 已完成；
- 真板零 deadline miss 或 measured latency。

这些分别由 T6.6、T6.7、T6.8 和 T6.9 的既定任务承接。

