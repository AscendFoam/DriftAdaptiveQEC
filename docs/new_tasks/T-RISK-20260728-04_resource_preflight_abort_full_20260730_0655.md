# T-RISK-20260728-04 旧 full-resource 事务受控终止记录

- 日期：2026-07-30
- 事务：`full_20260730_0655`
- 状态：`INCOMPLETE_RESOURCE_FAIL_CLOSED`
- 性质：失败尝试证据；**不是** task 完成记录、科学 `NO-GO`、twin qualification 或性能结论
- 后续：旧事务永久只读；修复后必须以新 run ID 和重新密封的 V4 合同 fresh 执行

## 为什么终止

运行期间的源代码/合同复核发现两个会使该事务无法被合法消费的 P0 问题：

1. 当前 config 已进入 V3，但 preformal audit 和其测试仍硬编码 V2。即使资源预检成功，preformal 也会确定性拒绝该事务。
2. 旧 independent verifier 把 hidden RESET outcome 与带噪声的 observed `reset_ack` 直接等同。首个真实 P11 receipt 中存在 7 个合法 acknowledgement noise flip；旧逻辑会错误拒绝这些行，也无法防御协调篡改。

preformal 还会绑定运行时和验证源快照。上述问题必须修改源代码才能修复，因此让旧进程继续生成约 117 GB 正式原始证据既不能产生可消费事务，也会造成证据语义混杂。确认唯一 owner/children 身份后，对该事务执行了定向受控终止；parent 随后自行写出失败报告和闭合的 `START -> FAIL` event chain。

`resource_preflight_failed.json` 中的三个 worker `exitcode=4294967295` 是受控终止的直接进程表现，不是根因。资源采样显示四 worker 峰值 aggregate RSS 仅为 `1,885,331,456 B`，没有 OOM 证据；supervisor stdout/stderr 均为空，也没有可支持“自然超时”或科学失败的证据。因此不得把本次失败归因为 OOM、host timeout 或 backend scientific disagreement。

## 保留的不可变证据

旧运行目录：

- `runs/t04_resource_preflight_full_20260730_0655/`
- `runs/t04_resource_supervisor_full_20260730_0655/`

关键文件（`bytes / SHA-256`）：

- `resource_preflight_failed.json`：`3,653 / ecb0c7a27faba70b81d1255663be3933a4e2bb3ec1867803182b944fcbea1904`
- `attempts.jsonl`：`1,288 / 3341af234dcdda552621421e5a86cb5e2512d33002ec4b4d8d2cd78cfdc51147`
- `heartbeat.json`：`468 / dbeb57d93367b93707f0d807ca25706555c4c44e9cf61d52bedc54f44137b98d`
- `resource_samples.jsonl`：`553,905 / 08b146bc8d416233dc9887f29f39eb97ae54d8e0c41fef7288af93fe77cbe361`
- 首个 P11 receipt：`9,049 / f2a33ce63d6de013afe4f5979f4a7f534ea864f29252676059ec374e0f853542`
- `launch_meta.json`：`1,201 / 74e039fcd8924d2c911588e7a1de7c808220d28a83c9966c39b47ca1ba931038`
- supervisor `stdout.log`、`stderr.log`：均为 `0 B / e3b0c442...b855`

失败报告自哈希为 `cdaacc50c46f4c092d6bac816f23b597f6e7590d69c058031e78829c268192a3`，其冻结输入包括：

- config SHA-256：`95a31cc8ccfae0d128d2cfc24d24abc88175602f7e890e49de11f5f8d0cdb634`
- plan SHA-256：`d2e6fd930c0fb9f0271c0e0ef66e8396fefd1b424b0ca1a2ba01007f6e28376a`
- source snapshot SHA-256：`7aede8b185d4a8553b4ea4136a4fbd291c37195dbd6d0197b90c7d90cf0bbd2c`
- resource sample count：`914`
- maximum live children：`4`
- peak aggregate RSS：`1,885,331,456 B`
- formal seed address accessed：`false`
- formal artifact namespace accessed：`false`

退出后已确认 owner `26952` 与 children `11220/21432/23880/27396` 全部不存在，没有孤儿进程。`staging/` 保留 `52` 个文件、`7,518,448,778 B`；content-addressed object store 保留 `15` 个唯一对象、`433,654,730 B`，不得清理、移动或复用到 fresh 正式事务。

## 首个 receipt 的有效范围

首个完成 chunk 为：

`probe_c36_probe_P11_RESET_FAIL_RESET_B__812af2e526470986c288`

它覆盖 backend B、cutoff 36、`P11_RESET_FAIL`、RESET action：

- expected/observed rows：`1,536 / 1,536`
- exception/missing/conservation failure：`0 / 0 / 0`
- RESET primary/sidecar rows：`1,536 / 1,536`
- object roles / unique objects：`18 / 15`
- unique object bytes：`433,654,730`
- receipt canonical self-hash：`ce688dbdc06fd75a7b46e0c7a762e708c5f4a3a9f0cbead9c4471035b43a231c`

独立 live audit 已重新读取全部对象，验证其路径均位于旧 object store 内，size 和 SHA-256 全部匹配。该 receipt 仅证明一个 chunk 的原始写入完整性和资源可行性；它不闭合 518-cell denominator，不可进入正式统计或 twin qualification。

## 修复与 fresh-run 门

旧事务之后实施的 V4 修复包括：

- preformal 只接受精确 V4，拒绝 V1--V3；
- 对 task ID、7 项 parent semantics、11 项 contract gates、claim key 集和全 `null` 做 exact 校验；
- 对 plan、seed registry、historical scan 做 live bytes 与独立重算 exact equality；
- independent verifier 以 stdlib 独立重放 A/B RESET 和 acknowledgement RNG 消费顺序，不导入 producer backend；
- hidden outcome、observed noisy acknowledgement、selected density 和 branch presence 分别校验；
- 增加旧 schema、源漂移、合同协调篡改、ack/hidden/density 协调篡改、非法概率、A/B 非零 round 与非正式尺寸冻结向量测试。

注册的 9 个测试文件在 V4 物化前最终通过 `162 passed, 1 skipped`。两次只读独立终审分别确认 V4 gate graph 无循环/兼容旁路，以及 A/B RNG replay 与真实 backend bit-exact。

只有在这些源代码和测试不再变化后，才允许物化新的 immutable V4 contract，并以新的 run ID 启动 full-resource preflight。任何 fresh run 仍须保持：

- `qualified_claim = null`
- `scientific_verdict = null`（resource/preformal 阶段）
- twin/LER/lifetime/physical/hardware/official/Puviani/SOTA/rank 全部 `null`
