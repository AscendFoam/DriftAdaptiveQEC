# T4.3.2 双参数库、原子切换与 hysteresis

## 1. 结论与适用边界

T4.3.2 新增 `AtomicParameterImageBank`，把 T4.2.1 的完整 parametric MAP-LUT image 作为事务单位，而不是逐字段修改在线表。实现使用 A/B 两个有效 bank slot 和独立 transfer buffer：partial bytes 永不进入有效 slot；只有完整 payload 通过 manifest、transfer、canonical image 和 image self-check 后，才一次发布到 inactive slot；active 指针只在满足 apply epoch、安全 cycle boundary、CAS、freshness、minimum residency 和 selection hysteresis 时切换。

本任务证明的是 `transactional_double_bank_software_contract_not_rtl_or_board`。它不是 RTL 原子性、FPGA/board commit timing、真实通信协议或 device-calibrated hysteresis 的证据，也不包含自动物理回滚。旧 `DualLoopScheduler`/`ParamBank` 仍保留兼容路径；新 bank 是后续 T4.3.3 fault-stress 和 T5/T6 硬件化的候选 production primitive，不能把局部通过写成全 runtime 已迁移。

## 2. 不变量

1. active slot 始终保存一个通过 `ParametricMAPLUTImage.verify()` 的完整 immutable image。
2. transfer buffer 与两个 valid slots 分离；`begin_stage`/`write_chunk` 不修改 active 或 inactive slot。
3. incomplete、torn、corrupt、noncanonical、stale、replayed 或 CAS-conflicting payload 均 fail closed。
4. `finalize_stage` 是候选 image 写入 inactive slot 的唯一位置，且只执行一次完整对象替换。
5. `commit_if_ready` 不遍历、不修改 LUT payload，只原子切换 active/inactive 指针；T4.2 pipeline 已发出的请求保留其 latched version，新请求才读取新 image。
6. 成功 ack 还不是 host 侧完成；必须用独立 readback 比对 bank、version、activation epoch、CRC32 和 SHA256。
7. 所有公共状态读取与事务写入共用 re-entrant lock；同一时刻只允许一个 transfer 或 pending commit。

## 3. 完整 image 与 manifest

payload 是 T4.2 parametric MAP-LUT image 的 canonical ASCII JSON，含 config、active bank version、source parameter hash、model mean/sigma、X/Z integer tables、saturation count、image CRC32/SHA256。反序列化时拒绝额外字段、非 canonical encoding、derived config 不一致和 image self-verification 失败。

| 类别 | 字段 | 语义 |
| --- | --- | --- |
| schema/identity | `schema_version`, `transaction_id`, `selection_key` | 协议版本、anti-replay transaction、hysteresis key |
| CAS/version | `expected_active_version`, `new_version` | 要求 active 未变化，且 `new=expected+1` |
| provenance/time | `source_window_id`, `created_epoch`, `created_timestamp_ns`, `apply_epoch` | 来源窗口、创建/应用周期；timestamp 是 `epoch × 5000 ns` 的单调 cycle-domain 时间，不是 UTC wall clock |
| transfer integrity | `payload_length`, `payload_crc32`, `payload_sha256` | 对实际收到的完整传输 bytes 重算 |
| image integrity | `image_crc32`, `image_sha256` | 对 decoded image 自验证并与 manifest 交叉核对 |
| header integrity | `manifest_crc32`, `manifest_sha256` | 对除自身以外的 canonical manifest 重算，阻断合法范围内的 header bit flip |

CRC32 用于快速 accidental-corruption detection，SHA256 用于强完整性对照；二者都不是身份认证或数字签名。transaction ID 一经 `begin_stage` 接受即进入 anti-replay set，即使随后传输失败也不能复用。

## 4. 事务状态机

```text
observe two eligible same-key windows
  -> seal manifest
  -> begin_stage (schema/time/CAS/hysteresis/anti-replay)
  -> write_chunk (out-of-order and identical retransmit allowed)
  -> finalize_stage (coverage + transfer CRC/SHA + canonical image + image CRC/SHA)
  -> pending verified inactive image
  -> wait apply epoch / safe boundary / minimum residency
  -> recheck CAS + freshness + hysteresis
  -> atomic pointer swap
  -> ack
  -> readback verification
```

冲突 overlap、越界 chunk 会终止 transfer；相同 bytes 的 overlap 作为幂等重传。未覆盖全部 bytes 时 `payload_incomplete`，允许继续补传而不发布 slot。verified candidate 在 unsafe boundary 或 minimum residency 未满足时只返回 non-final deferred ack；CAS、freshness 或 hysteresis 在 commit 前失效时返回 final rejected ack，active image 不变。

## 5. cadence 与 hysteresis policy

生产 reference 继承 T4.3.1：fast cycle `5000 ns`，两次连续 eligible 且同 `selection_key` 的窗口才可晋升，minimum residency `4000 cycles = 20 ms`，payload 最大 age `8192 cycles = 40.96 ms`，最大 payload `1 MiB`。`source_window_id` 必须等于最新 hysteresis evidence window；任一 ineligible window 或 key change 都会重置 run。

`safe_boundary_period_cycles=1` 表示软件 reference 可在每个 fast-cycle 边界切换，但调用方仍必须显式传入 `safe_boundary=True`。该布尔量目前是 caller contract，不是 CDC/RTL handshake 证据。上述 2-window、4000 和 8192 都是可审计 policy，不是从真实装置校准出的最优参数。

## 6. 非 demo 验证

production validator 生成 `7518` 行 Source Data，17/17 gates 通过：

| 验证族 | 覆盖 | 结果 |
| --- | ---: | --- |
| proper-prefix cut | 3745 个 cutpoint，覆盖长度 0 到 3744 | 全部 `payload_incomplete`；active v0 不变，inactive 未发布 |
| single-byte corruption | 3745 个 byte position 各 XOR 1 次 | 全部在 decode 前由 transfer CRC 拒绝；两个 valid slots 不变 |
| chunk/order matrix | 1/7/64/511/3745-byte × forward/reverse，共 10 例 | 中间 slot 不变；完整 image 均可 commit，unsafe boundary 均 deferred，ack/readback 均一致 |
| semantic negatives | 15 例、14 个稳定 reason code | timestamp、CAS、apply age、CRC/SHA、manifest、overlap、replay、stale、hysteresis 均保持 active 不变 |
| double-bank sequence | `A:v0 -> B:v1 -> A:v2` | 第二次 stage 期间 active 仍为 B:v1；两次 readback 均确认 |
| pipeline version latch | commit 跨越 T4.2 五级 pipeline | in-flight 输出 v0，commit 后新请求输出 v1 |
| concurrent writers | 两线程同时 `begin_stage` | 恰一例 accepted，另一例 `writer_conflict_transfer_in_progress` |
| determinism/source binding | 两次独立完整 evidence run | row/summary hash 相同；JSON 绑定实现与 CSV SHA256 |

负向 reason 集合包括 `hysteresis_not_satisfied`、`hysteresis_invalidated`、`timestamp_epoch_mismatch`、`expected_active_version_mismatch`、`apply_epoch_stale`、`transfer_crc_mismatch`、`transfer_sha256_mismatch`、`manifest_image_digest_mismatch`、`manifest_crc_mismatch`、`manifest_sha256_mismatch`、`conflicting_overlap`、`transaction_replay`、`payload_stale` 和 `payload_stale_before_commit`。

## 7. 产物与复验

- runtime：`cnn_fpga/runtime/atomic_parameter_bank.py`
- production validation：`cnn_fpga/benchmark/atomic_parameter_bank_validation.py`
- machine result：`docs/t4_3_2_atomic_parameter_bank_validation.json`
- Source Data：`docs/t4_3_2_atomic_parameter_bank_source_data.csv`
- direct tests：`tests/test_atomic_parameter_bank.py`、`tests/test_atomic_parameter_bank_validation.py`

```powershell
python -m cnn_fpga.benchmark.atomic_parameter_bank_validation
python -m pytest -q tests/test_atomic_parameter_bank.py tests/test_atomic_parameter_bank_validation.py
```

## 8. 尚未完成

- T4.3.3：把新 bank 接入闭环 fault harness，注入 drift/burst/leakage、host timeout、通信中断、jitter、pause、race 和 rollback，并验证无未定义动作。
- T5.1/T5.4：校准 hysteresis/age/OOD policy，做 correlated/OOD/long-horizon stability。
- T5.5/T6：实现 framed transport、CDC/RTL atomic handshake、synthesis/post-route 和真实 FPGA/board readback。
- 自动回滚、watchdog、transport timeout 和物理 reset/recovery 本任务未实现；active image 的 fail-closed 保留不能被改写为这些功能已经完成。
