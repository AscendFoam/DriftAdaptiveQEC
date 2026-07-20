# T6.2.1 production 定点 MAP-LUT 与事件 FSM 审计

## 结论

T6.2.1 通过板卡无关资格验证：`gkp_fast_path_core.sv` 不是组合 LUT 或 trace replay demo，
而是含真实 BRAM 读写、定点插值、事件/回退状态、frame accumulator 和 CRC 状态输出的同步
RTL。新增 `gkp_fast_path_production_top.sv` 后，配置与切换路径具有 inactive-bank-only、严格
514-word 顺序、CRC32、16-bit compare-and-swap version、safe-boundary 延迟提交、取消和
6-cycle retired-bank drain guard。

独立 Python reference 与实际 Yosys CXXRTL 在 1,681 个周期上逐 bit 一致，覆盖三次完整
514-word 写入、正确/错误 CRC、两次 bank switch、延迟/取消 commit、5 次一致性状态快照和
全部 11 类拒绝原因；`mismatch_count=0`，8/8 gates 通过，7/7 语义输出篡改被拒绝。

该结论仍是 synthesizable RTL + cycle-accurate CXXRTL，不是 transport/CDC、target P&R、
bitstream 或真板结果。

## Requirement-to-RTL mapping

| 要求 | 实现位置 | 关键语义 | 验证 |
| --- | --- | --- | --- |
| syndrome classification | `gkp_fast_path_core.sv` input unpack + `input_reserved_observation` | 58-bit input CRC；`g/e/leakage` 二位编码；保留值 fail-closed | T5.5 fault trace + T6.2.1 source gate |
| MAP-LUT | core 的 8 份 257×22 mirrored memory 与 5-stage pipeline | X/Z、A/B、合法 1R1W、ties-to-even interpolation | T5.5 全 4,096 code/bank/phase 对拍；本轮结构展开保留 8 memories / 45,232 bits |
| run-length / event FSM | `x_e_run`、`z_e_run`、`leakage_run`、`leakage_clean_run`、`mode` | 饱和递增、reset/hold/recovery/fallback | T5.5 full output/state 对拍；本轮未删除状态路径 |
| Pauli / phase frame | `pauli_frame_x/z`、`phase_frame_x/z` | accepted action 才更新；阻塞事件抑制 correction | CRC-protected 232-bit state 与 118-bit action 对拍 |
| trusted A/B bank | production top 的 `bank*_trusted/version` + config session | 只能写 inactive bank；开始写即撤销 trust；失败保持 untrusted | active-bank、order、incomplete、CRC、untrusted commit negative paths |
| version / CRC | config CRC32 + 16-bit CAS + state CRC16 | version 必须 `active+1`；`0xffff` fail-closed；CRC32 是完整性而非认证 | 正确/错误 CRC 全表流；version reject；5 次状态 snapshot CRC 复算 |
| saturation | core 的 LLR clamp、run/counter saturating increment | 禁止 signed overflow 和计数回绕 | T5.5 边界与 mutation；本轮保留 parameterized threshold |
| leakage | event mask bit 13、leakage/reset modes | leakage 可触发 reset/hold；与普通 `e` 分类分离 | fault trace 与 state mask 对拍 |
| deadline / fallback | event mask deadline/age/OOD/trust/CRC/version | blocking fault 禁止 action；production age ceiling 为 8,192 cycles | core 参数化 + source/semantic tests |
| action output | 5-cycle MAP + 1-cycle registered action | `correction_enable`、inhibited、action code、frame delta、fault mask、trusted version | T5.5 4,316 valid rows 0 mismatch |
| atomic commit | production top pending commit + core latch | safe boundary 前保持 pending，可取消；切换后 6 cycles 禁止重写 retired bank | deferred/cancel/switch/complete/drain 全路径逐周期对拍 |
| coherent readback | 144-bit payload snapshot + byte-serial CRC16 | 请求时冻结状态；18 cycles 后 `state_valid`；不把 144-bit CRC 放在 fast path | 5 snapshots，CRC 逐个复算一致 |

## 深审发现与修复

1. **raw 配置地址越界**：原 core 接收 9-bit address，但只在 activity harness 中约简；现 core
   自身只允许 `0..256`。
2. **活动 bank 可写**：原 core 没有底层保护；现 core 与 production top 双层拒绝。
3. **commit 过宽**：原 `commit_valid && safe_boundary` 不检查 trust、inactive bank 或版本；现
   commit 同时要求 trusted、inactive、`active != 0xffff` 与 `new=active+1`。
4. **demo 版本上限**：原事件 mask 将 `version>7` 视为故障；现阈值参数化，T5.5 默认仍为 7 以
   保持旧 golden，production top 使用完整 `uint16` 范围。
5. **production age 不匹配**：原硬编码 64 与闭环配置 8,192 冲突；现 core 参数化，production
   top 显式使用 8,192。
6. **写入中途失败仍可能被信任**：现 begin 即撤销 target trust；order、incomplete、CRC、冲突或
   abort 均保持 untrusted。
7. **in-flight bank collision**：commit 后加入 6-cycle drain guard，禁止立即重写刚退役 bank。
8. **管理状态 CRC 的长组合路径**：第一版 144-bit 组合 CRC 虽通过语义测试，但不适合作为
   production 管理路径；现改成冻结快照 + 1 byte/cycle，18 cycles 完成且不阻塞 fast path。

## 复现

```powershell
$env:PYTHONPATH='.'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
C:\ProgramData\anaconda3\envs\DLEnv\python.exe -m cnn_fpga.benchmark.production_rtl_audit
C:\ProgramData\anaconda3\envs\DLEnv\python.exe -m pytest -q `
  tests/test_production_rtl_audit.py tests/test_rtl_fast_path_equivalence.py
```

正式审计使用 Yosys 0.67 和 MSYS2 g++ 15.1.0。CXXRTL 模型 SHA256、可执行文件 SHA256、
工具版本、运行时间、gate 与 evidence boundary 均保存在机器报告中。CSV 保留全部逐周期 actual / expected。

## 剩余缺口

- 本 task 未实现 UART/USB-SPI framing、CDC、backpressure 或 host reconnect；这些属于 T6.1.2。
- 本 task 未执行 production top 的 target-device P&R；T5.5 的资源/Fmax 仍只对应此前 core/activity
  harness，production 集成的 target estimate 需由 T6.2.2/T6.3 接续。
- CRC32 只检测偶发传输损坏，不提供 authentication / anti-replay；SHA256 manifest 仍由 host
  provenance 验证，不能写成硬件安全认证。
- CXXRTL 是二值同步仿真，不覆盖亚稳态、CDC 和物理 SEU；T6.2.2 只可注入抽象 bit/CRC/stale/
  timeout 故障，真板 deadline/transport/power 仍等待 T6.4。
- 现有 action path 的 6-cycle latency 是 RTL contract；无板前不能升级为 measured source-to-action。

这些缺口由既有 T6.1.2、T6.2.2、T6.3 与 T6.4 正常任务承接，不需要插入旁路 task。

