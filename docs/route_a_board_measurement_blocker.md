# T6.9.2 Route-A 实物板测阻塞合同

当前没有实际开发板 inventory、真实 transport adapter、板级 timestamp calibration、board correctness smoke、board HIL qualification 或 bitstream manifest。T6.9.1 只提供 open-source P&R estimate，不能替代这些外部实物证据。

机器报告 `docs/t6_9_2_route_a_board_measurement_blocker.json` 固定以下 fail-closed 状态：

- execution branch：`BLOCKED_NO_PHYSICAL_BOARD_BITSTREAM_OR_TRANSPORT`；
- 六项 physical prerequisites 全部 `passed=false` 且 `observed_path=null`；
- measurement run manifest/raw data 均为 `null`；
- correctness、deadline、四层 latency、II/Fmax/jitter、resource、power 和 speed comparison 共 42 个 measured fields 全部为 `null`；
- board correctness=`NOT_RUN_BLOCKED`，zero deadline miss=`NOT_ESTABLISHED`，FPGA speed advantage/fastest/SOTA=`PROHIBITED`。

P&R clock model `222.222 ns` 和 analytic nominal power sensitivity `12.245 mW` 只保留在 `non_substitution` 区，明确标记没有复制到 measured source-to-action 或 measured power。

恢复 T6.9.2 必须同时具备九项条件：实板 inventory/照片版本、真实 framed transport、timestamp calibration、bitstream/source/tool/constraint hashes、board smoke、至少百万周期、零 mismatch/undefined/silent-overflow/deadline-miss及其零事件上界、分层 latency/resource/power、以及 speed claim 所需 same-task external comparator。只补一个文件不能解锁。

验证：

```powershell
python -m cnn_fpga.benchmark.route_a_board_measurement_gate --verify docs/t6_9_2_route_a_board_measurement_blocker.json
python -m pytest tests/test_route_a_board_measurement_gate.py -q
```

当前为 11/11 gates、11/11 semantic mutations、5 tests 通过；这是可信的 Blocked 证据，不是 Done 的板测结果。
