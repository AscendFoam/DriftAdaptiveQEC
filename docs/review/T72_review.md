# T72 Review

## 结论

`T72` 的目标已经完成，而且完成方式保持在任务包边界内：

- `probe_limitations` 不再是固定字符串，而是来自本次执行的结构化 probe 记录
- `source_records`、`repo_board_defaults`、`expected_byte_count_basis` 都已经改成 execution-derived / override-aware
- `--config` / `--mmio-path` / `--dma-path` 的 focused regression 已补齐
- `T49` replay verdict 与 `T72` current-host regeneration verdict 仍然一致，都是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

## 已验证的证据

本轮实际执行了：

1. `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
2. `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
3. `python -m unittest tests.test_t72_real_board_transfer_pack_provenance_hardening`
4. `python -m cnn_fpga.hwio.collect_t71_real_board_gate_artifacts --output-dir artifacts/t72_real_board_transfer_pack_provenance_hardening`
5. 一次用 `T72` artifact 驱动 gate helper 的真实执行
6. 一次用 `T49` checked-in artifact 驱动 gate helper 的 replay 执行
7. `--config` 与 `--mmio-path` / `--dma-path` override 的 focused regression（体现在 `tests/test_t72_real_board_transfer_pack_provenance_hardening.py`）

## 本轮值得认可的点

### 1. provenance 不再把“未执行”伪装成“已失败”

`host_fact_manifest.json` 现在同时保存：

- `probe_execution_records`
- `probe_limitations`

所以 reviewer 可以直接看到：

- 哪条 probe 成功执行
- 哪条 probe 命令失败
- 哪条 probe 因平台不匹配而 `not_applicable`

这正是 `T71` 留下的 `W1/W2` 想收口的地方。

### 2. override 终于不是“值变了，说明文字没变”

`repo_board_defaults` 和 `bitstream_evidence.source_records` 现在都带动态来源信息。

尤其是：

- `config_path`
- `config_argument_kind`
- `candidate_mmio_path_record`
- `candidate_dma_path_record`

这让 future-host 在使用 `--config` / `--mmio-path` / `--dma-path` 时，不会再出现“artifact 字段值换了，但 provenance 叙事还停留在默认 config”的问题。

### 3. hardening 没有污染 verdict

`artifacts/t72_real_board_transfer_pack_provenance_hardening/replay_vs_regeneration_comparison.json` 证明：

- verdict 没漂移
- strongest supported statement 没漂移
- device / bitstream / repo execution 三层 truth status 都没漂移

这说明本轮改动确实是 provenance 收紧，不是偷偷改 gate 口径。

## 仍需如实保留的边界

### 1. `T37` 仍然 blocked

当前缺口仍然是：

- 没有可只读打开的真实 `mmio` / `dma` 设备路径
- 没有 bitstream-to-RTL / DMA / fixed-point contract 绑定事实
- `board_backend.py` 仍是 placeholder-only

### 2. 这不是 production-ready host-transfer tool

本轮确实把 transfer-pack 做得更严谨，但还不能把它写成：

- 真板 ready
- 真板执行成功
- `hardware_validated`
- `deployment closure`

## 建议

建议 Captain 将 `T72` 视为对 `R31` 的直接收口候选，并重点复核两件事：

1. 文档是否始终把本轮成果表述为 provenance hardening，而不是 execution readiness
2. override-focused regression 是否足以覆盖 future-host 最常见的 config/path 迁移场景
