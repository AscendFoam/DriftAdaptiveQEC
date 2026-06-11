# T72 真板 Transfer-Pack Provenance 加固

## 结论先行

`T72` 没有改变 `T49` / `T71` 的 current-host 最终 gate verdict。

- `T49` checked-in artifact replay verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `T72` current-host regeneration verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

`T72` 解决的是 transfer-pack 的 provenance 严谨度，不是真板执行条件本身。

## 本轮加固了什么

### 1. `probe_limitations` 不再写死“当前事实”

`T71` 的问题是：有几条 probe 根本没有在当前执行里真正跑过，但 artifact 里已经被写成固定的 `access denied` 叙事。

`T72` 之后，`host_fact_manifest.json` 里新增并依赖：

- `probe_execution_records`
- 结构化的 `probe_limitations`

现在每条 probe 都会显式区分：

- `status=ok`
- `status=command_failed`
- `status=not_applicable`

当前 Windows 宿主上的真实结果是：

- `windows_cmd_ver = ok`
- `windows_get_ciminstance_win32_operatingsystem = command_failed`
- `windows_get_ciminstance_win32_computersystem = command_failed`
- `windows_get_pnpdevice_presentonly = command_failed`
- `windows_systeminfo = command_failed`
- `linux_lspci_nn = not_applicable`
- `linux_lsmod = not_applicable`

因此 reviewer 现在可以分清：

- 哪些命令真的执行过
- 哪些命令执行失败
- 哪些命令在当前平台根本不该执行

而不是再把“旧任务观察到的限制”伪装成“这次 regeneration 重新探测出来的事实”。

### 2. config/path provenance 变成 execution-derived / override-aware

`host_fact_manifest.json` 的 `repo_board_defaults` 现在不只是给出值，还会给出来源：

- `config_path`
- `config_argument_kind`
- `board_source_record`
- `bitstream_version_source_record`
- `candidate_mmio_path_record`
- `candidate_dma_path_record`

其中：

- `config_argument_kind=default` 表示这次使用默认 `cnn_fpga/config/hardware_hil.yaml`
- 若 future-host 用 `--config <other.yaml>`，则会变成 `override`
- 若 future-host 用 `--mmio-path` / `--dma-path`，对应的 `candidate_*_path_record.source_kind` 会变成 `cli_override`

`bitstream_evidence.source_records` 也不再写死 `hardware_hil.yaml` 的旧字符串，而是会反映本次实际使用的 config 路径和字段值。

### 3. `expected_byte_count_basis` 变成运行时推导

`code_side_audit.json` 的 `dma_contract.expected_byte_count_basis` 现在不再是固定文案：

- 旧口径：`32 x 32 float32 histogram -> 4096 bytes under current config defaults`

现在它会根据本次 collector 读到的：

- `histogram_shape`
- `dtype`
- `buffer_bytes`

动态推导，例如当前默认 config 会得到：

- `histogram_shape = [32, 32]`
- `dtype = float32`
- `dtype_itemsize_bytes = 4`
- `computed_byte_count = 4096`
- `matches_configured_buffer_bytes = true`
- `formula = 32 x 32 x 4 bytes-per-float32 = 4096 bytes`

这意味着 future-host 如果切到别的 config，artifact 中的 byte-count provenance 也会跟着变，而不是继续复述默认配置的话术。

## `--config` / `--mmio-path` / `--dma-path` 现在如何体现在 artifact 中

本轮新增的 focused regression 已覆盖两类 override：

1. `--config` override
   - `bitstream_evidence.source_records[*].config_path` 跟随新 config
   - `repo_board_defaults.config_path` / `config_argument_kind` 跟随新 config
   - `expected_byte_count_basis` 跟随新 `histogram_shape` / `dtype` / `buffer_bytes`

2. `--mmio-path` / `--dma-path` override
   - `repo_board_defaults.candidate_mmio_path` / `candidate_dma_path` 反映 effective path
   - `candidate_mmio_path_record` / `candidate_dma_path_record` 区分 `config_value` 与 `override_value`
   - `device_path_probe.json` 前两个 candidate 的 `source` 会从 `config_*` 切换成 `cli_override_*`

因此，future-host 现在不只是“值变了”，而是 provenance 也会一起变。

## 回放与再生成是否仍一致

一致。

对比文件：

- `artifacts/t72_real_board_transfer_pack_provenance_hardening/current_host_regenerated_gate.json`
- `artifacts/t72_real_board_transfer_pack_provenance_hardening/t49_checked_in_replay_gate.json`
- `artifacts/t72_real_board_transfer_pack_provenance_hardening/replay_vs_regeneration_comparison.json`

关键结论：

- `verdict_match = true`
- `strongest_statement_match = true`
- `device_path_truth_status_match = true`
- `bitstream_truth_status_match = true`
- `repo_execution_path_truth_status_match = true`

也就是说，`T72` 收紧 provenance 之后，没有把已有的 honest `NO_GO` 写漂移。

## 为什么 `T37` 仍然 blocked

`T37` 继续 blocked，不是因为 transfer-pack 不够整洁，而是因为真板执行前提本身仍未闭环：

1. `device_path_truth` 仍然是 `not_ready`
   - 当前宿主没有可只读打开的 `mmio` 与 `dma` 真板设备路径
2. `bitstream_and_contract_truth` 仍然是 `not_ready`
   - 缺 bitstream alignment、RTL 地址表绑定、DMA contract、固定点 contract 这四类外部事实
3. `repo_execution_path_truth` 仍然是 `placeholder_only`
   - `board_backend.py` 仍是 placeholder 语义，不是已验证的真实板级执行路径

因此，`T72` 只能证明 transfer-pack 更严谨了，不能证明真板 ready 了。

## strongest supported claim

`T72` 当前只能支持如下表述：

“仓库当前已有一套 checked-in、只读、role-aware、可 replay / regeneration 的 real-board gate 包；`T72` 进一步把其中的 provenance 从默认文案推进为更贴近实际执行上下文的动态导出，并补齐了 override 相关回归。当前 Windows 宿主再生成后的 verdict 仍然是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。这说明 mainline 进一步收紧了 future-host transfer-pack 的严谨度，但这仍不等于真板执行成功、real-board validation、`P3 real-board HIL complete` 或 deployment closure。”

## 仍然不能支持的说法

`T72` 仍然不能支持：

- `real-board smoke executed successfully`
- `T37 ready to execute on current host`
- `real-board host-transfer pack fully production ready`
- `hardware_validated`
- `P3 real-board HIL complete`
- `deployment closure`
