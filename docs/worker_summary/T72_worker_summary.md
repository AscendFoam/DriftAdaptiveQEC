# T72 Worker Summary

## 改了什么

本轮只改了 `T72` 允许路径：

- 更新 collector：`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- 新增 T72 focused regression：`tests/test_t72_real_board_transfer_pack_provenance_hardening.py`
- 写入 task-scoped outputs：`artifacts/t72_real_board_transfer_pack_provenance_hardening/`
- 新增主报告：`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
- 新增 review 草稿：`docs/review/T72_review.md`
- 新增人类解释：`docs/for_human/T72_explanation.md`
- 新增本文件：`docs/worker_summary/T72_worker_summary.md`
- 更新任务包回执：`docs/tasks/Phase2/T72_real_board_transfer_pack_provenance_hardening.md`

collector 本轮的核心变化有三类：

1. `probe_execution_records` / `probe_limitations` 改成结构化 probe provenance，区分 `ok`、`command_failed`、`not_applicable`
2. `repo_board_defaults` 与 `bitstream_evidence.source_records` 改成 execution-derived / override-aware
3. `dma_contract.expected_byte_count_basis` 改成由 `histogram_shape`、`dtype`、`buffer_bytes` 动态推导

## 如何验证

实际执行了以下验证：

1. `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
2. `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
3. `python -m unittest tests.test_t72_real_board_transfer_pack_provenance_hardening`
4. `python -m cnn_fpga.hwio.collect_t71_real_board_gate_artifacts --output-dir artifacts/t72_real_board_transfer_pack_provenance_hardening`
5. 一次用 `T72` artifact 驱动 gate helper 的真实执行
6. 一次用 `T49` checked-in artifact 驱动 gate helper 的 replay 执行
7. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- cnn_fpga/config`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

override-focused evidence 来自：

- `tests/test_t72_real_board_transfer_pack_provenance_hardening.py::test_config_override_updates_dynamic_provenance`
- `tests/test_t72_real_board_transfer_pack_provenance_hardening.py::test_path_overrides_are_reflected_in_probe_and_provenance`

关键结果：

- `T49` replay verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `T72` current-host regeneration verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `replay_vs_regeneration_comparison.json`：
  - `verdict_match = true`
  - `strongest_statement_match = true`

## 剩余风险

1. `T72` 收紧的是 provenance，不是硬件事实；所以 `T37` 仍然受 `device_path_truth`、`bitstream_and_contract_truth`、`repo_execution_path_truth` 三层阻塞。
2. 当前 collector 现在已经能区分 `command_failed` 与 `not_applicable`，但这并不等于 future-host 一定会给出更强结论；如果目标宿主仍缺真实设备节点或板级绑定证据，artifact 依然会诚实地停在 `NO_GO`。
3. `board_backend.py` 仍然是 placeholder-only，当前所有 real-board carry-forward 产物都只能被解读为 read-only gate / transfer-pack 证据，而不能被解读为真板执行成功。
