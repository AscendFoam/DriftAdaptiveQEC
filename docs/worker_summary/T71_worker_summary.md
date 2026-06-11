# T71 Worker Summary

## 改了什么

本轮只改了 `T71` 允许路径：

- 更新 helper：`cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
  - `device_path_truth` 改为 role-aware `mmio + dma` 判定
- 新增 collector：`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- 更新 helper tests：`tests/test_t49_real_board_smoke_execution_gate.py`
- 新增 regeneration/replay tests：`tests/test_t71_real_board_gate_regeneration_pack.py`
- 写入 task-scoped outputs：`artifacts/t71_real_board_gate_regeneration_pack/`
- 新增主报告：`docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
- 新增 review 草稿：`docs/review/T71_review.md`
- 新增人类解释：`docs/for_human/T71_explanation.md`
- 新增本文件：`docs/worker_summary/T71_worker_summary.md`

## 如何验证

实际执行并记录了：

1. `python -m unittest tests.test_t49_real_board_smoke_execution_gate`
2. `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
3. 一次当前宿主只读 artifact 再生成
   - `python -m cnn_fpga.hwio.collect_t71_real_board_gate_artifacts --output-dir artifacts/t71_real_board_gate_regeneration_pack`
4. 一次用 `T71` 再生成 artifact 驱动 gate helper 的真实执行
   - `current_host_regenerated_gate.json`
5. 一次用 `T49` checked-in artifact 驱动 gate helper 的 replay 执行
   - `t49_checked_in_replay_gate.json`
6. 一次 replay vs regeneration comparison
   - `replay_vs_regeneration_comparison.json`

本轮最终 fresh verification 还包含：

7. `python -m py_compile cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
8. `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
9. 边界检查
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

## 关键结果

### role-aware 加固后，当前宿主 verdict 是否变化

没有变化。

- `T49` replay verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `T71` current-host regeneration verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

### replay 与 regeneration 是否一致

一致。

- `verdict_match = true`
- `strongest_statement_match = true`
- `device_path_truth_status_match = true`
- `bitstream_truth_status_match = true`
- `repo_execution_path_truth_status_match = true`

### future-host 推荐入口命令

1. 收集只读 artifacts

```powershell
python -m cnn_fpga.hwio.collect_t71_real_board_gate_artifacts --output-dir <future_host_pack_dir> --mmio-path <MMIO_PATH> --dma-path <DMA_PATH>
```

2. 聚合 gate verdict

```powershell
python -m cnn_fpga.hwio.build_t49_real_board_smoke_gate --host-fact-manifest-json <future_host_pack_dir>/host_fact_manifest.json --device-path-probe-json <future_host_pack_dir>/device_path_probe.json --code-side-audit-json <future_host_pack_dir>/code_side_audit.json --output-json <future_host_pack_dir>/real_board_gate.json
```

### T37 当前仍被哪些层阻塞

1. `device_path_truth`
   - 当前宿主没有可读打开的 `mmio + dma` 双角色路径
2. `bitstream_and_contract_truth`
   - 没有当前宿主绑定的 bitstream / RTL / DMA contract 证据
3. `repo_execution_path_truth`
   - `board_backend.py` 仍是 placeholder-only

## 剩余风险

1. `T71` 解决的是 gate 可再生成性，不是硬件前提本身；所以 future-host 仍然需要真实设备路径、bitstream 绑定和非-placeholder repo path 才能继续。
2. collector 已经提供 checked-in read-only 入口，但在 Linux future-host 上对驱动线索的收集仍主要依赖 `lspci/lsmod` 可用性；若目标主机没有这些命令，artifact 仍会生成，但 clue 字段可能退化为 command-error 记录。
3. `T49` 历史 artifact 没有被改写；本轮 replay 输出会新增 role-aware 字段（如 `openable_mmio_paths` / `missing_roles`），后续解读时应明确它们属于 T71 helper replay 结果，而不是历史 T49 gate JSON 被篡改。
