# T72 Review

## Verdict

`PASS_WITH_WARNINGS`

核心任务已经完成：`probe_limitations` 不再写死固定文案，`source_records` / `repo_board_defaults` / `expected_byte_count_basis` 已改成 execution-derived / override-aware，新增了针对 `--config` / `--mmio-path` / `--dma-path` 的 focused regression，且 `T49` replay 与 `T72` current-host regeneration 仍稳定保持同一 `NO_GO` verdict。

## Captain Closeout Disposition

- Captain verdict: `PASS_WITH_WARNINGS`
- Warning classification:
  - `N1` 最小 config 场景下 path provenance 仍会把代码默认值写成 `source_kind=config_field` = `deferred -> R32`
  - `N2` Worker 原始主报告路径曾短暂落在任务包精确 allowed files 之外，但当前 `HEAD` 已整理回允许目录 = `accepted`
  - `N3` 缺少覆盖“YAML 未显式提供 path 字段时的 provenance 回退标签” focused regression = `deferred -> R32`
- 合并说明：
  - `Suspicious implementation details` 与 `Non-blocking issues` 的最小 config provenance 过强表述属于同一问题，按 `N1` 合并处理
- 收口结论：
  - `T72` 已完成并可切离当前唯一任务
  - `R31` 可视为已由 `T72` 收口
  - 新的残余风险收敛为 `R32`，仅针对 future-host 最小 config 场景下的 provenance 标签精确性

我复核了以下轻量证据：

- `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
- `python -m unittest tests.test_t72_real_board_transfer_pack_provenance_hardening`
- 一次当前宿主 collector 再生成
- 一次当前宿主 gate helper 再构建
- 一次 `T49` checked-in artifact replay 对比

## Blocking issues

- 无

## Non-blocking issues

- 路径 provenance 仍然不能区分“YAML 明确写了这个字段”和“`BoardFPGAConfig.from_config()` 用代码默认值补出来的字段”。`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py:518-531` 用 `board_cfg.mmio.path` / `board_cfg.dma.path` 生成 `candidate_*_path_record`，而 `cnn_fpga/hwio/board_backend.py:69-78` 会在字段缺失时回退到 `/dev/uio0` / `/dev/uio1`。我用一个临时 config 去掉这两个键后复核，artifact 仍会写成 `source_kind=config_field`。这不影响 T72 明确覆盖的默认 config 和 CLI override 场景，但 future-host 的“最小 config”场景还不算完全 provenance-clean。
- Worker 原始提交 `4b5b6a7` 最初把主报告落在 `docs/t72_real_board_transfer_pack_provenance_hardening.md`，不在 T72 任务包列出的精确 allowed files 里。当前 `HEAD` 已通过后续文档整理把等价内容归档到 `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`，所以我不据此 blocking，但原始提交边界并非完全严丝合缝。

## Missing tests

- 缺少一个专门覆盖“config 缺省 `hil.board_io.axi_uio_path` / `hil.board_io.dma_buffer_path` 时如何记录 provenance”的回归测试。现有测试覆盖了默认 config、显式自定义 config、CLI override 和 verdict stability，但没有覆盖代码默认值回退这个边角场景。

## Suspicious implementation details

- `candidate_mmio_path_record.config_value` 和 `candidate_dma_path_record.config_value` 取自 `BoardFPGAConfig` 归一化后的值，而不是原始 YAML 节点是否存在的事实。这让“值本身”是对的，但让“来源标签”在字段缺失场景下比代码真正能证明的更强。

## Recommended next action

- Captain 可以把 `T72` 视为 `R31` 的直接收口候选，并继续保持 `T37` blocked。
- closeout 时要继续把结果写成“read-only real-board gate / transfer-pack provenance hardening”，不能写成 real-board ready、real-board execution success、`hardware_validated` 或 deployment closure。
- 如果后续确实关心 future-host 的最小 config 迁移，建议补一个很小的 follow-up：给 path provenance 增加 `config_field_present` / `source_kind=code_default` 一类标记，并补一条对应回归测试。
