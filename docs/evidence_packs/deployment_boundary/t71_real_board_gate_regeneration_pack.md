# T71 真板 Gate 再生成加固与宿主迁移包

## 结论

`T71` 没有改变 `T49` 当前宿主的最终 gate verdict。

- `T49` checked-in artifact replay verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `T71` 当前宿主再生成 verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

这说明本轮完成的是 gate 路径的可再生成化与 role-aware 加固，不是真板执行推进，也不是真板 ready。

## 本轮实际做了什么

本轮只在允许范围内完成了四件事：

1. 把 `build_t49_real_board_smoke_gate.py` 的 `device_path_truth` 判定改成 role-aware
   - 现在必须同时满足：
     - 至少 `1` 条 `role=mmio` 且 `read_only_openable=true`
     - 至少 `1` 条 `role=dma` 且 `read_only_openable=true`
   - 不再接受“只要 openable path 总数 >= 2 就算 ready”
2. 新增 checked-in 的只读 artifact collector
   - `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
3. 补齐 focused regression
   - 同角色双路径不得误判 ready
   - `T49` checked-in artifact replay verdict 不得漂移
   - `T71` current-host regeneration verdict 必须与 `T49` replay 一致
4. 产出 current-host regeneration pack
   - `artifacts/t71_real_board_gate_regeneration_pack/`

## T49 replay 与 T71 regeneration 是否一致

一致。

本轮生成的比较文件：

- `artifacts/t71_real_board_gate_regeneration_pack/replay_vs_regeneration_comparison.json`

其结果为：

- `verdict_match = true`
- `strongest_statement_match = true`
- `device_path_truth_status_match = true`
- `bitstream_truth_status_match = true`
- `repo_execution_path_truth_status_match = true`

允许存在的差异只应当是：

- `generated_at_utc` 时间戳
- 宿主侧 clue 顺序或轻微环境噪声

本轮没有出现 verdict 漂移，也没有出现 strongest supported claim 漂移。

## role-aware 加固后，当前宿主 verdict 是否变化

没有变化。

变化的是 device 层的判定口径更严格了：

- 旧逻辑：只按 `openable_paths >= 2`
- 新逻辑：必须 `openable_mmio_paths >= 1` 且 `openable_dma_paths >= 1`

但当前 Windows 宿主本来就是：

- `openable_mmio_paths = 0`
- `openable_dma_paths = 0`

所以 current-host 结论仍然是：

`NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

## future-host 现在应该使用哪一个 checked-in 入口

推荐 future-host 使用两步命令：

### 1. 只读收集 gate artifacts

```powershell
python -m cnn_fpga.hwio.collect_t71_real_board_gate_artifacts --output-dir <future_host_pack_dir> --mmio-path <MMIO_PATH> --dma-path <DMA_PATH>
```

说明：

- `--mmio-path` 和 `--dma-path` 是为了 future-host 不必改仓库配置文件，也能把真实候选设备路径带入 probe
- 该命令只做 host/device/code-side 的只读收集，不做任何 MMIO/DMA/register 写入

### 2. 用 checked-in gate helper 聚合 verdict

```powershell
python -m cnn_fpga.hwio.build_t49_real_board_smoke_gate --host-fact-manifest-json <future_host_pack_dir>/host_fact_manifest.json --device-path-probe-json <future_host_pack_dir>/device_path_probe.json --code-side-audit-json <future_host_pack_dir>/code_side_audit.json --output-json <future_host_pack_dir>/real_board_gate.json
```

这就是当前推荐的 future-host host-transfer 入口。

## 当前仍未闭环的是哪几层

当前 `T37` 仍然不能开工，因为以下三层都还没闭环：

1. `device_path_truth`
   - 当前 Windows 宿主没有任何可读打开的 `mmio + dma` 双角色路径组合
2. `bitstream_and_contract_truth`
   - 只有 config-level `fpga_linear_v1` 记录
   - 没有当前宿主绑定的 bitstream 文件
   - 没有 RTL 地址表 / DMA contract / Q4.20 板侧一致性确认
3. `repo_execution_path_truth`
   - `board_backend.py` 仍是 placeholder-only
   - `schedule_commit()` 占位返回仍未升级
   - `step()` 仍不产出可验收的真板事件流

## 为什么 T37 现在仍然不能开工

因为 `T71` 解决的是“gate 能否被未来候选宿主再生成和复核”，不是“当前宿主是否已经具备真板执行条件”。

更具体地说：

- `T49` 已经证明当前宿主是 honest `NO_GO`
- `T71` 把这个 `NO_GO` 路径加固成了 checked-in、role-aware、可 replay / 可 regeneration 的 read-only gate 包
- 但这不等于当前宿主突然有了真实设备节点
- 也不等于 bitstream / RTL / DMA contract 被补齐
- 更不等于 repo 的 `board_backend.py` 不再 placeholder

所以 `T37` 继续保持 blocked 是正确结果，而不是失败。

## strongest supported claim

`T71` 当前只能支持如下表述：

“仓库现在有一个 checked-in、只读、可在候选宿主再生成的 real-board gate 入口；当前这台 Windows 宿主重放后仍然是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。这说明 mainline 已经把 current-host 真板前提检查从一次性 task 结果，提升成了可再生成的 gate 包，但这仍不等于真板执行成功、real-board validation、P3 real-board HIL complete 或 deployment closure。”
