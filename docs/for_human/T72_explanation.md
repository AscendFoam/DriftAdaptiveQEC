# T72 说明与 Review 解释

## 1. T72 在补什么

`T71` 已经把 current-host 的真板 gate 做成了可 replay / regeneration 的 checked-in 包，但当时还留了一个细节问题：

- artifact 里有一些 provenance 说明，还是偏“默认文案”
- future-host 一旦换 config 或换设备路径，值虽然会变，说明却可能没同步

`T72` 做的就是把这层说明也收紧。

## 2. 这次没有把 verdict 变成更乐观

最重要的一点是：

- `T49` replay verdict = `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `T72` current-host regeneration verdict = `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

所以这次不是“放宽 gate”，而是“把同一个 `NO_GO` 说得更严谨”。

## 3. 现在 probe 记录为什么更可信

以前的问题是，artifact 里会直接写类似“access denied”，但 reviewer 不容易判断：

- 这条命令是不是这次真的跑过
- 还是沿用了上次任务的经验文案

现在会直接记录每条 probe 的状态：

- `ok`
- `command_failed`
- `not_applicable`

这就把“真实执行结果”和“平台不适用”分开了。

## 4. 为什么说 override 现在更干净

future-host 以后最常见的变化就是两类：

1. 换 `--config`
2. 换 `--mmio-path` / `--dma-path`

`T72` 之后，artifact 不只是把 effective value 写进去，还会保留：

- 原 config 值是什么
- 最终 effective value 是什么
- 这次是不是 CLI override

这样 reviewer 能直接看出：

- 这次是默认配置
- 还是人为覆盖过路径

## 5. `expected_byte_count_basis` 为什么也要改

如果它一直写死成：

- `32 x 32 float32 -> 4096 bytes`

那 future-host 一旦用了别的 histogram 规模或别的 dtype，artifact 就会变成“数字是新的，解释还是旧的”。

现在它会直接根据：

- `histogram_shape`
- `dtype`
- `buffer_bytes`

算出本次应该是多少字节，所以这个说明终于和实际 config 绑在一起了。

## 6. 为什么 `T37` 还是不能开

因为 `T72` 解决的不是硬件事实，而是 provenance 质量。

当前真正没解决的，还是：

- 没有能打开的真实板级 `mmio` / `dma` 路径
- 没有 bitstream / RTL / DMA / fixed-point contract 的板级绑定证据
- 仓库里的 `board_backend.py` 仍然是 placeholder 路径

所以这次之后，结论仍然只能是：

- transfer-pack 更严谨了
- 不是 current-host 真板 ready 了

## 7. 一句话总结

`T72` 的价值不是把项目“推到真板执行前一步”，而是避免 future-host 以后把默认文案误当成真实执行事实。
