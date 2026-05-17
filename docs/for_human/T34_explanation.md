# T34 说明

这次不是写论文正文，也不是补新实验。

我做的是一份“论文 claim 和证据台账”：

- 哪些话现在可以写
- 哪些话只能带边界地写
- 哪些话现在还绝对不能写

重点是把几条最容易被误写强的边界钉死：

- `mock-backed software HIL` 不是 `real-board`
- `.tflite` stub / 入口存在 不是 真正 runtime 已验证
- `T40` 的一次 clean CPU 训练 smoke 不是 full reproducibility
- `statcalib` 的接口合同存在 不是 benchmark comparator 已有结果

后面如果要开始真正收论文，这份台账可以直接当“用词护栏”。
