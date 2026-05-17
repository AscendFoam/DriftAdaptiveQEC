# T35 说明

这次不是写正式论文正文，也不是补实验。

本轮只做两件事：

1. 先把论文写作骨架搭出来，明确每一节现在只能引用哪些 claim、图、表。
2. 先从审稿人视角把最容易被质疑的点列清楚，避免后续写作把 mock、smoke、readiness 误写成完成态。

最重要的边界没有变：

- `mock-backed software HIL` 不能写成真板验证
- `.tflite` 入口或 stub 不能写成真实运行时验证
- 一次 clean CPU smoke 不能写成完整可复现训练链
- frozen-set formal revalidation 不能写成广泛 paper-grade benchmark
- statcalib interface contract 不能写成集成 comparator 结果

这份输出的作用，是给后续写 paper 的人一个不会越界的模板，而不是提前把论文结论写强。
