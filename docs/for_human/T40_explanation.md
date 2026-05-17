# T40 说明：在干净 CPU 环境里真正跑通一次最小训练

这次 T40 做的事情很克制：没有去改主训练配置，也没有把结果写进历史正式产物目录，而是在 `T39` 已经建好的干净 Python 3.12 CPU-only 环境里，真实执行了一次最小规模训练。

这次训练复用了 `static_theta_v2` 现有数据输入，只新建了一个临时派生配置 `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml`。这个配置只做三类改动：

- 把输出目录重定向到 `artifacts/t40_train_smoke/...`
- 把训练样本裁到 `1024/256`
- 把 epoch/patience 压到 `3/2`

真实执行结果是成功的，训练报告里记录的 backend 是 `numpy`，device 是 `cpu`。这说明我们现在至少可以诚实地说：

- `T39` 的 clean env 不只是能 `--help` 或 dry-run
- 它已经能完成一次真正的 CPU-only `tiny_cnn` 训练 smoke
- 而且没有污染 canonical `artifacts/models/static_theta_v2/`、`artifacts/reports/static_theta_v2/` 或 `runs/`

但这仍然不是“训练链已经完全可复现”的结论。它还不能证明：

- 更大规模训练也稳定
- Linux 也一样
- GPU/CUDA 路径没问题
- `.tflite`、benchmark、真板路径已经被验证

所以更准确的说法是：`R11` 又收窄了一步，clean CPU 环境已经从“只会 dry-run/import”推进到了“能真实跑通一次最小训练 smoke”，但还没有被关闭。
