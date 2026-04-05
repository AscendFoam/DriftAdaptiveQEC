# CNN-FPGA-GKP TFLite 独立环境方案

## 1. 文档目标

本文件用于解决当前项目里一个非常具体的问题：

- 当前主开发环境是 `Python 3.13.5`
- 当前环境没有 `tensorflow`
- 当前环境也没有 `tflite_runtime`
- 因此 [export.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/export.py) 只能回退到 `.tflite.json` stub

如果后续要真正跑通：

1. 真实 `.tflite` 导出
2. `backend=tflite` 的真实解释执行
3. 为后续 FPGA/ARM 部署做更接近实机的软件验证

则建议单独准备一个与主开发环境隔离的 `Python 3.11` 或 `Python 3.10` 环境。

---

## 2. 先给结论

对当前这台 `Apple Silicon MacBook Air`，推荐优先采用：

1. 本机验证环境：`Python 3.11 + TensorFlow`
2. 不推荐在本机优先走：`tflite_runtime`
3. 如果以后确实需要“更接近板端”的最小 runtime：
   再单独准备 Linux 环境或按 LiteRT 官方流程源码构建 wheel

原因很直接：

1. 当前项目在 macOS 上的目标是“把真实 `.tflite` 跑通”
2. 官方 LiteRT 文档已说明：`tflite-runtime` 不再为 Windows 和 macOS 发布预编译 wheel
3. 因此在 macOS 上，本地验证最稳妥的路线是直接使用 `TensorFlow` 自带的 `tf.lite.Interpreter`

换句话说：

- 本机验证 `.tflite`：选 `TensorFlow`
- 未来轻量部署 runtime：再考虑 `tflite_runtime / LiteRT wheel / 板端服务`

---

## 3. 方案总览

建议把环境拆成两层：

### 3.1 环境 A：本机 `.tflite` 验证环境

用途：

- 导出真实 `.tflite`
- 在 macOS 本地验证 `backend=tflite`
- 跑 `run_hil_mode_benchmark`

建议配置：

- `Python 3.11`
- `tensorflow`
- `numpy`
- `pyyaml`

### 3.2 环境 B：板端/边缘部署环境

用途：

- 面向 Linux ARM / Linux x86 的轻量 runtime
- 尽量接近未来板端的软件结构

建议配置：

- `Python 3.10` 或 `Python 3.11`
- `tflite_runtime` 或 LiteRT wheel
- 若官方无预编译 wheel，则源码构建

当前阶段优先级：

- 先做环境 A
- 环境 B 暂时只做预案，不急着在这台 Mac 上硬啃

---

## 4. 为什么当前环境不适合直接跑

当前项目状态中，`.tflite` 没有真正跑起来，不是代码链路没接，而是运行环境不对。

具体表现为：

1. 当前 Python 版本过新

- 项目现在使用的是 `Python 3.13.5`
- 这不是当前项目里最稳妥的 TensorFlow/TFLite 验证版本

2. 当前环境缺依赖

- 没有 `tensorflow`
- 没有 `tflite_runtime`

3. 当前 macOS 本机不适合把 `tflite_runtime` 当成默认路线

- LiteRT 官方文档明确说明不再为 `Windows` 和 `macOS` 发布预编译 `tflite-runtime` wheel
- 所以在本机上强推 `tflite_runtime`，工程成本高、稳定性差、收益也不大

因此当前最合适的做法不是继续在 `Python 3.13` 上补依赖，而是：

- 直接新建一个隔离环境
- 使用 `Python 3.11`
- 用 `TensorFlow` 完成本机 `.tflite` 验证

---

## 5. 推荐方案 A：Conda + Python 3.11 + TensorFlow

结合本机环境检查结果：

- 本机已经有 `conda`
- 本机也有 `brew`

所以最推荐的实际落地方式是：

- 用 `conda` 创建独立环境
- 用 `pip` 安装 `tensorflow`

### 5.1 创建环境

```bash
conda create -n drift_tflite311 python=3.11 -y
conda activate drift_tflite311
python -m pip install --upgrade pip setuptools wheel
```

### 5.2 安装核心依赖

先安装最小依赖集：

```bash
pip install tensorflow numpy pyyaml
```

若后续要在该环境中直接跑完整 benchmark，再补常用依赖：

```bash
pip install matplotlib pandas scikit-learn
```

如果 `pip install tensorflow` 在你的镜像源里解析失败，再尝试：

```bash
pip install tensorflow-macos numpy pyyaml
```

这里的建议顺序是：

1. 先试 `tensorflow`
2. 失败再试 `tensorflow-macos`

原因是：

- 当前 TensorFlow 主包已经覆盖越来越多的平台组合
- 但在部分镜像环境或历史缓存环境中，`tensorflow-macos` 仍可能更容易装通

### 5.3 验证安装

```bash
python - <<'PY'
import tensorflow as tf
print("tf_version =", tf.__version__)
print("keras_ok =", hasattr(tf, "keras"))
print("tflite_interpreter_ok =", hasattr(tf.lite, "Interpreter"))
PY
```

通过标准：

1. 能正常打印 `tf_version`
2. `keras_ok = True`
3. `tflite_interpreter_ok = True`

---

## 6. 推荐方案 B：venv + Homebrew Python 3.11

如果你不想把 `.tflite` 验证环境混到 Conda 中，也可以走：

1. `brew install python@3.11`
2. 用 `venv` 建一个非常干净的环境

示例：

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv-tflite311
source .venv-tflite311/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install tensorflow numpy pyyaml
```

适用场景：

- 你希望这个环境尽量轻
- 你只想用来验证 `.tflite` 导出和解释执行

不太适用场景：

- 你后面想在这个环境里叠很多数据分析工具
- 你已经主要依赖 Conda 做环境管理

---

## 7. 本项目中的真实验证步骤

环境装好后，建议按下面顺序验证，而不是一上来直接跑最长 benchmark。

### 7.1 第一步：验证真实 `.tflite` 导出

在独立环境中运行：

```bash
python -m cnn_fpga.model.export \
  --config cnn_fpga/config/experiment_static_theta_v2.yaml \
  --model-path artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz
```

期望结果：

1. 输出文件后缀是 `.tflite`
2. 而不是 `.tflite.json`
3. 对应 export report 中的 `export_mode` 应为 `true_tflite`

如果仍输出 `.tflite.json`，说明：

- 环境里的 TensorFlow 还不可用
- 或者模型到 TFLite 的导出过程中仍有兼容性异常

### 7.2 第二步：验证 `.tflite` 单模型推理

建议先跑一个很小的自检脚本：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from cnn_fpga.runtime.inference_service import TFLiteHistogramPredictor

model_path = sorted(Path("artifacts/models/static_theta_v2").glob("*.tflite"))[-1]
predictor = TFLiteHistogramPredictor(
    model_path=model_path,
    label_names=("sigma", "mu_q", "mu_p", "theta_deg"),
)
hist = np.zeros((32, 32), dtype=np.float32)
pred = predictor.predict(hist)
print(pred.to_dict())
PY
```

通过标准：

1. 不触发 `tflite_runtime_unavailable`
2. `source` 不再是 `tflite_stub_service`
3. 能返回 4 个标签的预测值

### 7.3 第三步：验证 HIL mock 的 `tflite` 路径

确保 [hardware_hil.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/hardware_hil.yaml) 中：

- `int8_artifact_mock` 的 `inference_backend: tflite`
- `real_board` 的 `inference_backend: tflite`

然后运行：

```bash
python -m cnn_fpga.benchmark.run_hil_mode_benchmark --config cnn_fpga/config/hardware_hil.yaml
```

通过标准：

1. `int8_artifact_mock` 的 `artifact_path` 指向真实 `.tflite`
2. 不是 `.tflite.json`
3. benchmark 可以完整结束

### 7.4 第四步：再考虑板端服务化

只有前三步都通过后，才建议继续做：

- 慢回路独立 `tflite` 服务进程
- ARM 板端单独运行推理服务
- HIL 与真实板卡联调

---

## 8. 为什么当前不推荐把 `tflite_runtime` 作为 Mac 本机首选

原因不是它不好，而是它不适合当前这一步。

### 8.1 工程目标不匹配

当前真正需要解决的是：

- `.tflite` 是否能真实导出
- `backend=tflite` 是否能真实执行

这两个目标，用 `TensorFlow` 已经能覆盖。

### 8.2 平台支持不划算

LiteRT 官方文档已经说明：

- 不再为 `macOS` 提供预编译 `tflite-runtime` wheel

这意味着如果你在 Mac 上坚持走 `tflite_runtime`：

1. 要自己找第三方 wheel
2. 或者自己从源码构建
3. 排错成本会明显高于直接装 TensorFlow

### 8.3 当前阶段的最优策略

因此当前最优策略应写成：

1. 本机：`TensorFlow + tf.lite.Interpreter`
2. 板端：以后再做 `tflite_runtime / LiteRT` 轻量部署

---

## 9. 如果后续一定要最小 runtime，怎么做

如果后面你要把慢回路推理从本机搬到：

- Linux x86 工控机
- Linux ARM 边缘控制板
- FPGA 配套 ARM 核

那么再进入下面路线。

### 9.1 优先路线

优先找目标系统是否有官方可用 wheel：

- `tflite_runtime`
- 或 LiteRT Python wheel

如果有，就直接装 wheel，不要自己编。

### 9.2 无 wheel 时的路线

如果目标平台没有现成 wheel，再按 LiteRT 官方文档用 CMake/源码构建 wheel。

这一阶段再做是合理的，因为：

1. 目标平台已经明确
2. ABI、glibc、CPU 指令集都确定了
3. 编 wheel 的结果能直接用于真实部署

反过来，在当前这台 Mac 上先编一个只用于过渡验证的 `tflite_runtime`，收益不高。

---

## 10. 与当前代码的对应关系

独立环境就绪后，当前代码里会直接收益的入口有：

1. 真实 `.tflite` 导出

- [export.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/export.py)

2. 真实 `.tflite` 推理

- [inference_service.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/inference_service.py)

3. HIL 模式 benchmark

- [run_hil_mode_benchmark.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_hil_mode_benchmark.py)

4. P3 正式 HIL 会话

- [run_hil_suite.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_hil_suite.py)

---

## 11. 当前建议的执行顺序

建议你后续按这个顺序推进：

1. 创建 `conda` 的 `Python 3.11` 独立环境
2. 安装 `tensorflow`
3. 先验证 `.tflite` 导出不再回退为 stub
4. 再验证 `TFLiteHistogramPredictor`
5. 再跑 `run_hil_mode_benchmark`
6. 最后再考虑板端真实 runtime 与真实板卡联调

---

## 12. 参考资料

1. TensorFlow pip 安装文档  
   https://www.tensorflow.org/install/pip

2. LiteRT Python 指南  
   https://ai.google.dev/edge/litert/guide/python

3. TensorFlow PyPI 页面  
   https://pypi.org/project/tensorflow/

4. tensorflow-macos PyPI 页面  
   https://pypi.org/project/tensorflow-macos/

---

## 13. 最终建议

当前最稳妥、最省时间、最适合这台机器的方案是：

- 先建 `Python 3.11` 独立环境
- 在该环境里安装 `TensorFlow`
- 把真实 `.tflite` 导出和解释执行跑通

而不是：

- 继续在 `Python 3.13` 上补洞
- 或者优先在 macOS 上折腾 `tflite_runtime` 源码构建

这个选择更符合当前项目阶段目标，也更符合工程投入产出比。
