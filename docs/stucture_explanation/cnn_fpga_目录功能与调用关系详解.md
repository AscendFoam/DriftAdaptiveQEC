# cnn_fpga 目录代码功能与调用关系详解

本文档目标：
1. 系统梳理 `cnn_fpga/` 目录内各文件的职责与边界。
2. 逐函数解释输入、输出、主要逻辑与被调用关系。
3. 梳理配置文件字段如何映射到代码行为。
4. 给出端到端执行链路，帮助你调试与扩展。

适用范围：
- 目录：`cnn_fpga/`
- 覆盖文件：
  - Python 模块文件（`__init__.py`、`utils`、`data`、`model`、`benchmark`）
  - YAML 配置模板（`config/*.yaml`）

推荐阅读顺序：
1. 先看“第 1 章 目录定位”和“第 2 章 依赖全景”。
2. 再看“第 4 章 utils/config.py”。
3. 然后按 `data -> model -> benchmark` 顺序看。
4. 最后看“第 12 章 配置字段到代码路径映射总表”。

---

## 0. 目录结构速览

`cnn_fpga/` 当前结构：
- `__init__.py`
- `utils/`
  - `__init__.py`
  - `config.py`
- `data/`
  - `__init__.py`
  - `dataset_builder.py`
- `model/`
  - `__init__.py`
  - `train.py`
  - `evaluate.py`
  - `quantize.py`
- `benchmark/`
  - `__init__.py`
  - `run_drift_suite.py`
  - `run_hil_suite.py`
- `config/`
  - `experiment_static.yaml`
  - `experiment_drift.yaml`
  - `hardware_emulation.yaml`
  - `hardware_hil.yaml`

一句话定位：
- 这是“工程化脚手架层”，目的是先把数据、训练、评估、量化、漂移实验、HIL 骨架打通。

---

## 1. 总体架构定位

### 1.1 与 `physics/` 的关系

角色分工：
1. `physics/` 负责物理过程与逻辑错误统计。
2. `cnn_fpga/` 负责工程流程组织、配置驱动和产物管理。

关键耦合点：
1. `data/dataset_builder.py` 引入 `physics.gkp_state.LATTICE_CONST`。
2. `benchmark/run_drift_suite.py` 引入 `physics` 的测量/解码/追踪组件。

### 1.2 与 `docs/` 的关系

- 该目录是实验方案落地代码对应层。
- YAML 文件与文档中的默认参数约定保持一致（如 `T_fast=5us`、`window_size=2048`、`T_slow_update=20ms`）。

### 1.3 当前工程阶段定位

从代码内容看：
1. 训练模型是线性回归 baseline（不是 CNN 正式网络）。
2. 量化是 int8 对称量化 baseline（不是硬件最终量化链）。
3. HIL 是 mock backend（不是实板驱动）。

这意味着：
- 现在重点是“链路可执行、接口稳定、输出可复现”。

---

## 2. 依赖全景图（静态 import 视角）

### 2.1 根包层

`cnn_fpga/__init__.py`
- 不依赖内部子模块。
- 仅定义版本与说明。

### 2.2 工具层

`cnn_fpga/utils/config.py`
- 纯工具模块。
- 被几乎所有 CLI 脚本依赖。

### 2.3 数据层

`cnn_fpga/data/dataset_builder.py`
- 依赖 `cnn_fpga.utils.config`。
- 依赖 `physics.gkp_state.LATTICE_CONST`。

### 2.4 模型层

`cnn_fpga/model/train.py`
- 依赖 `cnn_fpga.utils.config`。

`cnn_fpga/model/evaluate.py`
- 依赖 `cnn_fpga.utils.config`。

`cnn_fpga/model/quantize.py`
- 依赖 `cnn_fpga.utils.config`。

### 2.5 benchmark 层

`cnn_fpga/benchmark/run_drift_suite.py`
- 依赖 `cnn_fpga.utils.config`。
- 依赖 `physics.error_correction.LinearDecoder`。
- 依赖 `physics.gkp_state.LATTICE_CONST`。
- 依赖 `physics.logical_tracking.LogicalErrorTracker`。
- 依赖 `physics.syndrome_measurement.MeasurementConfig`、`RealisticSyndromeMeasurement`。

`cnn_fpga/benchmark/run_hil_suite.py`
- 依赖 `cnn_fpga.utils.config`。
- 不依赖 `physics`。

### 2.6 配置层

`cnn_fpga/config/*.yaml`
- 被以上 CLI 脚本通过 `load_yaml_config` 解析。

---

## 3. 入口脚本总览（你最常执行的命令）

数据集生成入口：
- `python -m cnn_fpga.data.dataset_builder --config cnn_fpga/config/experiment_static.yaml`

训练入口：
- `python -m cnn_fpga.model.train --config cnn_fpga/config/experiment_static.yaml`

评估入口：
- `python -m cnn_fpga.model.evaluate --config cnn_fpga/config/experiment_static.yaml`

量化入口：
- `python -m cnn_fpga.model.quantize --config cnn_fpga/config/experiment_static.yaml`

漂移对比入口：
- `python -m cnn_fpga.benchmark.run_drift_suite --config cnn_fpga/config/experiment_drift.yaml`

HIL 脚手架入口：
- `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil.yaml`

---

## 4. 文件详解：`cnn_fpga/__init__.py`

文件作用：
1. 声明包级说明。
2. 暴露 `__version__`。

细节：
- `__all__ = ["__version__"]`。
- `__version__ = "0.1.0"`。

调用关系：
- 无运行时主流程逻辑。

---

## 5. 文件详解：`cnn_fpga/utils/__init__.py`

文件作用：
- 子包说明文件。

调用关系：
- 无函数定义。

价值：
- 给目录阅读者一个语义标签：`utils` 存放通用辅助函数。

---

## 6. 文件详解：`cnn_fpga/utils/config.py`

### 6.1 文件定位

这是整个 `cnn_fpga` 的“基础设施模块”。

核心职责：
1. 读 YAML。
2. 生成配置哈希。
3. 管理路径与目录创建。
4. 保存 JSON 结果。
5. 在无 `PyYAML` 时提供 fallback 解析器。

被依赖程度：
- 几乎所有 CLI 脚本都调用此文件。

### 6.2 函数：`load_yaml_config`

输入：
- `path: str | Path`

流程：
1. 解析绝对路径并检查是否存在。
2. 尝试 `import yaml`。
3. 成功则 `yaml.safe_load`。
4. 失败则走 `_load_yaml_fallback(content)`。
5. 校验根对象必须是 `dict`。

输出：
- 配置字典。

调用方：
1. `data/dataset_builder.py`
2. `model/train.py`
3. `model/evaluate.py`
4. `model/quantize.py`
5. `benchmark/run_drift_suite.py`
6. `benchmark/run_hil_suite.py`

### 6.3 函数：`config_hash`

输入：
1. `config` 字典。
2. `length`（默认 12）。

逻辑：
1. `json.dumps(sort_keys=True)`。
2. `sha256`。
3. 截断前 `length` 位。

用途：
- 为 run 目录、模型名、报告名提供稳定识别符。

### 6.4 函数：`now_tag`

作用：
- 返回时间戳字符串 `YYYYMMDD_HHMMSS`。

使用场景：
1. 训练 run_name。
2. 评估 report 名。
3. 量化产物名。
4. benchmark 运行目录名。

### 6.5 函数：`ensure_dir`

行为：
1. 路径展开并 resolve。
2. `mkdir(parents=True, exist_ok=True)`。
3. 返回 `Path`。

用途：
- 所有输出目录创建统一通过此函数。

### 6.6 函数：`get_path`

输入：
1. `config`
2. `key`
3. `default`

逻辑：
- 读取 `config['paths'][key]`，若无则回退 `default`。

特点：
- 统一了“路径键读取”的写法。

### 6.7 函数：`save_json`

行为：
1. 解析输出路径。
2. 自动创建父目录。
3. UTF-8 + `indent=2` 写盘。

用途：
- run summary / manifest / report 落盘。

### 6.8 fallback 解析器函数组

函数列表：
1. `_strip_comment`
2. `_to_tokens`
3. `_parse_scalar`
4. `_parse_block`
5. `_load_yaml_fallback`

#### `_strip_comment`
- 删除 `#` 后注释（不做复杂字符串内注释判定）。

#### `_to_tokens`
- 把文本行转成 `(indent, content)` token。
- 限制缩进为 2 空格倍数。

#### `_parse_scalar`
- 支持布尔、null、引号字符串、列表、数字、普通字符串。

#### `_parse_block`
- 递归解析 map/list 块。
- 是 fallback 解析核心。

#### `_load_yaml_fallback`
- 入口函数，返回最终字典。

### 6.9 为什么 fallback 有价值

1. 在轻量环境不装 `PyYAML` 也能跑脚手架。
2. 保证 CI 或简化环境不因依赖缺失中断。

### 6.10 已知限制

1. 不是完整 YAML 规范实现。
2. 对复杂语法支持有限。
3. 适合当前模板风格配置，不适合高度复杂 YAML。

### 6.11 你阅读时可重点看两段

第一段：
- `load_yaml_config` 主入口。

第二段：
- `_parse_block` 递归逻辑。

---

## 7. 文件详解：`cnn_fpga/data/__init__.py`

作用：
- 子包说明。

实际逻辑：
- 无函数与类定义。

---

## 8. 文件详解：`cnn_fpga/data/dataset_builder.py`

### 8.1 文件定位

角色：
- 生成合成直方图数据集。

输入：
- YAML 配置。

输出：
1. `train.npz`
2. `val.npz`
3. `test.npz`
4. `manifest.json`

标签定义：
- `LABEL_NAMES = ("sigma", "mu_q", "mu_p", "theta_deg")`

### 8.2 函数：`_sample_param`

作用：
- 在给定区间均匀采样一个参数。

输入：
1. `rng`
2. `bounds=(min,max)`

输出：
- `float`。

调用者：
- `_build_dataset`。

### 8.3 函数：`_wrap_to_fundamental`

作用：
- 把点映射到基本晶胞 `[-lambda/2,+lambda/2]`。

公式：
- `mod(value + lattice/2, lattice) - lattice/2`

调用者：
- `_histogram_from_params`。

### 8.4 函数：`_histogram_from_params`

输入：
1. `rng`
2. `sigma`
3. `mu_q`
4. `mu_p`
5. `theta_deg`
6. `points_per_sample`
7. `bins`
8. `lattice`

内部流程：
1. 把 `theta_deg` 转弧度，构造旋转矩阵。
2. 采样高斯点云 `base`。
3. 做旋转得到 `rotated`。
4. 加偏置得到 `displaced`。
5. wrap 回晶胞得到 `wrapped`。
6. `np.histogram2d` 统计二维直方图。
7. 归一化 `hist / hist_sum`。
8. 输出 `float32`。

输出：
- 单个 `bins x bins` 直方图样本。

### 8.5 函数：`_build_dataset`

输入：
1. `config`
2. `rng`

流程：
1. 从 `config['dataset']` 读取数据规模与参数范围。
2. 初始化 `histograms` 与 `labels` 大数组。
3. 外层循环采样配置级参数。
4. 内层循环在同配置下生成多个样本。
5. 每样本调用 `_histogram_from_params`。

输出：
- `(histograms, labels)`。

样本组织方式：
- “配置级参数 + 多样本复用”，能显式构造同参数族的样本簇。

### 8.6 函数：`_split_indices`

作用：
- 按随机索引分 train/val/test。

输入：
1. `n_samples`
2. `train_ratio`
3. `val_ratio`
4. `rng`

输出：
- `train_idx, val_idx, test_idx`。

### 8.7 函数：`_save_split`

行为：
1. `np.savez_compressed` 写 `histograms`、`labels`、`label_names`。
2. 返回保存路径。

调用者：
- `main`。

### 8.8 函数：`_build_arg_parser`

参数：
1. `--config` 必选。
2. `--dry-run` 可选。

作用：
- CLI 接口定义。

### 8.9 函数：`main`

完整流程：
1. 解析参数。
2. `load_yaml_config`。
3. 读取 `experiment.seed` 创建 `rng`。
4. `dataset_dir = ensure_dir(get_path(...))`。
5. `run_id = config_hash(config)`。
6. 从 `dataset.split` 读取比例。
7. 打印构建计划 `plan`。
8. 若 `--dry-run` 则退出。
9. 调 `_build_dataset`。
10. 调 `_split_indices`。
11. 调 `_save_split` 三次。
12. 写 `manifest.json`。
13. 打印完成信息。

输出文件规范：
- 每个 split 的 `.npz` 包含：
  1. `histograms`
  2. `labels`
  3. `label_names`

- manifest 包含：
  1. `run_id`
  2. `seed`
  3. 样本数量
  4. 标签名
  5. 文件路径

### 8.10 与其他模块关系

上游依赖：
1. `utils/config.py`
2. `physics.gkp_state.LATTICE_CONST`

下游消费者：
1. `model/train.py` 读取 `train/val.npz`
2. `model/evaluate.py` 读取任意 split

---

## 9. 文件详解：`cnn_fpga/model/__init__.py`

作用：
- 子包说明。

实际逻辑：
- 无函数与类。

---

## 10. 文件详解：`cnn_fpga/model/train.py`

### 10.1 文件定位

角色：
- 基线训练入口（线性回归）。

输出：
1. 模型文件 `.npz`
2. 训练报告 `.json`

### 10.2 函数：`_load_split`

输入：
1. `dataset_dir`
2. `split`

行为：
1. 读取 `split.npz`。
2. 返回 `histograms`, `labels`, `label_names`。

调用者：
- `main`。

### 10.3 函数：`_fit_linear_regression`

输入：
1. 特征矩阵 `x`
2. 标签矩阵 `y`
3. 正则系数 `l2_reg`

流程：
1. 对 `x` 做标准化（mean/std）。
2. `std<1e-8` 置 1.0 防止除零。
3. 对 `y` 做中心化。
4. 构造 Gram 矩阵 `X^T X + λI`。
5. 解线性方程 `gram * W = target`。
6. `bias = y_mean`。

输出：
1. `weights`
2. `bias`
3. `x_mean`
4. `x_std`

### 10.4 函数：`_predict`

输入：
1. `x`
2. `weights`
3. `bias`
4. `x_mean`
5. `x_std`

输出：
- `x_norm @ weights + bias`。

调用者：
- `main`。

### 10.5 函数：`_metrics`

输出指标：
1. `mse`
2. `mae`

调用者：
- `main`。

### 10.6 函数：`_arg_parser`

参数：
1. `--config` 必选。
2. `--train-split` 默认 `train`。
3. `--val-split` 默认 `val`。

### 10.7 函数：`main`

流程：
1. 读配置并算 `cfg_hash`。
2. 解析 `dataset_dir/model_dir/report_dir`。
3. 载入 train/val split。
4. 读取 `training.max_train_samples` 与 `l2_reg`。
5. 截断训练样本量（可控耗时/内存）。
6. 展平直方图为向量特征。
7. 训练 `_fit_linear_regression`。
8. 在 train/val 计算预测与指标。
9. 构造 `run_name = linear_baseline_{timestamp}_{cfg_hash}`。
10. 模型写盘 `.npz`。
11. 写训练报告 JSON。
12. 打印核心结果。

模型文件包含：
1. `model_type`
2. `weights`
3. `bias`
4. `x_mean`
5. `x_std`
6. `label_names`
7. `config_hash`

报告文件包含：
1. run 信息
2. 数据路径
3. 样本规模
4. 超参数
5. 训练/验证指标

### 10.8 与其他脚本的接口契约

被 `evaluate.py` 和 `quantize.py` 假定的字段：
1. `weights`
2. `bias`
3. `x_mean`
4. `x_std`
5. `label_names`

如果后续改模型格式：
- 需同步改 `evaluate._load_model` 与 `quantize.main`。

---

## 11. 文件详解：`cnn_fpga/model/evaluate.py`

### 11.1 文件定位

角色：
- 基线模型评估入口。

输出：
- 评估报告 JSON。

### 11.2 函数：`_load_model`

输入：
- `model_path`

行为：
1. 校验模型文件存在。
2. 读取 `weights/bias/x_mean/x_std/label_names`。
3. 全部转 `float64`（标签名除外）。

输出：
- 标准模型字典。

### 11.3 函数：`_load_split`

输入：
1. `dataset_dir`
2. `split`

行为：
1. 读取 split 文件。
2. 展平 `histograms`。
3. 返回 `(x,y)`。

### 11.4 函数：`_predict`

逻辑：
- 使用保存的标准化参数做推理。

输出：
- 预测矩阵。

### 11.5 函数：`_r2_score`

行为：
1. 计算 `ss_res` 与 `ss_tot`。
2. `ss_tot` 极小则返回 0。
3. 否则返回 `1 - ss_res/ss_tot`。

### 11.6 函数：`_metrics`

输出结构：
1. 全局 `mse`
2. 全局 `mae`
3. `r2_mean`（各维平均）
4. `per_label`（每标签 mse/mae/r2）

### 11.7 函数：`_arg_parser`

参数：
1. `--config` 必选
2. `--split` 可选（覆盖配置）
3. `--model-path` 可选（覆盖自动选择）

### 11.8 函数：`_find_latest_model`

行为：
1. `glob("*.npz")`
2. 排序后取最后一个。

注意：
- 依赖文件名排序，时间戳在文件名中时通常有效。

### 11.9 函数：`main`

流程：
1. 读配置。
2. 定位数据、模型、报告目录。
3. 解析评估 split（命令行优先）。
4. 解析模型路径（命令行优先，否则最新模型）。
5. 加载模型与数据。
6. 预测并计算 metrics。
7. 写 `eval_{split}_{timestamp}.json`。
8. 打印关键指标。

输出报告字段：
1. `run_name`
2. `model_path`
3. `split`
4. `n_samples`
5. `metrics`

---

## 12. 文件详解：`cnn_fpga/model/quantize.py`

### 12.1 文件定位

角色：
- 基线模型量化入口（int8 对称量化）。

输出：
1. 量化模型 `.npz`
2. 量化报告 JSON

### 12.2 函数：`_find_latest_model`

行为同 `evaluate.py`：
1. 枚举 `.npz`
2. 取排序最后

### 12.3 函数：`_quantize_symmetric`

输入：
- 浮点数组 `values`

逻辑：
1. `max_abs = max(abs(values))`
2. 若 `max_abs` 太小，`scale=1`
3. 否则 `scale = max_abs/127`
4. `q = round(values/scale)` 并裁剪到 `[-127,127]`

输出：
1. `q: int8`
2. `scale: float`

### 12.4 函数：`_arg_parser`

参数：
1. `--config` 必选
2. `--model-path` 可选

### 12.5 函数：`main`

流程：
1. 读配置。
2. 定位模型目录与报告目录。
3. 选择源模型路径。
4. 读取浮点 `weights/bias`。
5. 分别量化成 int8。
6. 反量化回浮点估算 `weight_mse/bias_mse`。
7. 输出量化模型文件。
8. 写量化报告。
9. 打印摘要。

量化模型字段：
1. `model_type=linear_regression_int8`
2. `q_weights`
3. `q_bias`
4. `w_scale`
5. `b_scale`
6. `x_mean`
7. `x_std`
8. `label_names`
9. `source_model`

---

## 13. 文件详解：`cnn_fpga/benchmark/__init__.py`

作用：
- benchmark 子包说明文件。

逻辑：
- 无函数定义。

---

## 14. 文件详解：`cnn_fpga/benchmark/run_drift_suite.py`

### 14.1 文件定位

角色：
- 漂移场景下对比 `full_qec` 与 `simplified` 的累计 LER 曲线。

输入：
- 漂移实验 YAML。

输出：
1. 每场景曲线 CSV
2. 汇总 `summary.json`

### 14.2 函数：`_wrap`

作用：
- 把标量映射到基本晶胞范围。

调用者：
- simplified 分支。

### 14.3 函数：`_build_sigma_trace`

输入：
1. `rng`
2. `n_rounds`
3. `scenario`

支持场景类型：
1. `linear`
2. `step`
3. `sin/sine/periodic`
4. `random_walk/rw`

输出：
- `sigma_trace`（长度 `n_rounds`）

统一后处理：
- 应用 `sigma_min/sigma_max` 裁剪。

### 14.4 函数：`_simulate_trace`

输入：
1. `sigma_trace`
2. `model_cfg`
3. `meas_cfg`
4. `use_full_qec_model`
5. `seed`

初始化：
1. `np.random.seed(seed)`
2. `LogicalErrorTracker()`
3. `rot`（由 `theta_deg`）
4. 读取 `gain/sigma_ratio_p/sigma_measurement/mu_q/mu_p`
5. `cumulative_residual=0`
6. 构建 `MeasurementConfig` 与 `RealisticSyndromeMeasurement`
7. `LinearDecoder(K=gain*rot,b=0)`

循环每轮逻辑：
1. 按 `sigma_trace[i]` 采样噪声。
2. 构造 `new_error = bias + rot @ base_noise`。
3. full 分支：
   - `total_error = cumulative_residual + new_error`
   - 真实测量 + 额外噪声
   - 线性解码
   - tracker 更新
   - `cumulative_residual` 用 tracker 内部 wrap 后状态
4. simplified 分支：
   - 不携带 residual
   - 分轴 wrap + 高斯测量噪声
   - correction = gain * syndrome
   - tracker 更新
5. 统计累计 LER 曲线 `curve[i]`。

输出：
- 单次重复的累计 LER 曲线。

### 14.5 函数：`_write_curve_csv`

列定义：
1. `round`
2. `full_mean`
3. `full_std`
4. `simplified_mean`
5. `simplified_std`
6. `gap_mean`

### 14.6 函数：`_arg_parser`

参数：
- `--config`。

### 14.7 函数：`main`

流程：
1. 加载配置并算 `cfg_hash`。
2. 读取 `seed/n_rounds/repeats`。
3. 读取 `model_cfg/meas_cfg/scenarios`。
4. 创建 run 目录 `runs/drift_suite/{name}_{time}_{hash}`。
5. 对每个场景重复仿真：
   - 构造 `sigma_trace`
   - 分别跑 full 和 simplified
6. 统计均值与标准差。
7. 写场景 CSV。
8. 构建场景 summary（含最终 gap）。
9. 写总 `summary.json`。
10. 打印场景摘要与完成信息。

summary.json 关键字段：
1. `config_hash`
2. `run_dir`
3. `n_rounds`
4. `repeats`
5. `scenarios`（数组）

每个场景字段：
1. `scenario`
2. `type`
3. `full_final_ler_mean/std`
4. `simplified_final_ler_mean/std`
5. `final_gap_mean`
6. `curve_csv`

### 14.8 与 `physics` 的调用链映射

full 分支调用链：
1. `MeasurementConfig`
2. `RealisticSyndromeMeasurement.measure`
3. `LinearDecoder.decode`
4. `LogicalErrorTracker.update`

simplified 分支调用链：
1. `_wrap`
2. `LogicalErrorTracker.update`

### 14.9 与 `benchmark/compare_full_vs_simplified_ler.py` 的关系

相同点：
1. 都做 full vs simplified 对照。
2. 都输出累计 LER 统计。

不同点：
1. `run_drift_suite.py` 强依赖 YAML 场景集。
2. 支持多场景批量。
3. 输出结构更工程化。

---

## 15. 文件详解：`cnn_fpga/benchmark/run_hil_suite.py`

### 15.1 文件定位

角色：
- HIL 脚手架入口（当前 mock backend）。

目标：
- 验证时延预算、违约率与异常注入统计链路。

### 15.2 函数：`_sample_positive_normal`

输入：
1. `rng`
2. `mean`
3. `std`
4. `size`

行为：
- 正态采样后 `clip(min=0)`。

用途：
- 各类时延分布抽样。

### 15.3 函数：`_run_mock_hil`

输入：
1. `config`
2. `run_dir`

步骤：
1. 读取 seed 创建 `rng`。
2. 校验 `hil.backend` 必须是 `mock`。
3. 读取时序规模：`n_updates`、`n_fast_cycles`。
4. 读取预算：`slow_budget_us`、`fast_budget_us`。
5. 读取 `latency_model` 参数并采样：
   - `dma`
   - `preprocess`
   - `inference`
   - `writeback`
6. 汇总慢回路总时延 `slow_total`，计算违约掩码。
7. 读取 `fast_path_model` 并采样 fast latency，计算违约掩码。
8. 读取 `error_model` 并按概率注入事件计数：
   - `dma_timeouts`
   - `axi_write_failures`
   - `cnn_infer_failures`
9. 汇总统计字典 `summary`。
10. 写 `hil_mock_summary.json`。

输出：
- `summary` 字典。

### 15.4 函数：`_arg_parser`

参数：
- `--config`。

### 15.5 函数：`main`

流程：
1. 读配置 + `cfg_hash`。
2. 创建 run 目录 `runs/hil_suite/{name}_{time}_{hash}`。
3. 调 `_run_mock_hil`。
4. 打印关键违约率与异常计数。

### 15.6 summary 字段说明

1. `n_slow_updates`
2. `n_fast_cycles`
3. `slow_update_budget_us`
4. `fast_cycle_budget_us`
5. `slow_update_mean_us`
6. `slow_update_p95_us`
7. `slow_update_p99_us`
8. `slow_update_violation_rate`
9. `fast_cycle_mean_us`
10. `fast_cycle_p95_us`
11. `fast_cycle_p99_us`
12. `fast_cycle_violation_rate`
13. `dma_timeouts`
14. `axi_write_failures`
15. `cnn_infer_failures`

---

## 16. 配置文件详解：`cnn_fpga/config/experiment_static.yaml`

### 16.1 文件用途

- 给数据构建、训练、评估、量化提供统一配置模板。

### 16.2 顶层键说明

`experiment`
- `name`: 实验名（如 `static_v1`）
- `stage`: 阶段标识（如 `p1_static`）
- `seed`: 随机种子

`paths`
- `output_root`
- `dataset_dir`
- `model_dir`
- `report_dir`

`hardware_defaults`
- `fixed_point`
- `t_fast_us`
- `window_size`
- `t_slow_update_ms`

`param_mapping`
- `alpha_bias`
- `beta_smoothing`
- `gain_clip`
- `theta_clip_deg`

`dataset`
- `n_configurations`
- `samples_per_configuration`
- `points_per_sample`
- `histogram_bins`
- `split.train/val/test`
- `parameter_ranges.sigma/mu_q/mu_p/theta_deg`

`training`
- `model_type`
- `max_train_samples`
- `l2_reg`

`evaluation`
- `target_split`

### 16.3 字段到代码映射

`experiment.seed`
- 用于 `dataset_builder.main` 的 RNG。

`paths.dataset_dir`
- 用于 `dataset_builder/main`, `train/main`, `evaluate/main`。

`paths.model_dir`
- 用于 `train/main`, `evaluate/main`, `quantize/main`。

`paths.report_dir`
- 用于 `train/main`, `evaluate/main`, `quantize/main`。

`dataset.*`
- 由 `dataset_builder._build_dataset` 读取。

`training.max_train_samples/l2_reg`
- 由 `train.main` 读取。

`evaluation.target_split`
- 由 `evaluate.main` 读取（可被 CLI 覆盖）。

### 16.4 当前未被代码使用但保留的键

`hardware_defaults`
- 当前训练链路不直接使用。
- 主要用于保持与工程方案语义一致。

`param_mapping`
- 当前训练链路不直接使用。
- 预留给未来 `(sigma,mu,theta)->(K,b)` 映射阶段。

---

## 17. 配置文件详解：`cnn_fpga/config/experiment_drift.yaml`

### 17.1 文件用途

- 驱动 `run_drift_suite.py` 执行多场景漂移对比。

### 17.2 顶层键

`experiment`
- name/stage/seed

`paths`
- output_root

`hardware_defaults`
- 与方案统一默认值

`param_mapping`
- 预留映射策略参数

`simulation`
- `n_rounds`
- `repeats`

`model`
- `gain`
- `theta_deg`
- `sigma_measurement`
- `sigma_ratio_p`
- `mu_q`
- `mu_p`

`measurement`
- `delta`
- `measurement_efficiency`
- `ancilla_error_rate`
- `add_shot_noise`

`drift.scenarios`
- 场景数组，含 `name/type/params`

### 17.3 场景类型说明

`linear`
- 参数：`sigma0`, `alpha`, `sigma_min`, `sigma_max`

`step`
- 参数：`sigma0`, `delta_sigma`, `jump_round`, `sigma_min`, `sigma_max`

`sin`
- 参数：`sigma0`, `amplitude`, `frequency`, `phase`, `sigma_min`, `sigma_max`

`random_walk`
- 参数：`sigma0`, `step_std`, `sigma_min`, `sigma_max`

### 17.4 字段到代码映射

`simulation.n_rounds/repeats`
- `run_drift_suite.main`

`model.*`
- `_simulate_trace`

`measurement.*`
- `_simulate_trace` -> `MeasurementConfig`

`drift.scenarios`
- `main` 循环每个场景

---

## 18. 配置文件详解：`cnn_fpga/config/hardware_emulation.yaml`

### 18.1 文件用途

- 用于 P2 行为仿真参数模板。

### 18.2 当前代码使用情况

- 当前仓库里没有专门 `run_hardware_emulation.py`。
- 该配置结构与 `experiment_drift.yaml` 接近，可由 `run_drift_suite.py` 思路复用。

### 18.3 顶层结构

1. `experiment`
2. `paths`
3. `hardware_defaults`
4. `param_mapping`
5. `simulation`
6. `model`
7. `measurement`
8. `drift.scenarios`

### 18.4 为什么保留

1. 对齐工程文档阶段划分。
2. 方便后续新增行为仿真脚本直接接入。
3. 保证参数模板先行稳定。

---

## 19. 配置文件详解：`cnn_fpga/config/hardware_hil.yaml`

### 19.1 文件用途

- 驱动 `run_hil_suite.py`（mock backend）运行。

### 19.2 顶层键

`experiment`
- name/stage/seed

`paths`
- output_root

`hardware_defaults`
- 工程默认硬件节拍参数

`param_mapping`
- 预留参数映射字段

`hil`
- `backend`
- `board`
- `bitstream_version`

`timing`
- `n_slow_updates`
- `n_fast_cycles`
- `slow_update_budget_us`
- `fast_cycle_budget_us`

`latency_model`
- `dma_mean_us/std_us`
- `preprocess_mean_us/std_us`
- `inference_mean_us/std_us`
- `writeback_mean_us/std_us`

`fast_path_model`
- `latency_mean_us`
- `latency_std_us`

`error_model`
- `dma_timeout_prob`
- `axi_write_fail_prob`
- `cnn_infer_fail_prob`

### 19.3 字段到代码映射

`hil.backend`
- `_run_mock_hil` 中检查必须是 `mock`。

`timing.*`
- `_run_mock_hil` 中用于样本规模与预算阈值。

`latency_model.*`
- `_run_mock_hil` 中用于慢回路四段时延采样。

`fast_path_model.*`
- `_run_mock_hil` 中用于 fast cycle 采样。

`error_model.*`
- `_run_mock_hil` 中用于异常注入计数。

---

## 20. 目录内调用关系矩阵（函数级）

说明：
- 左侧为调用者。
- 右侧为直接调用的同目录函数或外部关键函数。

`dataset_builder.main`
- 调用：`_build_arg_parser`
- 调用：`load_yaml_config`
- 调用：`ensure_dir`
- 调用：`get_path`
- 调用：`config_hash`
- 调用：`_build_dataset`
- 调用：`_split_indices`
- 调用：`_save_split`（三次）
- 调用：`save_json`

`dataset_builder._build_dataset`
- 调用：`_sample_param`
- 调用：`_histogram_from_params`

`dataset_builder._histogram_from_params`
- 调用：`_wrap_to_fundamental`

`train.main`
- 调用：`_arg_parser`
- 调用：`load_yaml_config`
- 调用：`config_hash`
- 调用：`get_path`
- 调用：`ensure_dir`
- 调用：`_load_split`（train/val）
- 调用：`_fit_linear_regression`
- 调用：`_predict`（train/val）
- 调用：`_metrics`（train/val）
- 调用：`now_tag`
- 调用：`save_json`

`evaluate.main`
- 调用：`_arg_parser`
- 调用：`load_yaml_config`
- 调用：`get_path`
- 调用：`ensure_dir`
- 调用：`_find_latest_model`（可选）
- 调用：`_load_model`
- 调用：`_load_split`
- 调用：`_predict`
- 调用：`_metrics`
- 调用：`now_tag`
- 调用：`save_json`

`evaluate._metrics`
- 调用：`_r2_score`（每个标签）

`quantize.main`
- 调用：`_arg_parser`
- 调用：`load_yaml_config`
- 调用：`get_path`
- 调用：`ensure_dir`
- 调用：`_find_latest_model`（可选）
- 调用：`_quantize_symmetric`（weights/bias）
- 调用：`now_tag`
- 调用：`save_json`

`run_drift_suite.main`
- 调用：`_arg_parser`
- 调用：`load_yaml_config`
- 调用：`config_hash`
- 调用：`get_path`
- 调用：`ensure_dir`
- 调用：`now_tag`
- 对每个场景/重复：
  - 调用：`_build_sigma_trace`
  - 调用：`_simulate_trace`（full）
  - 调用：`_simulate_trace`（simplified）
- 调用：`_write_curve_csv`
- 调用：`save_json`

`run_drift_suite._simulate_trace`（full 分支）
- 调用：`MeasurementConfig`
- 调用：`RealisticSyndromeMeasurement`
- 调用：`LinearDecoder`
- 每轮调用：`measurement.measure`
- 每轮调用：`decoder.decode`
- 每轮调用：`tracker.update`

`run_drift_suite._simulate_trace`（simplified 分支）
- 调用：`_wrap`
- 每轮调用：`tracker.update`

`run_hil_suite.main`
- 调用：`_arg_parser`
- 调用：`load_yaml_config`
- 调用：`config_hash`
- 调用：`get_path`
- 调用：`ensure_dir`
- 调用：`now_tag`
- 调用：`_run_mock_hil`

`run_hil_suite._run_mock_hil`
- 调用：`_sample_positive_normal`（多次）
- 调用：`save_json`

---

## 21. 跨文件数据契约

### 21.1 数据集文件契约（由 dataset_builder 产出）

每个 split `.npz` 必须至少有：
1. `histograms`
2. `labels`
3. `label_names`

`train.py` 假设：
- `histograms` 可以 reshape 成 `(N, -1)`。
- `labels` 维度与模型输出一致。

`evaluate.py` 假设：
- split `.npz` 同上结构。

### 21.2 模型文件契约（由 train 产出）

必须有：
1. `weights`
2. `bias`
3. `x_mean`
4. `x_std`
5. `label_names`

`evaluate.py` 和 `quantize.py` 都依赖这个格式。

### 21.3 量化模型契约（由 quantize 产出）

包含：
1. `q_weights`
2. `q_bias`
3. `w_scale`
4. `b_scale`
5. `x_mean`
6. `x_std`
7. `label_names`
8. `source_model`

当前仓库里没有“量化模型在线推理脚本”直接消费该契约，
但它是后续部署的中间产物格式。

---

## 22. 参数流动图（从配置到输出）

### 22.1 数据生成链路

`experiment_static.yaml`
-> `dataset_builder.main`
-> `dataset_cfg/parameter_ranges`
-> `_build_dataset`
-> `train/val/test.npz`
-> `manifest.json`

### 22.2 训练评估链路

`experiment_static.yaml`
-> `train.main`
-> 模型 `.npz`
-> `evaluate.main`
-> 评估报告 `.json`

### 22.3 量化链路

`experiment_static.yaml`
-> `quantize.main`
-> 量化模型 `.npz`
-> 量化报告 `.json`

### 22.4 漂移对比链路

`experiment_drift.yaml`
-> `run_drift_suite.main`
-> 多场景 full/simplified 曲线
-> 场景 CSV + summary.json

### 22.5 HIL 脚手架链路

`hardware_hil.yaml`
-> `run_hil_suite.main`
-> `_run_mock_hil`
-> `hil_mock_summary.json`

---

## 23. 执行流程详解（端到端）

### 23.1 从零开始跑静态基线

步骤 1：生成数据
- 脚本：`dataset_builder.py`
- 关键产物：`train/val/test.npz`

步骤 2：训练
- 脚本：`train.py`
- 关键产物：`linear_baseline_*.npz`

步骤 3：评估
- 脚本：`evaluate.py`
- 关键产物：`eval_*.json`

步骤 4：量化
- 脚本：`quantize.py`
- 关键产物：`*_int8_*.npz` + `*_quant_report.json`

### 23.2 漂移场景对比流程

步骤 1：配置场景
- 文件：`experiment_drift.yaml`

步骤 2：运行 suite
- 脚本：`run_drift_suite.py`

步骤 3：查看每场景 CSV
- 字段：`full_mean/simplified_mean/gap_mean`

步骤 4：查看 summary
- 关注：`final_gap_mean`

### 23.3 HIL 脚手架流程

步骤 1：配置时延模型与预算
- 文件：`hardware_hil.yaml`

步骤 2：运行 suite
- 脚本：`run_hil_suite.py`

步骤 3：分析 summary
- 关注：`slow_update_violation_rate`、`fast_cycle_violation_rate`

---

## 24. 你最容易踩坑的点

坑 1：配置路径默认值
- `get_path` 会读 `config['paths']`，没有就用默认。
- 如果忘写 `paths`，产物可能落到默认目录。

坑 2：模型选择“最新文件”
- `_find_latest_model` 按文件名排序，不是按文件修改时间。

坑 3：训练与评估 split 不一致
- `evaluate` 默认 `evaluation.target_split`，可被 `--split` 覆盖。

坑 4：fallback YAML 解析限制
- 复杂 YAML 语法可能不支持。

坑 5：随机数复现
- `run_drift_suite` 中既用 `default_rng` 生成轨迹，又用 `np.random.seed` 驱动每轮采样。
- 想严格复现时应固定所有相关 seed 字段。

坑 6：量化模型未直接联动评估
- 当前 `evaluate.py` 评估的是浮点模型格式。
- 若要评估量化精度，需新增量化推理路径。

坑 7：`hardware_emulation.yaml` 尚未有专用入口
- 它是模板，不是当前可直接运行入口脚本默认消费对象。

---

## 25. 模块扩展建议（按你后续可能要做的事）

### 25.1 从线性回归升级到 Tiny-CNN

需要改动点：
1. `train.py`：替换 `_fit_linear_regression`。
2. `evaluate.py`：替换 `_load_model/_predict`。
3. `quantize.py`：替换参数量化逻辑。

建议保持不变的接口：
1. CLI 参数。
2. 输出目录组织。
3. 报告 JSON 结构大框架。

### 25.2 增加量化模型评估脚本

建议新增：
- `cnn_fpga/model/evaluate_int8.py`

复用：
1. `load_yaml_config/get_path/ensure_dir/save_json`
2. `evaluate.py` 的 `_metrics`

### 25.3 增加真实 HIL 后端

建议新增模块：
1. `cnn_fpga/hwio/fpga_driver.py`
2. `cnn_fpga/hwio/dma_client.py`

并改 `run_hil_suite.py`：
- `if backend==mock` 走旧逻辑
- `elif backend==real` 调硬件接口

### 25.4 增加行为仿真入口

建议新增脚本：
- `cnn_fpga/benchmark/run_hardware_emulation.py`

输入配置：
- `hardware_emulation.yaml`

---

## 26. 代码阅读导航（按提问反查）

问题：数据集标签是什么？
- 看 `dataset_builder.py` 的 `LABEL_NAMES`。

问题：直方图是怎么构造的？
- 看 `_histogram_from_params`。

问题：训练到底用了什么模型？
- 看 `train.py` 的 `_fit_linear_regression`。

问题：评估指标有哪些？
- 看 `evaluate.py` 的 `_metrics`。

问题：int8 量化细节是什么？
- 看 `quantize.py` 的 `_quantize_symmetric`。

问题：漂移场景都支持什么类型？
- 看 `run_drift_suite.py` 的 `_build_sigma_trace`。

问题：full vs simplified 的代码分歧在哪？
- 看 `run_drift_suite.py` 的 `_simulate_trace` 分支。

问题：HIL 违约率怎么算？
- 看 `run_hil_suite.py` 的 `_run_mock_hil`。

问题：配置为何能在无 PyYAML 时读取？
- 看 `utils/config.py` fallback 解析链。

---

## 27. 每文件“用途一句话”速查

`cnn_fpga/__init__.py`
- 声明包版本。

`cnn_fpga/utils/config.py`
- 提供配置读取与结果写盘基础设施。

`cnn_fpga/data/dataset_builder.py`
- 合成直方图数据并切分 train/val/test。

`cnn_fpga/model/train.py`
- 训练线性回归基线并写模型与报告。

`cnn_fpga/model/evaluate.py`
- 读取模型与数据集分片做评估。

`cnn_fpga/model/quantize.py`
- 把浮点模型量化成 int8 并评估参数失真。

`cnn_fpga/benchmark/run_drift_suite.py`
- 在漂移场景下对比 full 与 simplified LER 曲线。

`cnn_fpga/benchmark/run_hil_suite.py`
- 用 mock 时延/故障模型执行 HIL 指标验证。

`cnn_fpga/config/experiment_static.yaml`
- 静态数据与训练评估配置模板。

`cnn_fpga/config/experiment_drift.yaml`
- 漂移实验配置模板。

`cnn_fpga/config/hardware_emulation.yaml`
- 硬件行为仿真模板（预留）。

`cnn_fpga/config/hardware_hil.yaml`
- HIL 脚手架配置模板。

---

## 28. 关键函数调用卡片（便于快速记忆）

卡片 1：`dataset_builder.main`
- 输入：YAML
- 输出：3 个 split + manifest
- 关键调用：`_build_dataset`

卡片 2：`train.main`
- 输入：dataset split
- 输出：浮点 baseline 模型
- 关键调用：`_fit_linear_regression`

卡片 3：`evaluate.main`
- 输入：模型 + split
- 输出：评估报告
- 关键调用：`_metrics`

卡片 4：`quantize.main`
- 输入：浮点模型
- 输出：int8 模型 + 量化误差报告
- 关键调用：`_quantize_symmetric`

卡片 5：`run_drift_suite.main`
- 输入：漂移场景配置
- 输出：场景曲线 CSV + summary
- 关键调用：`_simulate_trace`

卡片 6：`run_hil_suite.main`
- 输入：HIL mock 配置
- 输出：时延/违约率摘要
- 关键调用：`_run_mock_hil`

---

## 29. 配置字段到代码路径映射总表

### 29.1 `experiment_static.yaml`

`experiment.seed`
- `dataset_builder.main`

`paths.dataset_dir`
- `dataset_builder.main`
- `train.main`
- `evaluate.main`

`paths.model_dir`
- `train.main`
- `evaluate.main`
- `quantize.main`

`paths.report_dir`
- `train.main`
- `evaluate.main`
- `quantize.main`

`dataset.n_configurations`
- `_build_dataset`

`dataset.samples_per_configuration`
- `_build_dataset`

`dataset.points_per_sample`
- `_build_dataset`
- `_histogram_from_params`

`dataset.histogram_bins`
- `_build_dataset`
- `_histogram_from_params`

`dataset.split.train/val`
- `_split_indices`

`dataset.parameter_ranges.*`
- `_sample_param` -> `_build_dataset`

`training.max_train_samples`
- `train.main`

`training.l2_reg`
- `train.main` -> `_fit_linear_regression`

`evaluation.target_split`
- `evaluate.main`

### 29.2 `experiment_drift.yaml`

`simulation.n_rounds`
- `run_drift_suite.main`

`simulation.repeats`
- `run_drift_suite.main`

`model.gain`
- `_simulate_trace`（decoder 或 simplified correction）

`model.theta_deg`
- `_simulate_trace`（旋转矩阵）

`model.sigma_measurement`
- `_simulate_trace`（测量附加噪声）

`model.sigma_ratio_p`
- `_simulate_trace`（p 轴噪声比）

`model.mu_q/mu_p`
- `_simulate_trace`（偏置）

`measurement.*`
- `_simulate_trace` -> `MeasurementConfig`

`drift.scenarios[*]`
- `main` 循环
- `_build_sigma_trace`

### 29.3 `hardware_hil.yaml`

`timing.n_slow_updates`
- `_run_mock_hil`

`timing.n_fast_cycles`
- `_run_mock_hil`

`timing.slow_update_budget_us`
- `_run_mock_hil`

`timing.fast_cycle_budget_us`
- `_run_mock_hil`

`latency_model.dma_mean_us/std_us`
- `_run_mock_hil` -> `_sample_positive_normal`

`latency_model.preprocess_mean_us/std_us`
- `_run_mock_hil`

`latency_model.inference_mean_us/std_us`
- `_run_mock_hil`

`latency_model.writeback_mean_us/std_us`
- `_run_mock_hil`

`fast_path_model.latency_mean_us/std_us`
- `_run_mock_hil`

`error_model.*`
- `_run_mock_hil` 概率事件统计

---

## 30. I/O 产物总表

### 30.1 `dataset_builder.py`

输出目录：`paths.dataset_dir`

输出文件：
1. `train.npz`
2. `val.npz`
3. `test.npz`
4. `manifest.json`

### 30.2 `train.py`

输出目录：`paths.model_dir`、`paths.report_dir`

输出文件：
1. `linear_baseline_{time}_{hash}.npz`
2. `linear_baseline_{time}_{hash}_train_report.json`

### 30.3 `evaluate.py`

输出目录：`paths.report_dir`

输出文件：
1. `eval_{split}_{time}.json`

### 30.4 `quantize.py`

输出目录：`paths.model_dir`、`paths.report_dir`

输出文件：
1. `{source_stem}_int8_{time}.npz`
2. `{source_stem}_int8_{time}_quant_report.json`

### 30.5 `run_drift_suite.py`

输出目录：`paths.output_root/drift_suite/{name}_{time}_{hash}`

输出文件：
1. `{scenario_name}_curve.csv`（每场景）
2. `summary.json`

### 30.6 `run_hil_suite.py`

输出目录：`paths.output_root/hil_suite/{name}_{time}_{hash}`

输出文件：
1. `hil_mock_summary.json`

---

## 31. 代码一致性与工程可维护性观察

### 31.1 一致性优点

1. CLI 风格统一。
2. 路径解析统一走 `get_path/ensure_dir`。
3. JSON 报告统一走 `save_json`。
4. run 名中普遍包含 `now_tag` 或 `config_hash`。

### 31.2 一致性缺口

1. `evaluate.py` 与 `quantize.py` 的“最新模型选择”逻辑重复，可抽公共函数。
2. `train/evaluate` 的 `_load_split` 逻辑重复，可抽公共数据加载工具。
3. `run_drift_suite` 与 `physics.simulate_error_accumulation` 存在逻辑重复，后续可选择复用单一实现避免漂移。

### 31.3 可维护建议

1. 把模型 I/O 契约抽成 dataclass（例如 `ModelArtifact`）。
2. 把报告字段结构抽成 schema 常量。
3. 为 YAML 字段建立显式校验（如 pydantic）。

---

## 32. 调试手册（按症状定位）

症状：`Dataset split not found`
- 排查：`dataset_dir` 是否正确。
- 排查：是否先运行过 `dataset_builder`。

症状：`No model files in ...`
- 排查：`model_dir` 是否写错。
- 排查：是否先运行 `train`。

症状：`YAML root must be a mapping`
- 排查：配置根是否为字典。

症状：`Only mock backend is implemented`
- 排查：`hardware_hil.yaml` 的 `hil.backend` 是否为 `mock`。

症状：漂移 suite 提示 `No drift scenarios configured`
- 排查：`drift.scenarios` 是否为空。

症状：评估指标异常高
- 排查：`train` 与 `evaluate` 是否使用同一个数据域与标签定义。

症状：量化 MSE 异常大
- 排查：源模型参数是否范围过大。
- 排查：是否需要分通道量化替代全局量化。

---

## 33. 未来扩展时推荐保持不变的接口

1. CLI 参数名：`--config` 保持一致。
2. `paths` 键名：`dataset_dir/model_dir/report_dir/output_root` 保持不变。
3. 报告写盘方式：统一 JSON。
4. 运行目录命名：`{name}_{time}_{hash}` 格式保持不变。

这样可以保证：
- 旧脚本、旧文档、旧自动化流程不易失效。

---

## 34. 总结（你的代码地图）

如果你只记三件事：
1. `utils/config.py` 是全目录运行基础设施核心。
2. `data -> model -> benchmark` 是主业务流。
3. `config/*.yaml` 字段是行为开关，理解字段映射就是理解脚本行为。

如果你只记一条主链：
- `experiment_static.yaml`
-> `dataset_builder`
-> `train`
-> `evaluate`
-> `quantize`

如果你只记一个对照链：
- `experiment_drift.yaml`
-> `run_drift_suite`
-> `full vs simplified` LER 差异曲线

如果你只记一个工程验证链：
- `hardware_hil.yaml`
-> `run_hil_suite`
-> 时延预算与异常注入统计

---

## 35. 附录 A：符号索引（按文件）

### `cnn_fpga/utils/config.py`

`load_yaml_config`
- 读取 YAML 配置。

`config_hash`
- 生成配置哈希。

`now_tag`
- 生成时间戳标签。

`ensure_dir`
- 创建目录。

`get_path`
- 读取路径字段。

`save_json`
- 写 JSON。

`_strip_comment`
- fallback 注释剥离。

`_to_tokens`
- fallback 分词。

`_parse_scalar`
- fallback 标量解析。

`_parse_block`
- fallback 递归解析。

`_load_yaml_fallback`
- fallback 总入口。

### `cnn_fpga/data/dataset_builder.py`

`LABEL_NAMES`
- 标签名元组。

`_sample_param`
- 参数采样。

`_wrap_to_fundamental`
- 映射到晶胞。

`_histogram_from_params`
- 单样本直方图构造。

`_build_dataset`
- 整体数据构建。

`_split_indices`
- split 索引切分。

`_save_split`
- split 写盘。

`_build_arg_parser`
- CLI 参数。

`main`
- 数据构建入口。

### `cnn_fpga/model/train.py`

`_load_split`
- 加载 split。

`_fit_linear_regression`
- 拟合线性回归。

`_predict`
- 预测。

`_metrics`
- 训练指标。

`_arg_parser`
- CLI 参数。

`main`
- 训练入口。

### `cnn_fpga/model/evaluate.py`

`_load_model`
- 加载模型。

`_load_split`
- 加载评估数据。

`_predict`
- 预测。

`_r2_score`
- R2 计算。

`_metrics`
- 评估指标。

`_arg_parser`
- CLI 参数。

`_find_latest_model`
- 选择最新模型。

`main`
- 评估入口。

### `cnn_fpga/model/quantize.py`

`_find_latest_model`
- 选择最新模型。

`_quantize_symmetric`
- 对称量化。

`_arg_parser`
- CLI 参数。

`main`
- 量化入口。

### `cnn_fpga/benchmark/run_drift_suite.py`

`_wrap`
- 标量 wrap。

`_build_sigma_trace`
- 构建漂移轨迹。

`_simulate_trace`
- 单次曲线仿真。

`_write_curve_csv`
- 写曲线 CSV。

`_arg_parser`
- CLI 参数。

`main`
- 漂移 suite 入口。

### `cnn_fpga/benchmark/run_hil_suite.py`

`_sample_positive_normal`
- 正时延采样。

`_run_mock_hil`
- mock HIL 运行。

`_arg_parser`
- CLI 参数。

`main`
- HIL suite 入口。

---

## 36. 附录 B：你可以直接复用的检查清单

### 数据构建检查清单

1. 配置是否包含 `dataset`。
2. `split.train+val+test` 是否约等于 1。
3. `histogram_bins` 是否与模型输入预期一致。
4. 生成后是否有 `manifest.json`。

### 训练检查清单

1. `dataset_dir` 是否存在 train/val。
2. `max_train_samples` 是否过小导致欠拟合。
3. `l2_reg` 是否过大导致偏差过强。
4. 输出模型是否包含全部字段。

### 评估检查清单

1. `split` 是否正确。
2. 评估模型是否与训练配置一致。
3. `r2_mean` 是否为负（负值通常意味着效果不佳）。

### 量化检查清单

1. `source_model` 是否正确。
2. `weight_mse/bias_mse` 是否在可接受范围。
3. 量化模型是否写入预期目录。

### 漂移 suite 检查清单

1. 每个场景是否生成 CSV。
2. `summary.json` 是否包含全部场景。
3. `final_gap_mean` 符号与预期是否一致。

### HIL suite 检查清单

1. `slow_update_violation_rate` 是否低于预算目标。
2. `fast_cycle_violation_rate` 是否低于预算目标。
3. 异常计数是否在概率预期范围。

---

## 37. 文档维护建议

1. 每次新增脚本，先更新第 20 章调用矩阵。
2. 每次新增配置字段，先更新第 29 章映射表。
3. 每次改模型产物格式，先更新第 21 章契约说明。
4. 每次改路径命名规则，先更新第 30 章 I/O 总表。

