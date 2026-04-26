"""Build runtime-consistent CNN inputs from histogram windows and teacher states."""

# 中文说明：
# - 该模块把“窗口历史 + teacher 条件输入”编码成运行时一致的模型输入。
# - 旧版本会把 teacher 标量特征广播成常数平面，容易带来高冗余耦合。
# - 当前保留旧口径，同时新增“标量支路”表达：histogram 仍走 `(C, H, W)` 主干，
#   teacher params 可以改走低维 `scalar_features`，供训练与 slow-loop 共用。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


TEACHER_PREDICTION_NAMES = ("teacher_sigma", "teacher_mu_q", "teacher_mu_p", "teacher_theta_deg")
TEACHER_PARAM_NAMES = ("teacher_b_q", "teacher_b_p")
TEACHER_DELTA_NAMES = (
    "teacher_delta_sigma",
    "teacher_delta_mu_q",
    "teacher_delta_mu_p",
    "teacher_delta_theta_deg",
    "teacher_delta_b_q",
    "teacher_delta_b_p",
)
_SUPPORTED_LAYOUTS = {"broadcast", "scalar_branch"}


@dataclass(frozen=True)
class RuntimeFeatureConfig:
    """Configuration for runtime-consistent input feature encoding."""

    context_windows: int = 1
    context_padding: str = "edge_repeat"
    include_histogram_deltas: bool = False
    include_teacher_prediction: bool = False
    include_teacher_params: bool = False
    include_teacher_deltas: bool = False
    teacher_prediction_layout: str = "broadcast"
    teacher_params_layout: str = "broadcast"
    teacher_deltas_layout: str = "broadcast"
    teacher_scalar_features: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.context_windows <= 0:
            raise ValueError("context_windows must be positive")
        if str(self.context_padding).lower() not in {"edge_repeat"}:
            raise ValueError(f"unsupported_context_padding:{self.context_padding}")
        for field_name in ("teacher_prediction_layout", "teacher_params_layout", "teacher_deltas_layout"):
            value = str(getattr(self, field_name)).lower()
            if value not in _SUPPORTED_LAYOUTS:
                raise ValueError(f"unsupported_{field_name}:{value}")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "teacher_scalar_features", tuple(str(item) for item in self.teacher_scalar_features))


@dataclass(frozen=True)
class RuntimeFeatureSample:
    """One runtime-consistent model input sample."""

    spatial_tensor: np.ndarray
    scalar_features: np.ndarray | None = None


def _normalize_prediction_vector(prediction: Sequence[float] | dict) -> np.ndarray:
    """Convert teacher prediction to `[sigma, mu_q, mu_p, theta_deg]` array."""
    if isinstance(prediction, dict):
        return np.array(
            [
                float(prediction.get("sigma", 0.0)),
                float(prediction.get("mu_q", 0.0)),
                float(prediction.get("mu_p", 0.0)),
                float(prediction.get("theta_deg", 0.0)),
            ],
            dtype=np.float32,
        )
    array = np.asarray(prediction, dtype=np.float32).reshape(-1)
    if array.size != 4:
        raise ValueError(f"teacher prediction must have 4 values, got shape {array.shape}")
    return array.astype(np.float32)


def _normalize_param_vector(params: DecoderRuntimeParams | Sequence[float] | dict) -> np.ndarray:
    """Convert teacher runtime params to `[b_q, b_p]` array."""
    if isinstance(params, DecoderRuntimeParams):
        return np.asarray(params.b, dtype=np.float32).reshape(2)
    if isinstance(params, dict):
        raw = params.get("b", params.get("bias", [0.0, 0.0]))
        array = np.asarray(raw, dtype=np.float32).reshape(-1)
    else:
        array = np.asarray(params, dtype=np.float32).reshape(-1)
    if array.size != 2:
        raise ValueError(f"teacher param vector must have 2 values, got shape {array.shape}")
    return array.astype(np.float32)


def _scalar_feature_selected(config: RuntimeFeatureConfig, name: str) -> bool:
    selected = tuple(config.teacher_scalar_features)
    return not selected or str(name) in selected


def _append_scalar_items(
    scalar_names: List[str],
    scalar_values: List[float],
    names: Sequence[str],
    values: Sequence[float],
    config: RuntimeFeatureConfig,
) -> None:
    for name, value in zip(names, values):
        if _scalar_feature_selected(config, str(name)):
            scalar_names.append(str(name))
            scalar_values.append(float(value))


def _pad_sequence(values: Sequence[np.ndarray], size: int, *, padding: str) -> List[np.ndarray]:
    if not values:
        raise ValueError("sequence must be non-empty")
    if size <= 1:
        return [np.asarray(values[-1])]
    if len(values) >= size:
        return [np.asarray(item) for item in values[-size:]]
    if padding != "edge_repeat":
        raise ValueError(f"unsupported_context_padding:{padding}")
    head = np.asarray(values[0]).copy()
    return [head.copy() for _ in range(size - len(values))] + [np.asarray(item) for item in values]


def feature_channel_names(config: RuntimeFeatureConfig) -> List[str]:
    """Return ordered feature-channel names for the configured encoding."""
    names: List[str] = []
    hist_offsets = list(range(config.context_windows - 1, -1, -1))
    for offset in hist_offsets:
        names.append(f"hist_t-{offset}" if offset > 0 else "hist_t")

    if config.include_histogram_deltas:
        for offset in range(config.context_windows - 1, 0, -1):
            newer = offset - 1
            older = offset
            left = f"t-{newer}" if newer > 0 else "t"
            right = f"t-{older}"
            names.append(f"hist_delta_{left}_minus_{right}")

    if config.include_teacher_prediction and config.teacher_prediction_layout == "broadcast":
        names.extend(TEACHER_PREDICTION_NAMES)

    if config.include_teacher_params and config.teacher_params_layout == "broadcast":
        names.extend(TEACHER_PARAM_NAMES)

    if config.include_teacher_deltas and config.teacher_deltas_layout == "broadcast":
        names.extend(TEACHER_DELTA_NAMES)
    return names


def scalar_feature_names(config: RuntimeFeatureConfig) -> List[str]:
    """Return ordered scalar-feature names for side-branch encoding."""
    names: List[str] = []
    if config.include_teacher_prediction and config.teacher_prediction_layout == "scalar_branch":
        names.extend([name for name in TEACHER_PREDICTION_NAMES if _scalar_feature_selected(config, name)])
    if config.include_teacher_params and config.teacher_params_layout == "scalar_branch":
        names.extend([name for name in TEACHER_PARAM_NAMES if _scalar_feature_selected(config, name)])
    if config.include_teacher_deltas and config.teacher_deltas_layout == "scalar_branch":
        names.extend([name for name in TEACHER_DELTA_NAMES if _scalar_feature_selected(config, name)])
    return names


def build_feature_sample(
    histogram_history: Sequence[np.ndarray],
    config: RuntimeFeatureConfig,
    *,
    teacher_prediction_history: Sequence[Sequence[float] | dict] | None = None,
    teacher_param_history: Sequence[DecoderRuntimeParams | Sequence[float] | dict] | None = None,
) -> RuntimeFeatureSample:
    """Build one runtime-consistent input sample from window history and teacher state."""
    hist_frames = [np.asarray(item, dtype=np.float32) for item in histogram_history]
    if not hist_frames:
        raise ValueError("histogram_history must be non-empty")
    if hist_frames[0].ndim != 2:
        raise ValueError(f"each histogram must be 2D, got {hist_frames[0].shape}")

    selected_hist = _pad_sequence(hist_frames, config.context_windows, padding=config.context_padding)
    height, width = selected_hist[-1].shape
    channels: List[np.ndarray] = [frame.astype(np.float32) for frame in selected_hist]
    scalar_names: List[str] = []
    scalar_values: List[float] = []

    if config.include_histogram_deltas:
        for prev_frame, next_frame in zip(selected_hist[:-1], selected_hist[1:]):
            channels.append((next_frame - prev_frame).astype(np.float32))

    if config.include_teacher_prediction:
        if not teacher_prediction_history:
            raise ValueError("teacher_prediction_history is required when include_teacher_prediction=True")
        selected_teacher = _pad_sequence(
            [_normalize_prediction_vector(item) for item in teacher_prediction_history],
            config.context_windows,
            padding=config.context_padding,
        )
        current_teacher = selected_teacher[-1]
        if config.teacher_prediction_layout == "broadcast":
            for value in current_teacher:
                channels.append(np.full((height, width), float(value), dtype=np.float32))
        else:
            _append_scalar_items(scalar_names, scalar_values, TEACHER_PREDICTION_NAMES, current_teacher, config)
    else:
        selected_teacher = []

    if config.include_teacher_params:
        if not teacher_param_history:
            raise ValueError("teacher_param_history is required when include_teacher_params=True")
        selected_params = _pad_sequence(
            [_normalize_param_vector(item) for item in teacher_param_history],
            config.context_windows,
            padding=config.context_padding,
        )
        current_params = selected_params[-1]
        if config.teacher_params_layout == "broadcast":
            for value in current_params:
                channels.append(np.full((height, width), float(value), dtype=np.float32))
        else:
            _append_scalar_items(scalar_names, scalar_values, TEACHER_PARAM_NAMES, current_params, config)
    else:
        selected_params = []

    if config.include_teacher_deltas:
        if not selected_teacher:
            if not teacher_prediction_history:
                raise ValueError("teacher_prediction_history is required when include_teacher_deltas=True")
            selected_teacher = _pad_sequence(
                [_normalize_prediction_vector(item) for item in teacher_prediction_history],
                config.context_windows,
                padding=config.context_padding,
            )
        if not selected_params:
            if not teacher_param_history:
                raise ValueError("teacher_param_history is required when include_teacher_deltas=True")
            selected_params = _pad_sequence(
                [_normalize_param_vector(item) for item in teacher_param_history],
                config.context_windows,
                padding=config.context_padding,
            )
        prev_teacher = selected_teacher[-2] if len(selected_teacher) >= 2 else selected_teacher[-1]
        prev_params = selected_params[-2] if len(selected_params) >= 2 else selected_params[-1]
        teacher_delta = selected_teacher[-1] - prev_teacher
        b_delta = selected_params[-1] - prev_params
        delta_values = np.concatenate([teacher_delta, b_delta]).astype(np.float32)
        if config.teacher_deltas_layout == "broadcast":
            for value in delta_values:
                channels.append(np.full((height, width), float(value), dtype=np.float32))
        else:
            _append_scalar_items(scalar_names, scalar_values, TEACHER_DELTA_NAMES, delta_values, config)

    scalar_features: np.ndarray | None = None
    expected_scalar_names = scalar_feature_names(config)
    if expected_scalar_names:
        if scalar_names != expected_scalar_names:
            raise ValueError(f"scalar feature order mismatch: expected={expected_scalar_names}, actual={scalar_names}")
        scalar_features = np.asarray(scalar_values, dtype=np.float32).reshape(-1)

    return RuntimeFeatureSample(
        spatial_tensor=np.stack(channels, axis=0).astype(np.float32),
        scalar_features=scalar_features,
    )


def build_feature_tensor(
    histogram_history: Sequence[np.ndarray],
    config: RuntimeFeatureConfig,
    *,
    teacher_prediction_history: Sequence[Sequence[float] | dict] | None = None,
    teacher_param_history: Sequence[DecoderRuntimeParams | Sequence[float] | dict] | None = None,
) -> np.ndarray:
    """Backward-compatible `(C, H, W)` builder for broadcast-only feature layouts."""
    sample = build_feature_sample(
        histogram_history,
        config,
        teacher_prediction_history=teacher_prediction_history,
        teacher_param_history=teacher_param_history,
    )
    scalar = sample.scalar_features
    if scalar is not None and scalar.size > 0:
        raise ValueError("build_feature_tensor does not support scalar-branch teacher params; use build_feature_sample")
    return sample.spatial_tensor.astype(np.float32)


def infer_input_channels(config: RuntimeFeatureConfig) -> int:
    """Return the number of channels produced by the configured feature encoder."""
    return len(feature_channel_names(config))


def infer_scalar_feature_dim(config: RuntimeFeatureConfig) -> int:
    """Return the scalar side-branch feature dimension for the configured encoder."""
    return len(scalar_feature_names(config))
