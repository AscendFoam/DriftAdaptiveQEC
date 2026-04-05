"""Build runtime-consistent CNN inputs from histogram windows and teacher states."""

# 中文说明：
# - 该模块把“窗口历史 + teacher 条件输入”编码成单个 `(C, H, W)` 张量，
#   供数据集构建与运行时 slow-loop 共用，避免训练/部署口径漂移。
# - 为了兼容当前 Tiny-CNN / TFLite 单输入接口，teacher 标量特征会被广播成常数平面。

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


@dataclass(frozen=True)
class RuntimeFeatureConfig:
    """Configuration for runtime-consistent input feature encoding."""

    context_windows: int = 1
    context_padding: str = "edge_repeat"
    include_histogram_deltas: bool = False
    include_teacher_prediction: bool = False
    include_teacher_params: bool = False
    include_teacher_deltas: bool = False

    def __post_init__(self) -> None:
        if self.context_windows <= 0:
            raise ValueError("context_windows must be positive")
        if str(self.context_padding).lower() not in {"edge_repeat"}:
            raise ValueError(f"unsupported_context_padding:{self.context_padding}")


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

    if config.include_teacher_prediction:
        names.extend(["teacher_sigma", "teacher_mu_q", "teacher_mu_p", "teacher_theta_deg"])

    if config.include_teacher_params:
        names.extend(["teacher_b_q", "teacher_b_p"])

    if config.include_teacher_deltas:
        names.extend(
            [
                "teacher_delta_sigma",
                "teacher_delta_mu_q",
                "teacher_delta_mu_p",
                "teacher_delta_theta_deg",
                "teacher_delta_b_q",
                "teacher_delta_b_p",
            ]
        )
    return names


def build_feature_tensor(
    histogram_history: Sequence[np.ndarray],
    config: RuntimeFeatureConfig,
    *,
    teacher_prediction_history: Sequence[Sequence[float] | dict] | None = None,
    teacher_param_history: Sequence[DecoderRuntimeParams | Sequence[float] | dict] | None = None,
) -> np.ndarray:
    """Build one `(C, H, W)` input tensor from window history and teacher state."""
    hist_frames = [np.asarray(item, dtype=np.float32) for item in histogram_history]
    if not hist_frames:
        raise ValueError("histogram_history must be non-empty")
    if hist_frames[0].ndim != 2:
        raise ValueError(f"each histogram must be 2D, got {hist_frames[0].shape}")

    selected_hist = _pad_sequence(hist_frames, config.context_windows, padding=config.context_padding)
    height, width = selected_hist[-1].shape
    channels: List[np.ndarray] = [frame.astype(np.float32) for frame in selected_hist]

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
        for value in current_teacher:
            channels.append(np.full((height, width), float(value), dtype=np.float32))
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
        for value in current_params:
            channels.append(np.full((height, width), float(value), dtype=np.float32))
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
        for value in teacher_delta:
            channels.append(np.full((height, width), float(value), dtype=np.float32))
        for value in b_delta:
            channels.append(np.full((height, width), float(value), dtype=np.float32))

    return np.stack(channels, axis=0).astype(np.float32)


def infer_input_channels(config: RuntimeFeatureConfig) -> int:
    """Return the number of channels produced by the configured feature encoder."""
    return len(feature_channel_names(config))
