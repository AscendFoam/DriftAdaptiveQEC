"""Pure-NumPy Tiny-CNN used as the first main model path for CNN-FPGA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class TinyCNNConfig:
    """Training and architecture config for the NumPy Tiny-CNN."""

    conv_channels: int = 8
    kernel_size: int = 3
    hidden_dim: int = 32
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 5.0e-3
    weight_decay: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1.0e-8
    patience: int = 5
    seed: int = 1234
    label_weights: Dict[str, float] | None = None
    backend: str = "numpy"
    device: str = "auto"
    scalar_fusion_mode: str = "concat"
    scalar_gate_init_bias: float = 2.0

    @classmethod
    def from_config(cls, config: Dict) -> "TinyCNNConfig":
        training = config.get("training", {})
        cnn_cfg = training.get("tiny_cnn", {})
        experiment = config.get("experiment", {})
        return cls(
            conv_channels=int(cnn_cfg.get("conv_channels", 8)),
            kernel_size=int(cnn_cfg.get("kernel_size", 3)),
            hidden_dim=int(cnn_cfg.get("hidden_dim", 32)),
            batch_size=int(cnn_cfg.get("batch_size", 64)),
            epochs=int(cnn_cfg.get("epochs", 20)),
            learning_rate=float(cnn_cfg.get("learning_rate", 5.0e-3)),
            weight_decay=float(cnn_cfg.get("weight_decay", 1.0e-4)),
            beta1=float(cnn_cfg.get("beta1", 0.9)),
            beta2=float(cnn_cfg.get("beta2", 0.999)),
            adam_eps=float(cnn_cfg.get("adam_eps", 1.0e-8)),
            patience=int(cnn_cfg.get("patience", 5)),
            seed=int(cnn_cfg.get("seed", experiment.get("seed", 1234))),
            label_weights=dict(cnn_cfg.get("label_weights", {})),
            backend=str(cnn_cfg.get("backend", "numpy")).lower(),
            device=str(cnn_cfg.get("device", "auto")).lower(),
            scalar_fusion_mode=str(cnn_cfg.get("scalar_fusion_mode", "concat")).lower(),
            scalar_gate_init_bias=float(cnn_cfg.get("scalar_gate_init_bias", 2.0)),
        )


def _he_std(fan_in: int) -> float:
    return float(np.sqrt(2.0 / max(1, fan_in)))


def _same_padding(kernel_size: int) -> int:
    return kernel_size // 2


def _im2col(x: np.ndarray, kernel_size: int, padding: int) -> np.ndarray:
    """Convert `(N, C, H, W)` to im2col matrix for stride-1 convolution."""
    n_batch, channels, height, width = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    out_h = height
    out_w = width
    cols = np.empty((n_batch, out_h, out_w, channels, kernel_size, kernel_size), dtype=x.dtype)
    for i in range(kernel_size):
        for j in range(kernel_size):
            cols[:, :, :, :, i, j] = padded[:, :, i : i + out_h, j : j + out_w].transpose(0, 2, 3, 1)
    return cols.reshape(n_batch * out_h * out_w, channels * kernel_size * kernel_size)


def _col2im(grad_cols: np.ndarray, x_shape: Tuple[int, int, int, int], kernel_size: int, padding: int) -> np.ndarray:
    """Inverse of `_im2col` for stride-1 convolution."""
    n_batch, channels, height, width = x_shape
    out_h = height
    out_w = width
    grad_cols = grad_cols.reshape(n_batch, out_h, out_w, channels, kernel_size, kernel_size)
    grad_padded = np.zeros((n_batch, channels, height + 2 * padding, width + 2 * padding), dtype=float)
    for i in range(kernel_size):
        for j in range(kernel_size):
            grad_padded[:, :, i : i + out_h, j : j + out_w] += grad_cols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    if padding == 0:
        return grad_padded
    return grad_padded[:, :, padding:-padding, padding:-padding]


def _avg_pool2x2_forward(x: np.ndarray) -> np.ndarray:
    n_batch, channels, height, width = x.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("avg_pool2x2 expects even spatial dimensions")
    return x.reshape(n_batch, channels, height // 2, 2, width // 2, 2).mean(axis=(3, 5))


def _avg_pool2x2_backward(grad_output: np.ndarray, input_shape: Tuple[int, int, int, int]) -> np.ndarray:
    n_batch, channels, height, width = input_shape
    grad = np.repeat(np.repeat(grad_output, 2, axis=2), 2, axis=3) / 4.0
    return grad.reshape(n_batch, channels, height, width)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1.0e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _channelwise_input_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute channelwise normalization stats for `(N, C, H, W)` or `(N, H, W)` inputs."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 3:
        x_mean = array.mean(axis=(0, 1, 2), keepdims=True)
        x_std = array.std(axis=(0, 1, 2), keepdims=True)
    elif array.ndim == 4:
        x_mean = array.mean(axis=(0, 2, 3), keepdims=True)
        x_std = array.std(axis=(0, 2, 3), keepdims=True)
    else:
        raise ValueError(f"Unsupported input shape for channelwise stats: {array.shape}")
    x_std = np.where(x_std < 1.0e-8, 1.0, x_std)
    return x_mean.astype(np.float64), x_std.astype(np.float64)


def _normalize_inputs(x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    """Normalize histogram input with stored training-domain channel stats."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 3:
        mean = np.asarray(x_mean, dtype=np.float64).reshape(1, 1, 1)
        std = np.asarray(x_std, dtype=np.float64).reshape(1, 1, 1)
        return (array - mean) / std
    if array.ndim == 4:
        mean = np.asarray(x_mean, dtype=np.float64).reshape(1, -1, 1, 1)
        std = np.asarray(x_std, dtype=np.float64).reshape(1, -1, 1, 1)
        return (array - mean) / std
    raise ValueError(f"Unsupported input shape for normalization: {array.shape}")


def _build_label_weights(label_names, config: TinyCNNConfig) -> np.ndarray:
    weights = np.ones(len(label_names), dtype=np.float64)
    if config.label_weights:
        for idx, name in enumerate(label_names):
            weights[idx] = float(config.label_weights.get(str(name), 1.0))
    return weights


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _resolve_scalar_fusion_mode(scalar_feature_dim: int, fusion_mode: str) -> str:
    mode = str(fusion_mode).lower()
    if mode not in {"concat", "gated"}:
        raise ValueError(f"Unsupported scalar fusion mode: {fusion_mode}")
    if scalar_feature_dim <= 0 and mode == "gated":
        return "concat"
    return mode


def _build_tiny_cnn_artifact(
    *,
    model_params: Dict[str, np.ndarray],
    input_shape: tuple[int, int],
    input_channels: int,
    scalar_feature_dim: int,
    label_names,
    config: TinyCNNConfig,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> Dict[str, np.ndarray]:
    artifact = {
        "model_type": np.array(["tiny_cnn_float"]),
        "training_backend": np.array([str(config.backend)]),
        "conv_w": model_params["conv_w"].astype(np.float32),
        "conv_b": model_params["conv_b"].astype(np.float32),
        "fc1_w": model_params["fc1_w"].astype(np.float32),
        "fc1_b": model_params["fc1_b"].astype(np.float32),
        "fc2_w": model_params["fc2_w"].astype(np.float32),
        "fc2_b": model_params["fc2_b"].astype(np.float32),
        "x_mean": x_mean.astype(np.float32).reshape(-1),
        "x_std": x_std.astype(np.float32).reshape(-1),
        "scalar_mean": scalar_mean.astype(np.float32).reshape(-1),
        "scalar_std": scalar_std.astype(np.float32).reshape(-1),
        "y_mean": y_mean.astype(np.float32).reshape(-1),
        "y_std": y_std.astype(np.float32).reshape(-1),
        "label_names": np.asarray(label_names),
        "input_shape": np.asarray([input_shape[0], input_shape[1]], dtype=np.int32),
        "input_channels": np.asarray([input_channels], dtype=np.int32),
        "scalar_feature_dim": np.asarray([scalar_feature_dim], dtype=np.int32),
        "conv_channels": np.asarray([config.conv_channels], dtype=np.int32),
        "kernel_size": np.asarray([config.kernel_size], dtype=np.int32),
        "hidden_dim": np.asarray([config.hidden_dim], dtype=np.int32),
        "scalar_fusion_mode": np.array([_resolve_scalar_fusion_mode(scalar_feature_dim, config.scalar_fusion_mode)]),
        "scalar_gate_init_bias": np.asarray([config.scalar_gate_init_bias], dtype=np.float32),
        "label_weights": np.asarray(
            [float((config.label_weights or {}).get(str(name), 1.0)) for name in label_names],
            dtype=np.float32,
        ),
    }
    for optional_name in ("scalar_gate_w", "scalar_gate_b", "scalar_shift_w", "scalar_shift_b"):
        if optional_name in model_params:
            artifact[optional_name] = model_params[optional_name].astype(np.float32)
    return artifact


class TinyCNN:
    """A minimal conv-relu-pool-mlp regressor implemented in NumPy."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_dim: int,
        config: TinyCNNConfig,
        *,
        input_channels: int = 1,
        scalar_feature_dim: int = 0,
    ) -> None:
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.config = config
        self.input_channels = int(input_channels)
        self.scalar_feature_dim = int(scalar_feature_dim)
        self.scalar_fusion_mode = _resolve_scalar_fusion_mode(self.scalar_feature_dim, config.scalar_fusion_mode)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.scalar_feature_dim < 0:
            raise ValueError("scalar_feature_dim must be non-negative")
        self.padding = _same_padding(config.kernel_size)
        pooled_h = input_shape[0] // 2
        pooled_w = input_shape[1] // 2
        self.conv_flat_dim = config.conv_channels * pooled_h * pooled_w
        flat_dim = self.conv_flat_dim + (self.scalar_feature_dim if self.scalar_fusion_mode == "concat" else 0)

        rng = np.random.default_rng(config.seed)
        self.params: Dict[str, np.ndarray] = {
            "conv_w": rng.normal(
                0.0,
                _he_std(self.input_channels * config.kernel_size * config.kernel_size),
                size=(config.conv_channels, self.input_channels, config.kernel_size, config.kernel_size),
            ),
            "conv_b": np.zeros(config.conv_channels, dtype=float),
            "fc1_w": rng.normal(0.0, _he_std(flat_dim), size=(flat_dim, config.hidden_dim)),
            "fc1_b": np.zeros(config.hidden_dim, dtype=float),
            "fc2_w": rng.normal(0.0, _he_std(config.hidden_dim), size=(config.hidden_dim, output_dim)),
            "fc2_b": np.zeros(output_dim, dtype=float),
        }
        if self.scalar_fusion_mode == "gated":
            self.params["scalar_gate_w"] = rng.normal(
                0.0,
                _he_std(self.scalar_feature_dim),
                size=(self.scalar_feature_dim, config.hidden_dim),
            )
            self.params["scalar_gate_b"] = np.full(config.hidden_dim, float(config.scalar_gate_init_bias), dtype=float)
            self.params["scalar_shift_w"] = rng.normal(
                0.0,
                _he_std(self.scalar_feature_dim),
                size=(self.scalar_feature_dim, config.hidden_dim),
            )
            self.params["scalar_shift_b"] = np.zeros(config.hidden_dim, dtype=float)

    def _to_nchw(self, x: np.ndarray) -> np.ndarray:
        """Normalize inputs to `(N, C, H, W)`."""
        array = np.asarray(x, dtype=float)
        if array.ndim == 3:
            return array[:, None, :, :]
        if array.ndim == 4:
            return array
        raise ValueError(f"Expected input with ndim 3 or 4, got shape {array.shape}")

    def _forward_internal(self, x: np.ndarray, scalar: np.ndarray | None = None):
        x_4d = self._to_nchw(x)
        cols = _im2col(x_4d, self.config.kernel_size, self.padding)
        conv_w = self.params["conv_w"].reshape(self.config.conv_channels, -1)
        conv_out = cols @ conv_w.T + self.params["conv_b"]
        conv_out = conv_out.reshape(x_4d.shape[0], self.input_shape[0], self.input_shape[1], self.config.conv_channels)
        conv_out = conv_out.transpose(0, 3, 1, 2)

        relu1 = np.maximum(conv_out, 0.0)
        pool = _avg_pool2x2_forward(relu1)
        conv_flat = pool.reshape(x.shape[0], -1)
        scalar_batch = None
        if self.scalar_feature_dim > 0:
            if scalar is None:
                raise ValueError("scalar input is required when scalar_feature_dim > 0")
            scalar_batch = np.asarray(scalar, dtype=float)
            if scalar_batch.ndim != 2 or scalar_batch.shape != (x.shape[0], self.scalar_feature_dim):
                raise ValueError(
                    f"Expected scalar shape {(x.shape[0], self.scalar_feature_dim)}, got {scalar_batch.shape}"
                )
        gate = None
        scalar_shift = None
        if self.scalar_fusion_mode == "gated":
            gate = _sigmoid(scalar_batch @ self.params["scalar_gate_w"] + self.params["scalar_gate_b"])
            scalar_shift = scalar_batch @ self.params["scalar_shift_w"] + self.params["scalar_shift_b"]
            hidden_pre = conv_flat @ self.params["fc1_w"] + self.params["fc1_b"]
            hidden_pre = hidden_pre * gate + scalar_shift
            flat = conv_flat
        else:
            flat = conv_flat
            if self.scalar_feature_dim > 0:
                flat = np.concatenate([flat, scalar_batch], axis=1)
            hidden_pre = flat @ self.params["fc1_w"] + self.params["fc1_b"]
        hidden = np.maximum(hidden_pre, 0.0)
        output = hidden @ self.params["fc2_w"] + self.params["fc2_b"]
        cache = {
            "x_4d": x_4d,
            "cols": cols,
            "conv_out": conv_out,
            "relu1": relu1,
            "pool": pool,
            "conv_flat": conv_flat,
            "flat": flat,
            "scalar": scalar_batch,
            "gate": gate,
            "scalar_shift": scalar_shift,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
        }
        return output, cache

    def predict_normalized(self, x: np.ndarray, scalar: np.ndarray | None = None) -> np.ndarray:
        output, _ = self._forward_internal(x, scalar=scalar)
        return output

    def loss_and_grads(
        self,
        x: np.ndarray,
        scalar: np.ndarray | None,
        y_true: np.ndarray,
        loss_weights: np.ndarray | None = None,
    ) -> tuple[float, Dict[str, np.ndarray], np.ndarray]:
        pred, cache = self._forward_internal(x, scalar=scalar)
        diff = pred - y_true
        batch_size = x.shape[0]
        if loss_weights is None:
            weight_vec = np.ones((1, y_true.shape[1]), dtype=float)
        else:
            weight_vec = np.asarray(loss_weights, dtype=float).reshape(1, -1)
        loss = float(np.mean(weight_vec * (diff**2)))
        loss += 0.5 * self.config.weight_decay * (
            float(np.sum(self.params["conv_w"] ** 2))
            + float(np.sum(self.params["fc1_w"] ** 2))
            + float(np.sum(self.params["fc2_w"] ** 2))
        )

        grad_pred = (2.0 / (batch_size * y_true.shape[1])) * weight_vec * diff
        grads: Dict[str, np.ndarray] = {}

        grads["fc2_w"] = cache["hidden"].T @ grad_pred + self.config.weight_decay * self.params["fc2_w"]
        grads["fc2_b"] = grad_pred.sum(axis=0)
        grad_hidden = grad_pred @ self.params["fc2_w"].T
        grad_hidden[cache["hidden_pre"] <= 0.0] = 0.0

        if self.scalar_fusion_mode == "gated":
            gate = cache["gate"]
            assert gate is not None
            grads["fc1_w"] = cache["conv_flat"].T @ (grad_hidden * gate) + self.config.weight_decay * self.params["fc1_w"]
            grads["fc1_b"] = (grad_hidden * gate).sum(axis=0)
            grad_conv_flat = (grad_hidden * gate) @ self.params["fc1_w"].T
            grad_gate = grad_hidden * (cache["conv_flat"] @ self.params["fc1_w"] + self.params["fc1_b"])
            grad_gate_pre = grad_gate * gate * (1.0 - gate)
            grads["scalar_gate_w"] = cache["scalar"].T @ grad_gate_pre + self.config.weight_decay * self.params["scalar_gate_w"]
            grads["scalar_gate_b"] = grad_gate_pre.sum(axis=0)
            grads["scalar_shift_w"] = cache["scalar"].T @ grad_hidden + self.config.weight_decay * self.params["scalar_shift_w"]
            grads["scalar_shift_b"] = grad_hidden.sum(axis=0)
        else:
            grads["fc1_w"] = cache["flat"].T @ grad_hidden + self.config.weight_decay * self.params["fc1_w"]
            grads["fc1_b"] = grad_hidden.sum(axis=0)
            grad_flat = grad_hidden @ self.params["fc1_w"].T
            if self.scalar_feature_dim > 0:
                grad_conv_flat = grad_flat[:, :-self.scalar_feature_dim]
            else:
                grad_conv_flat = grad_flat
        grad_pool = grad_conv_flat.reshape(cache["pool"].shape)
        grad_relu1 = _avg_pool2x2_backward(grad_pool, cache["relu1"].shape)
        grad_conv_out = grad_relu1.copy()
        grad_conv_out[cache["conv_out"] <= 0.0] = 0.0

        grad_conv_2d = grad_conv_out.transpose(0, 2, 3, 1).reshape(-1, self.config.conv_channels)
        conv_w = self.params["conv_w"].reshape(self.config.conv_channels, -1)
        grads["conv_w"] = (
            grad_conv_2d.T @ cache["cols"]
        ).reshape(self.params["conv_w"].shape) + self.config.weight_decay * self.params["conv_w"]
        grads["conv_b"] = grad_conv_2d.sum(axis=0)
        grad_cols = grad_conv_2d @ conv_w
        _ = _col2im(grad_cols, cache["x_4d"].shape, self.config.kernel_size, self.padding)
        return loss, grads, pred

    def apply_adam(self, grads: Dict[str, np.ndarray], state: Dict[str, Dict[str, np.ndarray]], step: int) -> None:
        for name, grad in grads.items():
            state["m"][name] = self.config.beta1 * state["m"][name] + (1.0 - self.config.beta1) * grad
            state["v"][name] = self.config.beta2 * state["v"][name] + (1.0 - self.config.beta2) * (grad**2)
            m_hat = state["m"][name] / (1.0 - self.config.beta1**step)
            v_hat = state["v"][name] / (1.0 - self.config.beta2**step)
            self.params[name] -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.adam_eps)


def _build_optimizer_state(params: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        "m": {name: np.zeros_like(value) for name, value in params.items()},
        "v": {name: np.zeros_like(value) for name, value in params.items()},
    }


def _prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names) -> Dict:
    return {
        "mse": _mse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "r2_mean": float(np.mean([_r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])),
        "per_label": {
            str(name): {
                "mse": _mse(y_true[:, i], y_pred[:, i]),
                "mae": _mae(y_true[:, i], y_pred[:, i]),
                "r2": _r2_score(y_true[:, i], y_pred[:, i]),
            }
            for i, name in enumerate(label_names)
        },
    }


def _fit_tiny_cnn_numpy(
    x_train: np.ndarray,
    x_train_scalar: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_val_scalar: np.ndarray,
    y_val: np.ndarray,
    label_names,
    config: TinyCNNConfig,
) -> tuple[Dict[str, np.ndarray], Dict]:
    """Train the Tiny-CNN with the legacy NumPy backend."""
    if x_train.ndim == 3:
        input_shape = (x_train.shape[1], x_train.shape[2])
        input_channels = 1
    elif x_train.ndim == 4:
        input_shape = (x_train.shape[2], x_train.shape[3])
        input_channels = int(x_train.shape[1])
    else:
        raise ValueError(f"Unsupported x_train shape for Tiny-CNN: {x_train.shape}")
    model = TinyCNN(
        input_shape=input_shape,
        output_dim=y_train.shape[1],
        config=config,
        input_channels=input_channels,
        scalar_feature_dim=int(x_train_scalar.shape[1]),
    )
    optimizer_state = _build_optimizer_state(model.params)
    rng = np.random.default_rng(config.seed)

    x_mean, x_std = _channelwise_input_stats(x_train)
    x_train_norm = _normalize_inputs(x_train, x_mean, x_std)
    x_val_norm = _normalize_inputs(x_val, x_mean, x_std)
    x_train_scalar = np.asarray(x_train_scalar, dtype=np.float64)
    x_val_scalar = np.asarray(x_val_scalar, dtype=np.float64)
    scalar_mean = x_train_scalar.mean(axis=0, keepdims=True) if x_train_scalar.shape[1] > 0 else np.zeros((1, 0), dtype=np.float64)
    scalar_std = x_train_scalar.std(axis=0, keepdims=True) if x_train_scalar.shape[1] > 0 else np.ones((1, 0), dtype=np.float64)
    scalar_std = np.where(scalar_std < 1.0e-8, 1.0, scalar_std)
    x_train_scalar_norm = (x_train_scalar - scalar_mean) / scalar_std if x_train_scalar.shape[1] > 0 else x_train_scalar
    x_val_scalar_norm = (x_val_scalar - scalar_mean) / scalar_std if x_val_scalar.shape[1] > 0 else x_val_scalar

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1.0e-8, 1.0, y_std)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    label_weights = _build_label_weights(label_names, config).astype(float)

    step = 0
    best_params = {name: value.copy() for name, value in model.params.items()}
    best_val_loss = float("inf")
    history = []
    patience_left = config.patience

    for epoch in range(config.epochs):
        order = rng.permutation(x_train.shape[0])
        epoch_losses = []
        for start in range(0, x_train.shape[0], config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            x_batch = x_train_norm[batch_idx]
            scalar_batch = x_train_scalar_norm[batch_idx]
            y_batch = y_train_norm[batch_idx]
            loss, grads, _ = model.loss_and_grads(x_batch, scalar_batch, y_batch, loss_weights=label_weights)
            step += 1
            model.apply_adam(grads, optimizer_state, step)
            epoch_losses.append(loss)

        train_pred_norm = model.predict_normalized(x_train_norm, x_train_scalar_norm)
        val_pred_norm = model.predict_normalized(x_val_norm, x_val_scalar_norm)
        train_loss = float(np.mean(label_weights.reshape(1, -1) * ((train_pred_norm - y_train_norm) ** 2)))
        val_loss = float(np.mean(label_weights.reshape(1, -1) * ((val_pred_norm - y_val_norm) ** 2)))
        history.append(
            {
                "epoch": epoch + 1,
                "batch_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "train_loss_norm": train_loss,
                "val_loss_norm": val_loss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {name: value.copy() for name, value in model.params.items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.params = best_params
    train_pred = model.predict_normalized(
        x_train_norm,
        x_train_scalar_norm if model.scalar_feature_dim > 0 else None,
    ) * y_std + y_mean
    val_pred = model.predict_normalized(
        x_val_norm,
        x_val_scalar_norm if model.scalar_feature_dim > 0 else None,
    ) * y_std + y_mean
    artifact = _build_tiny_cnn_artifact(
        model_params=model.params,
        input_shape=input_shape,
        input_channels=input_channels,
        scalar_feature_dim=int(x_train_scalar.shape[1]),
        label_names=label_names,
        config=config,
        x_mean=x_mean,
        x_std=x_std,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    report = {
        "backend": "numpy",
        "device": "cpu",
        "history": history,
        "train_metrics": _prediction_metrics(y_train, train_pred, label_names),
        "val_metrics": _prediction_metrics(y_val, val_pred, label_names),
    }
    return artifact, report


def _resolve_torch_device(device_name: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError("PyTorch backend requested but torch is not installed.") from exc

    requested = str(device_name).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch backend requested CUDA but torch.cuda.is_available() is False.")
    return torch.device(requested)


class _TorchTinyCNN:
    def __init__(
        self,
        input_shape: tuple[int, int],
        output_dim: int,
        config: TinyCNNConfig,
        *,
        input_channels: int,
        scalar_feature_dim: int,
    ) -> None:
        import torch

        self.torch = torch
        self.config = config
        self.scalar_feature_dim = int(scalar_feature_dim)
        self.scalar_fusion_mode = _resolve_scalar_fusion_mode(self.scalar_feature_dim, config.scalar_fusion_mode)
        self.padding = _same_padding(config.kernel_size)
        pooled_h = input_shape[0] // 2
        pooled_w = input_shape[1] // 2
        conv_flat_dim = config.conv_channels * pooled_h * pooled_w
        self.conv_flat_dim = conv_flat_dim
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, config.conv_channels, config.kernel_size, padding=self.padding),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Flatten(),
        )
        fc1_in_dim = conv_flat_dim + (self.scalar_feature_dim if self.scalar_fusion_mode == "concat" else 0)
        self.fc1 = torch.nn.Linear(fc1_in_dim, config.hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(config.hidden_dim, output_dim)
        self.scalar_gate = None
        self.scalar_shift = None
        if self.scalar_fusion_mode == "gated":
            self.scalar_gate = torch.nn.Linear(self.scalar_feature_dim, config.hidden_dim)
            self.scalar_shift = torch.nn.Linear(self.scalar_feature_dim, config.hidden_dim)
        self._reset_parameters(
            input_channels=input_channels,
            flat_dim=fc1_in_dim,
        )

    def _reset_parameters(self, *, input_channels: int, flat_dim: int) -> None:
        torch = self.torch
        with torch.no_grad():
            conv = self.feature_extractor[0]
            fc1 = self.fc1
            fc2 = self.fc2
            torch.nn.init.normal_(conv.weight, mean=0.0, std=_he_std(input_channels * self.config.kernel_size * self.config.kernel_size))
            torch.nn.init.zeros_(conv.bias)
            torch.nn.init.normal_(fc1.weight, mean=0.0, std=_he_std(flat_dim))
            torch.nn.init.zeros_(fc1.bias)
            torch.nn.init.normal_(fc2.weight, mean=0.0, std=_he_std(self.config.hidden_dim))
            torch.nn.init.zeros_(fc2.bias)
            if self.scalar_fusion_mode == "gated":
                assert self.scalar_gate is not None
                assert self.scalar_shift is not None
                torch.nn.init.normal_(self.scalar_gate.weight, mean=0.0, std=_he_std(self.scalar_feature_dim))
                torch.nn.init.constant_(self.scalar_gate.bias, float(self.config.scalar_gate_init_bias))
                torch.nn.init.normal_(self.scalar_shift.weight, mean=0.0, std=_he_std(self.scalar_feature_dim))
                torch.nn.init.zeros_(self.scalar_shift.bias)

    def predict_normalized(self, x_batch, scalar_batch=None):
        features = self.feature_extractor(x_batch)
        if self.scalar_fusion_mode == "gated":
            if scalar_batch is None:
                raise ValueError("scalar_batch is required when scalar_feature_dim > 0")
            assert self.scalar_gate is not None
            assert self.scalar_shift is not None
            base_hidden = self.fc1(features)
            gate = self.torch.sigmoid(self.scalar_gate(scalar_batch))
            hidden = self.relu(base_hidden * gate + self.scalar_shift(scalar_batch))
            return self.fc2(hidden)
        if self.scalar_feature_dim > 0:
            if scalar_batch is None:
                raise ValueError("scalar_batch is required when scalar_feature_dim > 0")
            features = self.torch.cat([features, scalar_batch], dim=1)
        hidden = self.relu(self.fc1(features))
        return self.fc2(hidden)


def _predict_torch_batches(model, x: np.ndarray, *, batch_size: int, device) -> np.ndarray:
    import torch

    outputs = []
    model.feature_extractor.eval()
    model.fc1.eval()
    model.fc2.eval()
    if model.scalar_gate is not None:
        model.scalar_gate.eval()
    if model.scalar_shift is not None:
        model.scalar_shift.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            x_batch = torch.from_numpy(x[start : start + batch_size].astype(np.float32, copy=False)).to(device)
            outputs.append(model.predict_normalized(x_batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def _predict_torch_batches_with_scalar(model, x: np.ndarray, scalar: np.ndarray, *, batch_size: int, device) -> np.ndarray:
    import torch

    outputs = []
    model.feature_extractor.eval()
    model.fc1.eval()
    model.fc2.eval()
    if model.scalar_gate is not None:
        model.scalar_gate.eval()
    if model.scalar_shift is not None:
        model.scalar_shift.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            x_batch = torch.from_numpy(x[start : start + batch_size].astype(np.float32, copy=False)).to(device)
            scalar_batch = torch.from_numpy(scalar[start : start + batch_size].astype(np.float32, copy=False)).to(device)
            outputs.append(model.predict_normalized(x_batch, scalar_batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def _torch_model_params(model) -> Dict[str, np.ndarray]:
    state = {
        **model.feature_extractor.state_dict(prefix="feature_extractor."),
        **model.fc1.state_dict(prefix="fc1."),
        **model.fc2.state_dict(prefix="fc2."),
    }
    if model.scalar_gate is not None:
        state.update(model.scalar_gate.state_dict(prefix="scalar_gate."))
    if model.scalar_shift is not None:
        state.update(model.scalar_shift.state_dict(prefix="scalar_shift."))
    params = {
        "conv_w": state["feature_extractor.0.weight"].detach().cpu().numpy(),
        "conv_b": state["feature_extractor.0.bias"].detach().cpu().numpy(),
        "fc1_w": state["fc1.weight"].detach().cpu().numpy().T,
        "fc1_b": state["fc1.bias"].detach().cpu().numpy(),
        "fc2_w": state["fc2.weight"].detach().cpu().numpy().T,
        "fc2_b": state["fc2.bias"].detach().cpu().numpy(),
    }
    if "scalar_gate.weight" in state:
        params["scalar_gate_w"] = state["scalar_gate.weight"].detach().cpu().numpy().T
        params["scalar_gate_b"] = state["scalar_gate.bias"].detach().cpu().numpy()
        params["scalar_shift_w"] = state["scalar_shift.weight"].detach().cpu().numpy().T
        params["scalar_shift_b"] = state["scalar_shift.bias"].detach().cpu().numpy()
    return params


def _fit_tiny_cnn_torch(
    x_train: np.ndarray,
    x_train_scalar: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_val_scalar: np.ndarray,
    y_val: np.ndarray,
    label_names,
    config: TinyCNNConfig,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Train the Tiny-CNN with a PyTorch backend and export the same NPZ artifact format."""
    import torch

    if x_train.ndim == 3:
        input_shape = (x_train.shape[1], x_train.shape[2])
        input_channels = 1
    elif x_train.ndim == 4:
        input_shape = (x_train.shape[2], x_train.shape[3])
        input_channels = int(x_train.shape[1])
    else:
        raise ValueError(f"Unsupported x_train shape for Tiny-CNN: {x_train.shape}")

    torch.manual_seed(config.seed)
    device = _resolve_torch_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    x_mean, x_std = _channelwise_input_stats(x_train)
    x_train_norm = _normalize_inputs(x_train, x_mean, x_std).astype(np.float32)
    x_val_norm = _normalize_inputs(x_val, x_mean, x_std).astype(np.float32)
    x_train_scalar = np.asarray(x_train_scalar, dtype=np.float32)
    x_val_scalar = np.asarray(x_val_scalar, dtype=np.float32)
    scalar_mean = x_train_scalar.mean(axis=0, keepdims=True) if x_train_scalar.shape[1] > 0 else np.zeros((1, 0), dtype=np.float32)
    scalar_std = x_train_scalar.std(axis=0, keepdims=True) if x_train_scalar.shape[1] > 0 else np.ones((1, 0), dtype=np.float32)
    scalar_std = np.where(scalar_std < 1.0e-8, 1.0, scalar_std)
    x_train_scalar_norm = (x_train_scalar - scalar_mean) / scalar_std if x_train_scalar.shape[1] > 0 else x_train_scalar
    x_val_scalar_norm = (x_val_scalar - scalar_mean) / scalar_std if x_val_scalar.shape[1] > 0 else x_val_scalar

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1.0e-8, 1.0, y_std)
    y_train_norm = ((y_train - y_mean) / y_std).astype(np.float32)
    y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32)
    label_weights = _build_label_weights(label_names, config).astype(np.float32)

    wrapper = _TorchTinyCNN(
        input_shape=input_shape,
        output_dim=y_train.shape[1],
        config=config,
        input_channels=input_channels,
        scalar_feature_dim=int(x_train_scalar.shape[1]),
    )
    wrapper.feature_extractor = wrapper.feature_extractor.to(device)
    wrapper.fc1 = wrapper.fc1.to(device)
    wrapper.relu = wrapper.relu.to(device)
    wrapper.fc2 = wrapper.fc2.to(device)
    if wrapper.scalar_gate is not None:
        wrapper.scalar_gate = wrapper.scalar_gate.to(device)
    if wrapper.scalar_shift is not None:
        wrapper.scalar_shift = wrapper.scalar_shift.to(device)
    weight_params = [wrapper.feature_extractor[0].weight, wrapper.fc1.weight, wrapper.fc2.weight]
    bias_params = [wrapper.feature_extractor[0].bias, wrapper.fc1.bias, wrapper.fc2.bias]
    if wrapper.scalar_gate is not None and wrapper.scalar_shift is not None:
        weight_params.extend([wrapper.scalar_gate.weight, wrapper.scalar_shift.weight])
        bias_params.extend([wrapper.scalar_gate.bias, wrapper.scalar_shift.bias])
    optimizer = torch.optim.Adam(
        [
            {"params": weight_params, "weight_decay": config.weight_decay},
            {"params": bias_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
    )

    rng = np.random.default_rng(config.seed)
    label_weights_t = torch.from_numpy(label_weights.reshape(1, -1)).to(device)
    best_state = {
        "feature_extractor": {name: value.detach().cpu().clone() for name, value in wrapper.feature_extractor.state_dict().items()},
        "fc1": {name: value.detach().cpu().clone() for name, value in wrapper.fc1.state_dict().items()},
        "fc2": {name: value.detach().cpu().clone() for name, value in wrapper.fc2.state_dict().items()},
    }
    if wrapper.scalar_gate is not None:
        best_state["scalar_gate"] = {name: value.detach().cpu().clone() for name, value in wrapper.scalar_gate.state_dict().items()}
    if wrapper.scalar_shift is not None:
        best_state["scalar_shift"] = {name: value.detach().cpu().clone() for name, value in wrapper.scalar_shift.state_dict().items()}
    best_val_loss = float("inf")
    patience_left = config.patience
    history = []

    for epoch in range(config.epochs):
        order = rng.permutation(x_train_norm.shape[0])
        epoch_losses = []
        wrapper.feature_extractor.train()
        wrapper.fc1.train()
        wrapper.fc2.train()
        if wrapper.scalar_gate is not None:
            wrapper.scalar_gate.train()
        if wrapper.scalar_shift is not None:
            wrapper.scalar_shift.train()
        for start in range(0, x_train_norm.shape[0], config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            x_batch = torch.from_numpy(x_train_norm[batch_idx]).to(device)
            scalar_batch = torch.from_numpy(x_train_scalar_norm[batch_idx]).to(device)
            y_batch = torch.from_numpy(y_train_norm[batch_idx]).to(device)
            pred = wrapper.predict_normalized(x_batch, scalar_batch if wrapper.scalar_feature_dim > 0 else None)
            loss = torch.mean(label_weights_t * (pred - y_batch) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        if wrapper.scalar_feature_dim > 0:
            train_pred_norm = _predict_torch_batches_with_scalar(wrapper, x_train_norm, x_train_scalar_norm, batch_size=config.batch_size, device=device)
            val_pred_norm = _predict_torch_batches_with_scalar(wrapper, x_val_norm, x_val_scalar_norm, batch_size=config.batch_size, device=device)
        else:
            train_pred_norm = _predict_torch_batches(wrapper, x_train_norm, batch_size=config.batch_size, device=device)
            val_pred_norm = _predict_torch_batches(wrapper, x_val_norm, batch_size=config.batch_size, device=device)
        train_loss = float(np.mean(label_weights.reshape(1, -1) * ((train_pred_norm - y_train_norm) ** 2)))
        val_loss = float(np.mean(label_weights.reshape(1, -1) * ((val_pred_norm - y_val_norm) ** 2)))
        history.append(
            {
                "epoch": epoch + 1,
                "batch_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "train_loss_norm": train_loss,
                "val_loss_norm": val_loss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "feature_extractor": {name: value.detach().cpu().clone() for name, value in wrapper.feature_extractor.state_dict().items()},
                "fc1": {name: value.detach().cpu().clone() for name, value in wrapper.fc1.state_dict().items()},
                "fc2": {name: value.detach().cpu().clone() for name, value in wrapper.fc2.state_dict().items()},
            }
            if wrapper.scalar_gate is not None:
                best_state["scalar_gate"] = {name: value.detach().cpu().clone() for name, value in wrapper.scalar_gate.state_dict().items()}
            if wrapper.scalar_shift is not None:
                best_state["scalar_shift"] = {name: value.detach().cpu().clone() for name, value in wrapper.scalar_shift.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    wrapper.feature_extractor.load_state_dict(best_state["feature_extractor"])
    wrapper.fc1.load_state_dict(best_state["fc1"])
    wrapper.fc2.load_state_dict(best_state["fc2"])
    if wrapper.scalar_gate is not None and "scalar_gate" in best_state:
        wrapper.scalar_gate.load_state_dict(best_state["scalar_gate"])
    if wrapper.scalar_shift is not None and "scalar_shift" in best_state:
        wrapper.scalar_shift.load_state_dict(best_state["scalar_shift"])
    if wrapper.scalar_feature_dim > 0:
        train_pred = _predict_torch_batches_with_scalar(wrapper, x_train_norm, x_train_scalar_norm, batch_size=config.batch_size, device=device) * y_std + y_mean
        val_pred = _predict_torch_batches_with_scalar(wrapper, x_val_norm, x_val_scalar_norm, batch_size=config.batch_size, device=device) * y_std + y_mean
    else:
        train_pred = _predict_torch_batches(wrapper, x_train_norm, batch_size=config.batch_size, device=device) * y_std + y_mean
        val_pred = _predict_torch_batches(wrapper, x_val_norm, batch_size=config.batch_size, device=device) * y_std + y_mean
    artifact = _build_tiny_cnn_artifact(
        model_params=_torch_model_params(wrapper),
        input_shape=input_shape,
        input_channels=input_channels,
        scalar_feature_dim=int(x_train_scalar.shape[1]),
        label_names=label_names,
        config=config,
        x_mean=x_mean,
        x_std=x_std,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    report = {
        "backend": "torch",
        "device": str(device),
        "history": history,
        "train_metrics": _prediction_metrics(y_train, train_pred, label_names),
        "val_metrics": _prediction_metrics(y_val, val_pred, label_names),
    }
    return artifact, report


def fit_tiny_cnn(
    x_train: np.ndarray,
    x_train_scalar: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_val_scalar: np.ndarray,
    y_val: np.ndarray,
    label_names,
    config: TinyCNNConfig,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Train the Tiny-CNN and return artifact payload + report."""
    backend = str(config.backend).lower()
    if backend == "numpy":
        return _fit_tiny_cnn_numpy(x_train, x_train_scalar, y_train, x_val, x_val_scalar, y_val, label_names, config)
    if backend == "torch":
        return _fit_tiny_cnn_torch(x_train, x_train_scalar, y_train, x_val, x_val_scalar, y_val, label_names, config)
    raise ValueError(f"Unsupported Tiny-CNN backend: {config.backend}")


def _load_float_params(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    params = {
        "conv_w": data["conv_w"].astype(np.float64),
        "conv_b": data["conv_b"].astype(np.float64),
        "fc1_w": data["fc1_w"].astype(np.float64),
        "fc1_b": data["fc1_b"].astype(np.float64),
        "fc2_w": data["fc2_w"].astype(np.float64),
        "fc2_b": data["fc2_b"].astype(np.float64),
    }
    for optional_name in ("scalar_gate_w", "scalar_gate_b", "scalar_shift_w", "scalar_shift_b"):
        if optional_name in data:
            params[optional_name] = data[optional_name].astype(np.float64)
    return params


def _load_quantized_params(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    params = {}
    for name in ("conv_w", "conv_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b", "scalar_gate_w", "scalar_gate_b", "scalar_shift_w", "scalar_shift_b"):
        q_name = f"q_{name}"
        scale_name = f"{name}_scale"
        if q_name in data and scale_name in data:
            params[name] = data[q_name].astype(np.float64) * float(data[scale_name][0])
    return params


def _predict_with_params(
    params: Dict[str, np.ndarray],
    histogram_batch: np.ndarray,
    scalar_batch: np.ndarray,
    artifact: Dict[str, np.ndarray],
) -> np.ndarray:
    input_shape = tuple(int(x) for x in artifact["input_shape"])
    input_channels = int(artifact["input_channels"][0]) if "input_channels" in artifact else 1
    scalar_feature_dim = int(artifact["scalar_feature_dim"][0]) if "scalar_feature_dim" in artifact else 0
    config = TinyCNNConfig(
        conv_channels=int(artifact["conv_channels"][0]),
        kernel_size=int(artifact["kernel_size"][0]),
        hidden_dim=int(artifact["hidden_dim"][0]),
        scalar_fusion_mode=str(artifact["scalar_fusion_mode"][0]) if "scalar_fusion_mode" in artifact else "concat",
        scalar_gate_init_bias=float(artifact["scalar_gate_init_bias"][0]) if "scalar_gate_init_bias" in artifact else 2.0,
    )
    model = TinyCNN(
        input_shape=input_shape,
        output_dim=int(artifact["y_mean"].shape[0]),
        config=config,
        input_channels=input_channels,
        scalar_feature_dim=scalar_feature_dim,
    )
    model.params = {name: value.astype(float).copy() for name, value in params.items()}
    x_mean = artifact["x_mean"].astype(float).reshape(-1) if "x_mean" in artifact else np.zeros((input_channels,), dtype=float)
    x_std = artifact["x_std"].astype(float).reshape(-1) if "x_std" in artifact else np.ones((input_channels,), dtype=float)
    scalar_mean = artifact["scalar_mean"].astype(float).reshape(1, -1) if "scalar_mean" in artifact else np.zeros((1, scalar_feature_dim), dtype=float)
    scalar_std = artifact["scalar_std"].astype(float).reshape(1, -1) if "scalar_std" in artifact else np.ones((1, scalar_feature_dim), dtype=float)
    normalized_inputs = _normalize_inputs(histogram_batch.astype(float), x_mean, x_std)
    scalar_inputs = np.asarray(scalar_batch, dtype=float).reshape(histogram_batch.shape[0], scalar_feature_dim) if scalar_feature_dim > 0 else np.zeros((histogram_batch.shape[0], 0), dtype=float)
    normalized_scalar = (scalar_inputs - scalar_mean) / scalar_std if scalar_feature_dim > 0 else scalar_inputs
    pred_norm = model.predict_normalized(normalized_inputs, normalized_scalar if scalar_feature_dim > 0 else None)
    y_mean = artifact["y_mean"].astype(float).reshape(1, -1)
    y_std = artifact["y_std"].astype(float).reshape(1, -1)
    return pred_norm * y_std + y_mean


def _gate_from_params(
    params: Dict[str, np.ndarray],
    normalized_scalar: np.ndarray,
) -> np.ndarray | None:
    if {"scalar_gate_w", "scalar_gate_b"}.issubset(params.keys()):
        return _sigmoid(normalized_scalar @ params["scalar_gate_w"] + params["scalar_gate_b"].reshape(1, -1))
    return None


def _normalize_artifact_input(
    histogram: np.ndarray | tuple[np.ndarray, np.ndarray] | list[np.ndarray],
    artifact: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Normalize one-sample or batch histogram input for artifact inference."""
    input_shape = tuple(int(x) for x in artifact["input_shape"])
    input_channels = int(artifact["input_channels"][0]) if "input_channels" in artifact else 1
    scalar_feature_dim = int(artifact["scalar_feature_dim"][0]) if "scalar_feature_dim" in artifact else 0
    if isinstance(histogram, (tuple, list)) and len(histogram) == 2:
        spatial_input = np.asarray(histogram[0], dtype=np.float64)
        scalar_input = np.asarray(histogram[1], dtype=np.float64)
    else:
        spatial_input = np.asarray(histogram, dtype=np.float64)
        scalar_input = np.zeros((0,), dtype=np.float64)
    array = spatial_input
    if array.ndim == 2:
        if input_channels != 1:
            raise ValueError(f"Artifact expects {input_channels} channels, got single 2D histogram")
        batch = array[None, :, :]
        single_sample = True
    elif array.ndim == 3:
        if array.shape == (input_channels, input_shape[0], input_shape[1]):
            batch = array[None, :, :, :]
            single_sample = True
        elif input_channels == 1 and array.shape[1:] == input_shape:
            batch = array
            single_sample = False
        else:
            raise ValueError(f"Unsupported 3D histogram input shape {array.shape} for input_channels={input_channels}")
    elif array.ndim == 4:
        if array.shape[1:] != (input_channels, input_shape[0], input_shape[1]):
            raise ValueError(
                f"Unsupported 4D histogram batch shape {array.shape} for input_channels={input_channels}"
            )
        batch = array
        single_sample = False
    else:
        raise ValueError(f"Unsupported histogram ndim for Tiny-CNN inference: {array.ndim}")

    if scalar_feature_dim <= 0:
        scalar_batch = np.zeros((batch.shape[0], 0), dtype=np.float64)
    else:
        scalar_array = np.asarray(scalar_input, dtype=np.float64)
        if scalar_array.ndim == 1:
            if scalar_array.size != scalar_feature_dim:
                raise ValueError(f"Expected scalar feature dim {scalar_feature_dim}, got {scalar_array.shape}")
            scalar_batch = scalar_array.reshape(1, -1) if single_sample else np.repeat(scalar_array.reshape(1, -1), batch.shape[0], axis=0)
        elif scalar_array.ndim == 2:
            if scalar_array.shape != (batch.shape[0], scalar_feature_dim):
                raise ValueError(f"Expected scalar batch shape {(batch.shape[0], scalar_feature_dim)}, got {scalar_array.shape}")
            scalar_batch = scalar_array
        else:
            raise ValueError(f"Unsupported scalar feature ndim: {scalar_array.ndim}")
    return batch, scalar_batch, single_sample


def explain_from_loaded_artifact(
    data,
    histogram: np.ndarray | tuple[np.ndarray, np.ndarray] | list[np.ndarray],
) -> Dict[str, Any]:
    """Return prediction plus scalar-branch diagnostics for one sample or a batch."""
    model_type = str(data["model_type"][0])
    histogram_batch, scalar_batch, single_sample = _normalize_artifact_input(histogram, data)
    scalar_feature_dim = int(data["scalar_feature_dim"][0]) if "scalar_feature_dim" in data else 0
    scalar_fusion_mode = str(data["scalar_fusion_mode"][0]) if "scalar_fusion_mode" in data else "concat"
    label_names = [str(item) for item in data["label_names"]]

    if model_type == "tiny_cnn_float":
        params = _load_float_params(data)
    elif model_type == "tiny_cnn_int8":
        params = _load_quantized_params(data)
    else:
        raise ValueError(f"Unsupported Tiny-CNN artifact type for explain: {model_type}")

    prediction = _predict_with_params(params, histogram_batch, scalar_batch, data)
    result: Dict[str, Any] = {
        "prediction": prediction[0].tolist() if single_sample else prediction.tolist(),
        "label_names": label_names,
        "scalar_feature_dim": scalar_feature_dim,
        "scalar_fusion_mode": scalar_fusion_mode,
    }
    if scalar_feature_dim <= 0:
        return result

    scalar_mean = data["scalar_mean"].astype(float).reshape(1, -1) if "scalar_mean" in data else np.zeros((1, scalar_feature_dim), dtype=float)
    scalar_std = data["scalar_std"].astype(float).reshape(1, -1) if "scalar_std" in data else np.ones((1, scalar_feature_dim), dtype=float)
    baseline_scalar = np.repeat(scalar_mean, scalar_batch.shape[0], axis=0)
    prediction_without_teacher = _predict_with_params(params, histogram_batch, baseline_scalar, data)
    contribution = prediction - prediction_without_teacher
    scalar_names = [str(item) for item in data["scalar_feature_names"]] if "scalar_feature_names" in data else []
    normalized_scalar = (scalar_batch - scalar_mean) / scalar_std
    result.update(
        {
            "scalar_feature_names": scalar_names,
            "scalar_features_raw": scalar_batch[0].tolist() if single_sample else scalar_batch.tolist(),
            "scalar_features_normalized": normalized_scalar[0].tolist() if single_sample else normalized_scalar.tolist(),
            "prediction_without_teacher": prediction_without_teacher[0].tolist() if single_sample else prediction_without_teacher.tolist(),
            "teacher_contribution": contribution[0].tolist() if single_sample else contribution.tolist(),
            "teacher_contribution_l2": float(np.linalg.norm(contribution[0] if single_sample else contribution, ord=2)),
        }
    )
    per_scalar = []
    for idx in range(scalar_feature_dim):
        ablated_scalar = scalar_batch.copy()
        ablated_scalar[:, idx] = scalar_mean.reshape(-1)[idx]
        ablated_prediction = _predict_with_params(params, histogram_batch, ablated_scalar, data)
        delta = prediction - ablated_prediction
        delta_array = delta[0] if single_sample else delta
        item = {
            "name": scalar_names[idx] if idx < len(scalar_names) else f"scalar_{idx}",
            "raw": float(scalar_batch[0, idx]) if single_sample else scalar_batch[:, idx].tolist(),
            "normalized": float(normalized_scalar[0, idx]) if single_sample else normalized_scalar[:, idx].tolist(),
            "ablation_delta": delta_array.tolist(),
            "ablation_l2": float(np.linalg.norm(delta_array, ord=2)),
        }
        per_scalar.append(item)
    result["per_scalar_contribution"] = per_scalar

    if scalar_fusion_mode == "gated" and {"scalar_gate_w", "scalar_gate_b"}.issubset(params.keys()):
        gate = _gate_from_params(params, normalized_scalar)
        assert gate is not None
        gate_array = gate[0] if single_sample else gate
        result["teacher_gate"] = gate_array.tolist()
        result["teacher_gate_mean"] = float(np.mean(gate_array))
        result["teacher_gate_std"] = float(np.std(gate_array))
        result["teacher_gate_min"] = float(np.min(gate_array))
        result["teacher_gate_max"] = float(np.max(gate_array))
        gate_effects = []
        for idx in range(scalar_feature_dim):
            ablated_normalized_scalar = normalized_scalar.copy()
            ablated_normalized_scalar[:, idx] = 0.0
            ablated_gate = _gate_from_params(params, ablated_normalized_scalar)
            if ablated_gate is None:
                continue
            delta_gate = gate - ablated_gate
            delta_array = delta_gate[0] if single_sample else delta_gate
            gate_effects.append(
                {
                    "name": scalar_names[idx] if idx < len(scalar_names) else f"scalar_{idx}",
                    "gate_delta_mean": float(np.mean(delta_array)),
                    "gate_delta_l2": float(np.linalg.norm(delta_array, ord=2)),
                }
            )
        result["per_scalar_gate_effect"] = gate_effects
    return result


def predict_from_loaded_artifact(data, histogram: np.ndarray | tuple[np.ndarray, np.ndarray] | list[np.ndarray]) -> np.ndarray:
    """Predict one sample or a batch from an already loaded Tiny-CNN artifact."""
    model_type = str(data["model_type"][0])
    histogram_batch, scalar_batch, single_sample = _normalize_artifact_input(histogram, data)

    if model_type == "tiny_cnn_float":
        params = _load_float_params(data)
    elif model_type == "tiny_cnn_int8":
        params = _load_quantized_params(data)
    else:
        raise ValueError(f"Unsupported Tiny-CNN artifact type: {model_type}")

    prediction = _predict_with_params(params, histogram_batch, scalar_batch, data)
    if single_sample:
        return prediction[0]
    return prediction


def predict_from_artifact(path: str | Path, histogram: np.ndarray | tuple[np.ndarray, np.ndarray] | list[np.ndarray]) -> np.ndarray:
    """Predict one sample or a batch from a saved Tiny-CNN artifact."""
    data = np.load(Path(path).expanduser().resolve(), allow_pickle=True)
    return predict_from_loaded_artifact(data, histogram)
