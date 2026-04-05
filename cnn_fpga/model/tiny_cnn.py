"""Pure-NumPy Tiny-CNN used as the first main model path for CNN-FPGA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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


class TinyCNN:
    """A minimal conv-relu-pool-mlp regressor implemented in NumPy."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_dim: int,
        config: TinyCNNConfig,
        *,
        input_channels: int = 1,
    ) -> None:
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.config = config
        self.input_channels = int(input_channels)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        self.padding = _same_padding(config.kernel_size)
        pooled_h = input_shape[0] // 2
        pooled_w = input_shape[1] // 2
        flat_dim = config.conv_channels * pooled_h * pooled_w

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

    def _to_nchw(self, x: np.ndarray) -> np.ndarray:
        """Normalize inputs to `(N, C, H, W)`."""
        array = np.asarray(x, dtype=float)
        if array.ndim == 3:
            return array[:, None, :, :]
        if array.ndim == 4:
            return array
        raise ValueError(f"Expected input with ndim 3 or 4, got shape {array.shape}")

    def _forward_internal(self, x: np.ndarray):
        x_4d = self._to_nchw(x)
        cols = _im2col(x_4d, self.config.kernel_size, self.padding)
        conv_w = self.params["conv_w"].reshape(self.config.conv_channels, -1)
        conv_out = cols @ conv_w.T + self.params["conv_b"]
        conv_out = conv_out.reshape(x_4d.shape[0], self.input_shape[0], self.input_shape[1], self.config.conv_channels)
        conv_out = conv_out.transpose(0, 3, 1, 2)

        relu1 = np.maximum(conv_out, 0.0)
        pool = _avg_pool2x2_forward(relu1)
        flat = pool.reshape(x.shape[0], -1)
        hidden_pre = flat @ self.params["fc1_w"] + self.params["fc1_b"]
        hidden = np.maximum(hidden_pre, 0.0)
        output = hidden @ self.params["fc2_w"] + self.params["fc2_b"]
        cache = {
            "x_4d": x_4d,
            "cols": cols,
            "conv_out": conv_out,
            "relu1": relu1,
            "pool": pool,
            "flat": flat,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
        }
        return output, cache

    def predict_normalized(self, x: np.ndarray) -> np.ndarray:
        output, _ = self._forward_internal(x)
        return output

    def loss_and_grads(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        loss_weights: np.ndarray | None = None,
    ) -> tuple[float, Dict[str, np.ndarray], np.ndarray]:
        pred, cache = self._forward_internal(x)
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

        grads["fc1_w"] = cache["flat"].T @ grad_hidden + self.config.weight_decay * self.params["fc1_w"]
        grads["fc1_b"] = grad_hidden.sum(axis=0)
        grad_flat = grad_hidden @ self.params["fc1_w"].T
        grad_pool = grad_flat.reshape(cache["pool"].shape)
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


def fit_tiny_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    label_names,
    config: TinyCNNConfig,
) -> tuple[Dict[str, np.ndarray], Dict]:
    """Train the Tiny-CNN and return artifact payload + report."""
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
    )
    optimizer_state = _build_optimizer_state(model.params)
    rng = np.random.default_rng(config.seed)

    x_mean, x_std = _channelwise_input_stats(x_train)
    x_train_norm = _normalize_inputs(x_train, x_mean, x_std)
    x_val_norm = _normalize_inputs(x_val, x_mean, x_std)

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1.0e-8, 1.0, y_std)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    label_weights = np.ones(y_train.shape[1], dtype=float)
    if config.label_weights:
        for idx, name in enumerate(label_names):
            label_weights[idx] = float(config.label_weights.get(str(name), 1.0))

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
            y_batch = y_train_norm[batch_idx]
            loss, grads, _ = model.loss_and_grads(x_batch, y_batch, loss_weights=label_weights)
            step += 1
            model.apply_adam(grads, optimizer_state, step)
            epoch_losses.append(loss)

        train_pred_norm = model.predict_normalized(x_train_norm)
        val_pred_norm = model.predict_normalized(x_val_norm)
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
    train_pred = model.predict_normalized(x_train) * y_std + y_mean
    val_pred = model.predict_normalized(x_val) * y_std + y_mean
    artifact = {
        "model_type": np.array(["tiny_cnn_float"]),
        "conv_w": model.params["conv_w"].astype(np.float32),
        "conv_b": model.params["conv_b"].astype(np.float32),
        "fc1_w": model.params["fc1_w"].astype(np.float32),
        "fc1_b": model.params["fc1_b"].astype(np.float32),
        "fc2_w": model.params["fc2_w"].astype(np.float32),
        "fc2_b": model.params["fc2_b"].astype(np.float32),
        "x_mean": x_mean.astype(np.float32).reshape(-1),
        "x_std": x_std.astype(np.float32).reshape(-1),
        "y_mean": y_mean.astype(np.float32).reshape(-1),
        "y_std": y_std.astype(np.float32).reshape(-1),
        "label_names": np.asarray(label_names),
        "input_shape": np.asarray([input_shape[0], input_shape[1]], dtype=np.int32),
        "input_channels": np.asarray([input_channels], dtype=np.int32),
        "conv_channels": np.asarray([config.conv_channels], dtype=np.int32),
        "kernel_size": np.asarray([config.kernel_size], dtype=np.int32),
        "hidden_dim": np.asarray([config.hidden_dim], dtype=np.int32),
        "label_weights": np.asarray(
            [float((config.label_weights or {}).get(str(name), 1.0)) for name in label_names],
            dtype=np.float32,
        ),
    }
    report = {
        "history": history,
        "train_metrics": _prediction_metrics(y_train, train_pred, label_names),
        "val_metrics": _prediction_metrics(y_val, val_pred, label_names),
    }
    return artifact, report


def _load_float_params(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "conv_w": data["conv_w"].astype(np.float64),
        "conv_b": data["conv_b"].astype(np.float64),
        "fc1_w": data["fc1_w"].astype(np.float64),
        "fc1_b": data["fc1_b"].astype(np.float64),
        "fc2_w": data["fc2_w"].astype(np.float64),
        "fc2_b": data["fc2_b"].astype(np.float64),
    }


def _load_quantized_params(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    params = {}
    for name in ("conv_w", "conv_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b"):
        q_name = f"q_{name}"
        scale_name = f"{name}_scale"
        params[name] = data[q_name].astype(np.float64) * float(data[scale_name][0])
    return params


def _predict_with_params(params: Dict[str, np.ndarray], histogram_batch: np.ndarray, artifact: Dict[str, np.ndarray]) -> np.ndarray:
    input_shape = tuple(int(x) for x in artifact["input_shape"])
    input_channels = int(artifact["input_channels"][0]) if "input_channels" in artifact else 1
    config = TinyCNNConfig(
        conv_channels=int(artifact["conv_channels"][0]),
        kernel_size=int(artifact["kernel_size"][0]),
        hidden_dim=int(artifact["hidden_dim"][0]),
    )
    model = TinyCNN(
        input_shape=input_shape,
        output_dim=int(artifact["y_mean"].shape[0]),
        config=config,
        input_channels=input_channels,
    )
    model.params = {name: value.astype(float).copy() for name, value in params.items()}
    x_mean = artifact["x_mean"].astype(float).reshape(-1) if "x_mean" in artifact else np.zeros((input_channels,), dtype=float)
    x_std = artifact["x_std"].astype(float).reshape(-1) if "x_std" in artifact else np.ones((input_channels,), dtype=float)
    normalized_inputs = _normalize_inputs(histogram_batch.astype(float), x_mean, x_std)
    pred_norm = model.predict_normalized(normalized_inputs)
    y_mean = artifact["y_mean"].astype(float).reshape(1, -1)
    y_std = artifact["y_std"].astype(float).reshape(1, -1)
    return pred_norm * y_std + y_mean


def _normalize_artifact_input(histogram: np.ndarray, artifact: Dict[str, np.ndarray]) -> tuple[np.ndarray, bool]:
    """Normalize one-sample or batch histogram input for artifact inference."""
    input_shape = tuple(int(x) for x in artifact["input_shape"])
    input_channels = int(artifact["input_channels"][0]) if "input_channels" in artifact else 1
    array = np.asarray(histogram, dtype=np.float64)
    if array.ndim == 2:
        if input_channels != 1:
            raise ValueError(f"Artifact expects {input_channels} channels, got single 2D histogram")
        return array[None, :, :], True
    if array.ndim == 3:
        if array.shape == (input_channels, input_shape[0], input_shape[1]):
            return array[None, :, :, :], True
        if input_channels == 1 and array.shape[1:] == input_shape:
            return array, False
        raise ValueError(f"Unsupported 3D histogram input shape {array.shape} for input_channels={input_channels}")
    if array.ndim == 4:
        if array.shape[1:] != (input_channels, input_shape[0], input_shape[1]):
            raise ValueError(
                f"Unsupported 4D histogram batch shape {array.shape} for input_channels={input_channels}"
            )
        return array, False
    raise ValueError(f"Unsupported histogram ndim for Tiny-CNN inference: {array.ndim}")


def predict_from_loaded_artifact(data, histogram: np.ndarray) -> np.ndarray:
    """Predict one sample or a batch from an already loaded Tiny-CNN artifact."""
    model_type = str(data["model_type"][0])
    histogram_batch, single_sample = _normalize_artifact_input(histogram, data)

    if model_type == "tiny_cnn_float":
        params = _load_float_params(data)
    elif model_type == "tiny_cnn_int8":
        params = _load_quantized_params(data)
    else:
        raise ValueError(f"Unsupported Tiny-CNN artifact type: {model_type}")

    prediction = _predict_with_params(params, histogram_batch, data)
    if single_sample:
        return prediction[0]
    return prediction


def predict_from_artifact(path: str | Path, histogram: np.ndarray) -> np.ndarray:
    """Predict one sample or a batch from a saved Tiny-CNN artifact."""
    data = np.load(Path(path).expanduser().resolve(), allow_pickle=True)
    return predict_from_loaded_artifact(data, histogram)
