"""Inference-service abstractions for slow-loop deployment modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

from cnn_fpga.decoder.param_mapper import NoisePrediction
from cnn_fpga.utils.config import get_path


class InferenceServiceError(RuntimeError):
    """Structured inference-service failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class InferenceServiceConfig:
    """Configuration for selecting and launching slow-loop inference services."""

    mode: str = "inproc"
    backend: str = "artifact_npz"
    model_path: Optional[str] = None
    use_latest_model_dir_artifact: bool = True
    artifact_selector: str = "latest_any"
    python_executable: Optional[str] = None
    worker_module: str = "cnn_fpga.runtime.inference_worker"
    startup_timeout_s: float = 5.0
    request_timeout_s: float = 10.0
    tflite_path: Optional[str] = None
    label_names: Sequence[str] = ("sigma", "mu_q", "mu_p", "theta_deg")

    @classmethod
    def from_runtime_config(
        cls,
        config: Dict[str, Any],
        *,
        default_model_path: Optional[str],
        default_selector: str,
        default_use_latest_model_dir_artifact: bool,
    ) -> "InferenceServiceConfig":
        slow_cfg = config.get("slow_loop", {})
        service_cfg = slow_cfg.get("inference_service", {})
        tflite_cfg = service_cfg.get("tflite", {})
        raw_labels = service_cfg.get("label_names", ["sigma", "mu_q", "mu_p", "theta_deg"])
        return cls(
            mode=str(service_cfg.get("mode", "inproc")).lower(),
            backend=str(service_cfg.get("backend", "artifact_npz")).lower(),
            model_path=service_cfg.get("model_path", default_model_path),
            use_latest_model_dir_artifact=bool(
                service_cfg.get("use_latest_model_dir_artifact", default_use_latest_model_dir_artifact)
            ),
            artifact_selector=str(service_cfg.get("artifact_selector", default_selector)).lower(),
            python_executable=service_cfg.get("python_executable"),
            worker_module=str(service_cfg.get("worker_module", "cnn_fpga.runtime.inference_worker")),
            startup_timeout_s=float(service_cfg.get("startup_timeout_s", 5.0)),
            request_timeout_s=float(service_cfg.get("request_timeout_s", 10.0)),
            tflite_path=tflite_cfg.get("path"),
            label_names=tuple(str(item) for item in raw_labels),
        )


def resolve_model_path(
    config: Dict[str, Any],
    *,
    model_path: Optional[str],
    use_latest_model_dir_artifact: bool,
    artifact_selector: str,
) -> Path:
    """Resolve model path from config or the latest artifact directory."""
    if model_path:
        return Path(model_path).expanduser().resolve()

    if not use_latest_model_dir_artifact:
        raise InferenceServiceError("model_path_missing")

    model_dir = get_path(config, "model_dir", "artifacts/models/default")
    candidates = sorted(model_dir.glob("*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise InferenceServiceError(f"no_model_artifact_in_dir:{model_dir}")

    selector = artifact_selector.lower()
    if selector == "latest_any":
        return candidates[-1]
    if selector == "latest_float":
        filtered = [path for path in candidates if "_int8_" not in path.name]
        if not filtered:
            raise InferenceServiceError(f"no_float_model_artifact_in_dir:{model_dir}")
        return filtered[-1]
    if selector == "latest_int8":
        filtered = [path for path in candidates if "_int8_" in path.name]
        if not filtered:
            raise InferenceServiceError(f"no_int8_model_artifact_in_dir:{model_dir}")
        return filtered[-1]
    raise InferenceServiceError(f"unsupported_artifact_selector:{artifact_selector}")


def resolve_tflite_path(
    config: Dict[str, Any],
    *,
    tflite_path: Optional[str],
    use_latest_model_dir_artifact: bool,
    artifact_selector: str,
) -> Path:
    """Resolve `.tflite` or `.tflite.json` path from config or model directory."""
    if tflite_path:
        return Path(tflite_path).expanduser().resolve()

    if not use_latest_model_dir_artifact:
        raise InferenceServiceError("tflite_path_missing")

    model_dir = get_path(config, "model_dir", "artifacts/models/default")
    candidates = sorted(
        list(model_dir.glob("*.tflite")) + list(model_dir.glob("*.tflite.json")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise InferenceServiceError(f"no_tflite_artifact_in_dir:{model_dir}")

    selector = artifact_selector.lower()
    if selector == "latest_any":
        return candidates[-1]
    if selector == "latest_float":
        filtered = [path for path in candidates if "_int8_" not in path.name]
        if not filtered:
            raise InferenceServiceError(f"no_float_tflite_artifact_in_dir:{model_dir}")
        return filtered[-1]
    if selector == "latest_int8":
        filtered = [path for path in candidates if "_int8_" in path.name]
        if not filtered:
            raise InferenceServiceError(f"no_int8_tflite_artifact_in_dir:{model_dir}")
        return filtered[-1]
    raise InferenceServiceError(f"unsupported_artifact_selector:{artifact_selector}")


class BaseHistogramPredictor(ABC):
    """Abstract histogram -> noise-parameter predictor."""

    @abstractmethod
    def predict(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> NoisePrediction:
        raise NotImplementedError

    def explain(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        prediction = self.predict(histogram)
        return {
            "prediction": prediction.to_dict(),
        }


class ArtifactHistogramPredictor(BaseHistogramPredictor):
    """Predictor for float/int8 `.npz` artifacts."""

    def __init__(self, artifact_path: Path) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        self.artifact_path = artifact_path
        self.data = np.load(artifact_path, allow_pickle=True)
        self.model_type = str(self.data["model_type"][0])
        self.label_names = [str(item) for item in self.data["label_names"]]

    def _predict_linear(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        if isinstance(histogram, tuple):
            histogram = histogram[0]
        x = histogram.reshape(1, -1).astype(np.float64)
        x_mean = self.data["x_mean"].astype(np.float64).reshape(1, -1)
        x_std = self.data["x_std"].astype(np.float64).reshape(1, -1)
        x_norm = (x - x_mean) / x_std

        if self.model_type == "linear_regression_baseline":
            weights = self.data["weights"].astype(np.float64)
            bias = self.data["bias"].astype(np.float64)
            return (x_norm @ weights + bias).reshape(-1)

        if self.model_type == "linear_regression_int8":
            weights = self.data["q_weights"].astype(np.float64) * float(self.data["w_scale"][0])
            bias = self.data["q_bias"].astype(np.float64) * float(self.data["b_scale"][0])
            return (x_norm @ weights + bias).reshape(-1)

        raise InferenceServiceError(f"unsupported_linear_artifact:{self.model_type}")

    def _predict_tiny_cnn(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        try:
            from cnn_fpga.model.tiny_cnn import predict_from_loaded_artifact
        except ImportError as exc:  # pragma: no cover
            raise InferenceServiceError("tiny_cnn_artifact_support_unavailable") from exc
        if isinstance(histogram, tuple):
            spatial, scalar = histogram
            model_input = (
                np.asarray(spatial, dtype=np.float32),
                np.asarray(scalar, dtype=np.float32),
            )
        else:
            model_input = np.asarray(histogram, dtype=np.float32)
        return np.asarray(predict_from_loaded_artifact(self.data, model_input), dtype=float).reshape(-1)

    def predict(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> NoisePrediction:
        if self.model_type.startswith("linear_regression"):
            prediction = self._predict_linear(histogram)
        elif self.model_type.startswith("tiny_cnn"):
            prediction = self._predict_tiny_cnn(histogram)
        else:
            raise InferenceServiceError(f"unsupported_model_artifact:{self.model_type}")

        mapping = dict(zip(self.label_names, prediction.tolist()))
        return NoisePrediction(
            sigma=float(mapping.get("sigma", 0.0)),
            mu_q=float(mapping.get("mu_q", 0.0)),
            mu_p=float(mapping.get("mu_p", 0.0)),
            theta_deg=float(mapping.get("theta_deg", 0.0)),
            source="model_artifact",
            metadata={
                "artifact_path": str(self.artifact_path),
                "model_type": self.model_type,
                "label_names": list(self.label_names),
                "raw_prediction": mapping,
            },
        )

    def explain(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        if not self.model_type.startswith("tiny_cnn"):
            return {"prediction": self.predict(histogram).to_dict()}
        try:
            from cnn_fpga.model.tiny_cnn import explain_from_loaded_artifact
        except ImportError as exc:  # pragma: no cover
            raise InferenceServiceError("tiny_cnn_artifact_support_unavailable") from exc
        model_input: np.ndarray | tuple[np.ndarray, np.ndarray]
        if isinstance(histogram, tuple):
            model_input = (
                np.asarray(histogram[0], dtype=np.float32),
                np.asarray(histogram[1], dtype=np.float32),
            )
        else:
            model_input = np.asarray(histogram, dtype=np.float32)
        explanation = explain_from_loaded_artifact(self.data, model_input)
        explanation["artifact_path"] = str(self.artifact_path)
        explanation["model_type"] = self.model_type
        return explanation


class TFLiteHistogramPredictor(BaseHistogramPredictor):
    """Predictor for `.tflite` artifacts when runtime support is installed."""

    def __init__(self, model_path: Path, label_names: Sequence[str]) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        self.model_path = model_path
        self.label_names = [str(item) for item in label_names]
        self._stub_predictor: ArtifactHistogramPredictor | None = None
        self._stub_manifest: Dict[str, Any] | None = None

        if model_path.suffixes[-2:] == [".tflite", ".json"] or model_path.suffix == ".json":
            manifest = json.loads(model_path.read_text(encoding="utf-8"))
            if manifest.get("format") != "tflite_stub_v1":
                raise InferenceServiceError(f"unsupported_tflite_manifest:{model_path}")
            weights_path = Path(manifest["weights_path"]).expanduser().resolve()
            self._stub_manifest = manifest
            self._stub_predictor = ArtifactHistogramPredictor(weights_path)
            return

        interpreter_cls = None
        try:  # pragma: no cover
            from tflite_runtime.interpreter import Interpreter as interpreter_cls  # type: ignore
        except ImportError:
            try:
                import tensorflow as tf  # type: ignore

                interpreter_cls = tf.lite.Interpreter
            except ImportError as exc:
                raise InferenceServiceError("tflite_runtime_unavailable") from exc

        self.interpreter = interpreter_cls(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def predict(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> NoisePrediction:
        if self._stub_predictor is not None:
            prediction = self._stub_predictor.predict(histogram)
            metadata = dict(prediction.metadata or {})
            metadata.update(
                {
                    "tflite_mode": "stub_manifest",
                    "tflite_manifest_path": str(self.model_path),
                    "tflite_manifest": dict(self._stub_manifest or {}),
                }
            )
            return NoisePrediction(
                sigma=prediction.sigma,
                mu_q=prediction.mu_q,
                mu_p=prediction.mu_p,
                theta_deg=prediction.theta_deg,
                source="tflite_stub_service",
                metadata=metadata,
            )

        scalar_array: np.ndarray | None = None
        if isinstance(histogram, tuple):
            histogram, scalar_array = histogram
        input_array = np.asarray(histogram, dtype=np.float32)
        input_shape = tuple(int(x) for x in self.input_details["shape"])
        if input_array.ndim == 2:
            if len(input_shape) != 4 or input_shape[-1] != 1:
                raise InferenceServiceError(f"tflite_input_shape_mismatch:{input_array.shape}:{input_shape}")
            input_tensor = input_array.reshape(input_shape[1], input_shape[2], 1)[None, ...]
        elif input_array.ndim == 3:
            # 中文注释：
            # - 运行时上下文输入按 `(C, H, W)` 存储；
            # - TFLite / Keras 使用 `(H, W, C)`，因此这里做一次显式转置。
            if len(input_shape) != 4:
                raise InferenceServiceError(f"tflite_input_shape_mismatch:{input_array.shape}:{input_shape}")
            if input_array.shape[0] != input_shape[-1]:
                raise InferenceServiceError(f"tflite_context_channels_mismatch:{input_array.shape}:{input_shape}")
            input_tensor = np.transpose(input_array, (1, 2, 0))[None, ...]
        else:
            raise InferenceServiceError(f"unsupported_tflite_histogram_ndim:{input_array.ndim}")
        self.interpreter.set_tensor(self.input_details["index"], input_tensor)
        if scalar_array is not None:
            input_details = self.interpreter.get_input_details()
            if len(input_details) < 2:
                raise InferenceServiceError("tflite_scalar_input_not_supported_by_model")
            scalar_tensor = np.asarray(scalar_array, dtype=np.float32).reshape(1, -1)
            self.interpreter.set_tensor(input_details[1]["index"], scalar_tensor)
        self.interpreter.invoke()
        output = np.asarray(self.interpreter.get_tensor(self.output_details["index"]), dtype=float).reshape(-1)
        mapping = dict(zip(self.label_names, output.tolist()))
        return NoisePrediction(
            sigma=float(mapping.get("sigma", 0.0)),
            mu_q=float(mapping.get("mu_q", 0.0)),
            mu_p=float(mapping.get("mu_p", 0.0)),
            theta_deg=float(mapping.get("theta_deg", 0.0)),
            source="tflite_service",
            metadata={
                "model_path": str(self.model_path),
                "label_names": list(self.label_names),
                "raw_prediction": mapping,
            },
            )

    def explain(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        prediction = self.predict(histogram)
        metadata = dict(prediction.metadata or {})
        explanation = {"prediction": prediction.to_dict()}
        if self._stub_predictor is not None:
            explanation.update(self._stub_predictor.explain(histogram))
        explanation["metadata"] = metadata
        return explanation


class InferenceService(ABC):
    """Abstract client interface used by SlowLoopRuntime."""

    @abstractmethod
    def predict(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> NoisePrediction:
        raise NotImplementedError

    def close(self) -> None:
        return None


class InProcInferenceService(InferenceService):
    """In-process predictor used for local verification."""

    def __init__(self, predictor: BaseHistogramPredictor) -> None:
        self.predictor = predictor

    def predict(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> NoisePrediction:
        return self.predictor.predict(histogram)

    def explain(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        if hasattr(self.predictor, "explain"):
            return getattr(self.predictor, "explain")(histogram)
        return super().explain(histogram)


class SubprocessInferenceService(InferenceService):
    """JSON-line subprocess client for artifact/TFLite inference workers."""

    def __init__(self, service_config: InferenceServiceConfig) -> None:
        python_exe = service_config.python_executable or sys.executable
        python_exe = str(Path(python_exe).expanduser().resolve()) if python_exe not in {sys.executable, "python", "python3"} else python_exe
        model_arg = service_config.tflite_path if service_config.backend == "tflite" else service_config.model_path
        if model_arg is None:
            raise InferenceServiceError("subprocess_model_path_missing")

        command = [
            python_exe,
            "-m",
            service_config.worker_module,
            "--backend",
            service_config.backend,
            "--model-path",
            str(model_arg),
            "--label-names",
            ",".join(service_config.label_names),
        ]
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise InferenceServiceError("subprocess_pipe_init_failed")

    def predict(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> NoisePrediction:
        payload = self._request(histogram, explain=False)
        return NoisePrediction(
            sigma=float(payload.get("sigma", 0.0)),
            mu_q=float(payload.get("mu_q", 0.0)),
            mu_p=float(payload.get("mu_p", 0.0)),
            theta_deg=float(payload.get("theta_deg", 0.0)),
            source=str(payload.get("source", "subprocess_service")),
            metadata=dict(payload.get("metadata", {})),
        )

    def explain(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        return self._request(histogram, explain=True)

    def _request(self, histogram: np.ndarray | tuple[np.ndarray, np.ndarray], *, explain: bool) -> Dict[str, Any]:
        if self._proc.poll() is not None:
            stderr = "" if self._proc.stderr is None else self._proc.stderr.read()
            raise InferenceServiceError(f"inference_worker_exited:{self._proc.returncode}:{stderr.strip()}")

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        if isinstance(histogram, tuple):
            spatial, scalar = histogram
            payload_obj = {
                "histogram": np.asarray(spatial, dtype=float).tolist(),
                "scalar_features": np.asarray(scalar, dtype=float).tolist(),
                "explain": bool(explain),
            }
        else:
            payload_obj = {"histogram": np.asarray(histogram, dtype=float).tolist(), "explain": bool(explain)}
        self._proc.stdin.write(json.dumps(payload_obj, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        if not response_line:
            stderr = "" if self._proc.stderr is None else self._proc.stderr.read()
            raise InferenceServiceError(f"inference_worker_no_response:{stderr.strip()}")

        payload = json.loads(response_line)
        if "error" in payload:
            raise InferenceServiceError(str(payload["error"]))
        return payload

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()

    def __del__(self) -> None:  # pragma: no cover
        self.close()


def build_inference_service(
    config: Dict[str, Any],
    *,
    default_model_path: Optional[str],
    default_selector: str,
    default_use_latest_model_dir_artifact: bool,
) -> tuple[InferenceService, Optional[Path], InferenceServiceConfig]:
    """Instantiate configured inference service and return resolved model path."""
    service_cfg = InferenceServiceConfig.from_runtime_config(
        config,
        default_model_path=default_model_path,
        default_selector=default_selector,
        default_use_latest_model_dir_artifact=default_use_latest_model_dir_artifact,
    )

    resolved_path: Optional[Path] = None
    if service_cfg.backend == "artifact_npz":
        resolved_path = resolve_model_path(
            config,
            model_path=service_cfg.model_path,
            use_latest_model_dir_artifact=service_cfg.use_latest_model_dir_artifact,
            artifact_selector=service_cfg.artifact_selector,
        )
    elif service_cfg.backend == "tflite":
        resolved_path = resolve_tflite_path(
            config,
            tflite_path=service_cfg.tflite_path or service_cfg.model_path,
            use_latest_model_dir_artifact=service_cfg.use_latest_model_dir_artifact,
            artifact_selector=service_cfg.artifact_selector,
        )
    else:
        raise InferenceServiceError(f"unsupported_inference_backend:{service_cfg.backend}")

    if service_cfg.mode == "inproc":
        if service_cfg.backend == "artifact_npz":
            predictor = ArtifactHistogramPredictor(resolved_path)
        elif service_cfg.backend == "tflite":
            predictor = TFLiteHistogramPredictor(resolved_path, service_cfg.label_names)
        else:  # pragma: no cover
            raise InferenceServiceError(f"unsupported_inference_backend:{service_cfg.backend}")
        return InProcInferenceService(predictor), resolved_path, service_cfg
    if service_cfg.mode == "subprocess":
        worker_cfg = InferenceServiceConfig(
            mode=service_cfg.mode,
            backend=service_cfg.backend,
            model_path=None if resolved_path is None else str(resolved_path),
            use_latest_model_dir_artifact=False,
            artifact_selector=service_cfg.artifact_selector,
            python_executable=service_cfg.python_executable,
            worker_module=service_cfg.worker_module,
            startup_timeout_s=service_cfg.startup_timeout_s,
            request_timeout_s=service_cfg.request_timeout_s,
            tflite_path=None if resolved_path is None else str(resolved_path),
            label_names=service_cfg.label_names,
        )
        return SubprocessInferenceService(worker_cfg), resolved_path, worker_cfg
    raise InferenceServiceError(f"unsupported_inference_service_mode:{service_cfg.mode}")
