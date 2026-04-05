"""Slow-loop runtime: fair baselines and pluggable inference-service mode."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np

from cnn_fpga.decoder import (
    EKFBaseline,
    ParamMapper,
    ParticleFilterBaseline,
    ParticleFilterResidualBBaseline,
    RLSResidualBBaseline,
    UKFBaseline,
    WindowVarianceBaseline,
)
from cnn_fpga.decoder.param_mapper import NoisePrediction, analyze_decoder_aggressiveness
from cnn_fpga.runtime.feature_builder import RuntimeFeatureConfig, build_feature_tensor
from cnn_fpga.runtime.inference_service import InferenceService, InferenceServiceError, build_inference_service
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.scheduler import WindowFrame
from cnn_fpga.utils.config import get_path


class SlowLoopRuntimeError(RuntimeError):
    """Structured error for slow-loop failures."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SlowLoopRuntimeConfig:
    """Configuration for slow-loop runtime behavior."""

    mode: str = "fixed_baseline"
    fallback_sigma: float = 0.25
    fallback_mu_q: float = 0.0
    fallback_mu_p: float = 0.0
    fallback_theta_deg: float = 0.0
    oracle_delay_windows: int = 1
    sigma_prediction_std: float = 0.01
    mu_prediction_std: float = 0.005
    theta_prediction_std_deg: float = 0.75
    dma_timeout_prob: float = 1.0e-4
    axi_write_fail_prob: float = 5.0e-5
    cnn_infer_fail_prob: float = 2.0e-5
    model_artifact_path: Optional[str] = None
    use_latest_model_dir_artifact: bool = True
    artifact_selector: str = "latest_any"
    inference_service_mode: str = "inproc"
    inference_service_backend: str = "artifact_npz"
    fixed_sigma: float = 0.25
    fixed_mu_q: float = 0.0
    fixed_mu_p: float = 0.0
    fixed_theta_deg: float = 0.0
    teacher_mode: str = "window_variance"
    feature_config: RuntimeFeatureConfig = RuntimeFeatureConfig()
    constant_delta_mu_q: float = 0.0
    constant_delta_mu_p: float = 0.0
    residual_scale_mu: float = 1.0
    residual_clip_mu: float = 0.25
    residual_scale_b: float = 1.0
    residual_clip_b: float = 0.25
    ema_alpha: float = 1.0
    ema_apply_to: str = "mu"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SlowLoopRuntimeConfig":
        slow_cfg = config.get("slow_loop", {})
        predictor_cfg = slow_cfg.get("prediction_noise", {})
        err_cfg = config.get("error_model", {})
        artifact_cfg = slow_cfg.get("model_artifact", {})
        fixed_cfg = slow_cfg.get("fixed_baseline", {})
        service_cfg = slow_cfg.get("inference_service", {})
        hybrid_cfg = slow_cfg.get("hybrid_residual_mu", {})
        hybrid_b_cfg = slow_cfg.get("hybrid_residual_b", {})
        rls_b_cfg = slow_cfg.get("rls_residual_b", {})
        pf_b_cfg = slow_cfg.get("particle_filter_residual_b", {})
        constant_mu_cfg = slow_cfg.get("constant_residual_mu", {})
        post_cfg = config.get("postprocess", slow_cfg.get("postprocess", {}))
        model_cfg = config.get("model", {})
        mode = str(slow_cfg.get("mode", "fixed_baseline")).lower()
        feature_source_cfg = slow_cfg
        if mode == "hybrid_residual_mu":
            feature_source_cfg = hybrid_cfg
        elif mode == "hybrid_residual_b":
            feature_source_cfg = hybrid_b_cfg
        elif mode == "rls_residual_b":
            feature_source_cfg = rls_b_cfg
        elif mode == "particle_filter_residual_b":
            feature_source_cfg = pf_b_cfg

        constant_delta_mu_q = float(constant_mu_cfg.get("delta_mu_q", 0.0))
        constant_delta_mu_p = float(constant_mu_cfg.get("delta_mu_p", 0.0))
        if bool(constant_mu_cfg.get("use_dataset_train_mean", False)):
            dataset_dir = get_path(config, "dataset_dir", "artifacts/datasets/default")
            train_path = dataset_dir / "train.npz"
            if not train_path.exists():
                raise FileNotFoundError(f"constant_residual_mu dataset split not found: {train_path}")
            data = np.load(train_path, allow_pickle=True)
            label_names = [str(item) for item in data["label_names"]]
            if label_names != ["mu_q", "mu_p"]:
                raise ValueError(f"constant_residual_mu expects residual_mu labels, got {label_names}")
            label_mean = data["labels"].astype(np.float64).mean(axis=0)
            constant_delta_mu_q = float(label_mean[0])
            constant_delta_mu_p = float(label_mean[1])

        return cls(
            mode=mode,
            fallback_sigma=float(slow_cfg.get("fallback_sigma", 0.25)),
            fallback_mu_q=float(slow_cfg.get("fallback_mu_q", 0.0)),
            fallback_mu_p=float(slow_cfg.get("fallback_mu_p", 0.0)),
            fallback_theta_deg=float(slow_cfg.get("fallback_theta_deg", 0.0)),
            oracle_delay_windows=int(slow_cfg.get("oracle_delay_windows", 1)),
            sigma_prediction_std=float(predictor_cfg.get("sigma_std", 0.01)),
            mu_prediction_std=float(predictor_cfg.get("mu_std", 0.005)),
            theta_prediction_std_deg=float(predictor_cfg.get("theta_deg_std", 0.75)),
            dma_timeout_prob=float(err_cfg.get("dma_timeout_prob", 1.0e-4)),
            axi_write_fail_prob=float(err_cfg.get("axi_write_fail_prob", 5.0e-5)),
            cnn_infer_fail_prob=float(err_cfg.get("cnn_infer_fail_prob", 2.0e-5)),
            model_artifact_path=artifact_cfg.get("path"),
            use_latest_model_dir_artifact=bool(artifact_cfg.get("use_latest_model_dir_artifact", True)),
            artifact_selector=str(artifact_cfg.get("artifact_selector", "latest_any")).lower(),
            inference_service_mode=str(service_cfg.get("mode", "inproc")).lower(),
            inference_service_backend=str(service_cfg.get("backend", "artifact_npz")).lower(),
            fixed_sigma=float(fixed_cfg.get("sigma", model_cfg.get("sigma_measurement", 0.25))),
            fixed_mu_q=float(fixed_cfg.get("mu_q", model_cfg.get("mu_q", 0.0))),
            fixed_mu_p=float(fixed_cfg.get("mu_p", model_cfg.get("mu_p", 0.0))),
            fixed_theta_deg=float(fixed_cfg.get("theta_deg", model_cfg.get("theta_deg", 0.0))),
            teacher_mode=str(
                feature_source_cfg.get(
                    "teacher_mode",
                    constant_mu_cfg.get(
                        "teacher_mode",
                        pf_b_cfg.get(
                            "teacher_mode",
                            rls_b_cfg.get("teacher_mode", hybrid_cfg.get("teacher_mode", "window_variance")),
                        ),
                    ),
                )
            ).lower(),
            feature_config=RuntimeFeatureConfig(
                context_windows=int(feature_source_cfg.get("context_windows", slow_cfg.get("context_windows", 1))),
                context_padding=str(feature_source_cfg.get("context_padding", slow_cfg.get("context_padding", "edge_repeat"))).lower(),
                include_histogram_deltas=bool(feature_source_cfg.get("include_histogram_deltas", slow_cfg.get("include_histogram_deltas", False))),
                include_teacher_prediction=bool(feature_source_cfg.get("include_teacher_prediction", slow_cfg.get("include_teacher_prediction", False))),
                include_teacher_params=bool(feature_source_cfg.get("include_teacher_params", slow_cfg.get("include_teacher_params", False))),
                include_teacher_deltas=bool(feature_source_cfg.get("include_teacher_deltas", slow_cfg.get("include_teacher_deltas", False))),
            ),
            constant_delta_mu_q=constant_delta_mu_q,
            constant_delta_mu_p=constant_delta_mu_p,
            residual_scale_mu=float(hybrid_cfg.get("residual_scale_mu", 1.0)),
            residual_clip_mu=float(hybrid_cfg.get("residual_clip_mu", 0.25)),
            residual_scale_b=float(hybrid_b_cfg.get("residual_scale_b", 1.0)),
            residual_clip_b=float(hybrid_b_cfg.get("residual_clip_b", 0.25)),
            ema_alpha=float(post_cfg.get("ema_alpha", rls_b_cfg.get("ema_alpha", hybrid_cfg.get("ema_alpha", 1.0)))),
            ema_apply_to=str(
                post_cfg.get(
                    "ema_apply_to",
                    rls_b_cfg.get(
                        "ema_apply_to",
                        hybrid_b_cfg.get("ema_apply_to", hybrid_cfg.get("ema_apply_to", constant_mu_cfg.get("ema_apply_to", "mu"))),
                    ),
                )
            ).lower(),
        )


class SlowLoopRuntime:
    """Slow-loop implementation: prediction -> parameter mapping -> staged update."""

    def __init__(
        self,
        config: SlowLoopRuntimeConfig,
        *,
        param_mapper: ParamMapper,
        window_variance_baseline: Optional[WindowVarianceBaseline] = None,
        ekf_baseline: Optional[EKFBaseline] = None,
        particle_filter_baseline: Optional[ParticleFilterBaseline] = None,
        particle_filter_residual_b_baseline: Optional[ParticleFilterResidualBBaseline] = None,
        ukf_baseline: Optional[UKFBaseline] = None,
        rls_residual_b_baseline: Optional[RLSResidualBBaseline] = None,
        inference_service: Optional[InferenceService] = None,
        inference_model_path: Optional[Path] = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.param_mapper = param_mapper
        self.window_variance_baseline = window_variance_baseline
        self.ekf_baseline = ekf_baseline
        self.particle_filter_baseline = particle_filter_baseline
        self.particle_filter_residual_b_baseline = particle_filter_residual_b_baseline
        self.ukf_baseline = ukf_baseline
        self.rls_residual_b_baseline = rls_residual_b_baseline
        self.inference_service = inference_service
        self.inference_model_path = inference_model_path
        self._rng = np.random.default_rng(seed)
        self._oracle_queue: Deque[NoisePrediction] = deque()
        max_history = max(1, self.config.feature_config.context_windows)
        self._histogram_context: Deque[np.ndarray] = deque(maxlen=max_history)
        self._teacher_prediction_context: Deque[np.ndarray] = deque(maxlen=max_history)
        self._teacher_param_context: Deque[np.ndarray] = deque(maxlen=max_history)
        self._ema_prediction: Optional[NoisePrediction] = None
        self._ema_b: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any], seed: int | None = None) -> "SlowLoopRuntime":
        runtime_cfg = SlowLoopRuntimeConfig.from_config(config)
        inference_service = None
        inference_model_path = None
        if runtime_cfg.mode in {"model_artifact", "hybrid_residual_mu", "hybrid_residual_b"}:
            try:
                inference_service, inference_model_path, _ = build_inference_service(
                    config,
                    default_model_path=runtime_cfg.model_artifact_path,
                    default_selector=runtime_cfg.artifact_selector,
                    default_use_latest_model_dir_artifact=runtime_cfg.use_latest_model_dir_artifact,
                )
            except InferenceServiceError as exc:
                raise SlowLoopRuntimeError(exc.reason) from exc
        return cls(
            runtime_cfg,
            param_mapper=ParamMapper.from_config(config),
            window_variance_baseline=WindowVarianceBaseline.from_config(config),
            ekf_baseline=EKFBaseline.from_config(config),
            particle_filter_baseline=ParticleFilterBaseline.from_config(config, seed=seed),
            particle_filter_residual_b_baseline=ParticleFilterResidualBBaseline.from_config(config, seed=seed),
            ukf_baseline=UKFBaseline.from_config(config),
            rls_residual_b_baseline=RLSResidualBBaseline.from_config(config),
            inference_service=inference_service,
            inference_model_path=inference_model_path,
            seed=seed,
        )

    def _extract_target_prediction(self, window: WindowFrame) -> NoisePrediction:
        target = window.payload.get("target_params", {})
        return NoisePrediction(
            sigma=float(target.get("sigma", self.config.fallback_sigma)),
            mu_q=float(target.get("mu_q", self.config.fallback_mu_q)),
            mu_p=float(target.get("mu_p", self.config.fallback_mu_p)),
            theta_deg=float(target.get("theta_deg", self.config.fallback_theta_deg)),
            source="window_target",
            metadata={"window_id": window.window_id},
        )

    def _apply_prediction_noise(self, target: NoisePrediction) -> NoisePrediction:
        return NoisePrediction(
            sigma=max(0.0, target.sigma + float(self._rng.normal(0.0, self.config.sigma_prediction_std))),
            mu_q=target.mu_q + float(self._rng.normal(0.0, self.config.mu_prediction_std)),
            mu_p=target.mu_p + float(self._rng.normal(0.0, self.config.mu_prediction_std)),
            theta_deg=target.theta_deg + float(self._rng.normal(0.0, self.config.theta_prediction_std_deg)),
            source="mock_cnn",
            metadata={"target_prediction": target.to_dict()},
        )

    def _fixed_baseline_prediction(self) -> NoisePrediction:
        return NoisePrediction(
            sigma=float(self.config.fixed_sigma),
            mu_q=float(self.config.fixed_mu_q),
            mu_p=float(self.config.fixed_mu_p),
            theta_deg=float(self.config.fixed_theta_deg),
            source="fixed_baseline",
            metadata={
                "fixed_sigma": float(self.config.fixed_sigma),
                "fixed_mu_q": float(self.config.fixed_mu_q),
                "fixed_mu_p": float(self.config.fixed_mu_p),
                "fixed_theta_deg": float(self.config.fixed_theta_deg),
            },
        )

    def _static_linear_prediction(self, *, source: str = "static_linear") -> NoisePrediction:
        prediction = self._fixed_baseline_prediction()
        return NoisePrediction(
            sigma=prediction.sigma,
            mu_q=prediction.mu_q,
            mu_p=prediction.mu_p,
            theta_deg=prediction.theta_deg,
            source=source,
            metadata=dict(prediction.metadata or {}),
        )

    def _predict_from_delayed_oracle(self, window: WindowFrame) -> NoisePrediction:
        current_target = self._extract_target_prediction(window)
        self._oracle_queue.append(current_target)
        if len(self._oracle_queue) <= self.config.oracle_delay_windows:
            delayed_target = self._fixed_baseline_prediction()
        else:
            delayed_target = self._oracle_queue.popleft()
        prediction = self._apply_prediction_noise(delayed_target)
        metadata = dict(prediction.metadata or {})
        metadata.update(
            {
                "oracle_delay_windows": self.config.oracle_delay_windows,
                "current_target_prediction": current_target.to_dict(),
            }
        )
        return NoisePrediction(
            sigma=prediction.sigma,
            mu_q=prediction.mu_q,
            mu_p=prediction.mu_p,
            theta_deg=prediction.theta_deg,
            source="oracle_delayed",
            metadata=metadata,
        )

    def _extract_histogram(self, window: WindowFrame) -> np.ndarray:
        if "histogram" not in window.payload:
            raise SlowLoopRuntimeError("window_histogram_missing")
        histogram = np.asarray(window.payload["histogram"], dtype=np.float64)
        if histogram.ndim != 2:
            raise SlowLoopRuntimeError("window_histogram_invalid_shape")
        return histogram

    def _prediction_vector(self, prediction: NoisePrediction) -> np.ndarray:
        return np.array(
            [prediction.sigma, prediction.mu_q, prediction.mu_p, prediction.theta_deg],
            dtype=np.float32,
        )

    def _build_artifact_input(
        self,
        histogram: np.ndarray,
        *,
        teacher_prediction: NoisePrediction | None = None,
        teacher_params: DecoderRuntimeParams | None = None,
    ) -> np.ndarray:
        hist = np.asarray(histogram, dtype=np.float32)
        self._histogram_context.append(hist)
        if teacher_prediction is not None:
            self._teacher_prediction_context.append(self._prediction_vector(teacher_prediction))
        if teacher_params is not None:
            self._teacher_param_context.append(np.asarray(teacher_params.b, dtype=np.float32))

        feature_cfg = self.config.feature_config
        teacher_prediction_history = list(self._teacher_prediction_context) if (
            feature_cfg.include_teacher_prediction or feature_cfg.include_teacher_deltas
        ) else None
        teacher_param_history = list(self._teacher_param_context) if (
            feature_cfg.include_teacher_params or feature_cfg.include_teacher_deltas
        ) else None
        return build_feature_tensor(
            list(self._histogram_context),
            feature_cfg,
            teacher_prediction_history=teacher_prediction_history,
            teacher_param_history=teacher_param_history,
        )

    def _predict_from_artifact(
        self,
        window: WindowFrame,
        *,
        teacher_prediction: NoisePrediction | None = None,
        teacher_params: DecoderRuntimeParams | None = None,
    ) -> NoisePrediction:
        if self.inference_service is None:
            raise SlowLoopRuntimeError("inference_service_not_initialized")
        histogram = self._extract_histogram(window)
        artifact_input = self._build_artifact_input(
            histogram,
            teacher_prediction=teacher_prediction,
            teacher_params=teacher_params,
        )
        try:
            prediction = self.inference_service.predict(artifact_input)
        except InferenceServiceError as exc:
            raise SlowLoopRuntimeError(exc.reason) from exc
        metadata = dict(prediction.metadata or {})
        metadata.update(
            {
                "window_id": window.window_id,
                "window_diagnostics": dict(window.payload.get("diagnostics", {})),
                "inference_service_mode": self.config.inference_service_mode,
                "inference_service_backend": self.config.inference_service_backend,
                "context_windows": self.config.feature_config.context_windows,
            }
        )
        return NoisePrediction(
            sigma=prediction.sigma,
            mu_q=prediction.mu_q,
            mu_p=prediction.mu_p,
            theta_deg=prediction.theta_deg,
            source=prediction.source,
            metadata=metadata,
        )

    def _predict_from_window_variance(self, window: WindowFrame) -> NoisePrediction:
        if self.window_variance_baseline is None:
            raise SlowLoopRuntimeError("window_variance_not_initialized")
        histogram = self._extract_histogram(window)
        prediction = self.window_variance_baseline.predict(histogram, window_id=window.window_id)
        metadata = dict(prediction.metadata or {})
        metadata.update(
            {
                "window_id": window.window_id,
                "window_diagnostics": dict(window.payload.get("diagnostics", {})),
            }
        )
        return NoisePrediction(
            sigma=prediction.sigma,
            mu_q=prediction.mu_q,
            mu_p=prediction.mu_p,
            theta_deg=prediction.theta_deg,
            source="window_variance",
            metadata=metadata,
        )

    def _predict_from_ekf(self, window: WindowFrame) -> NoisePrediction:
        if self.ekf_baseline is None:
            raise SlowLoopRuntimeError("ekf_baseline_not_initialized")
        histogram = self._extract_histogram(window)
        prediction = self.ekf_baseline.predict(histogram, window_id=window.window_id)
        metadata = dict(prediction.metadata or {})
        metadata.update(
            {
                "window_id": window.window_id,
                "window_diagnostics": dict(window.payload.get("diagnostics", {})),
            }
        )
        return NoisePrediction(
            sigma=prediction.sigma,
            mu_q=prediction.mu_q,
            mu_p=prediction.mu_p,
            theta_deg=prediction.theta_deg,
            source="ekf_baseline",
            metadata=metadata,
        )

    def _predict_from_ukf(self, window: WindowFrame) -> NoisePrediction:
        if self.ukf_baseline is None:
            raise SlowLoopRuntimeError("ukf_baseline_not_initialized")
        histogram = self._extract_histogram(window)
        prediction = self.ukf_baseline.predict(histogram, window_id=window.window_id)
        metadata = dict(prediction.metadata or {})
        metadata.update(
            {
                "window_id": window.window_id,
                "window_diagnostics": dict(window.payload.get("diagnostics", {})),
            }
        )
        return NoisePrediction(
            sigma=prediction.sigma,
            mu_q=prediction.mu_q,
            mu_p=prediction.mu_p,
            theta_deg=prediction.theta_deg,
            source="ukf_baseline",
            metadata=metadata,
        )

    def _predict_from_particle_filter(self, window: WindowFrame) -> NoisePrediction:
        if self.particle_filter_baseline is None:
            raise SlowLoopRuntimeError("particle_filter_baseline_not_initialized")
        histogram = self._extract_histogram(window)
        prediction = self.particle_filter_baseline.predict(histogram, window_id=window.window_id)
        metadata = dict(prediction.metadata or {})
        metadata.update(
            {
                "window_id": window.window_id,
                "window_diagnostics": dict(window.payload.get("diagnostics", {})),
            }
        )
        return NoisePrediction(
            sigma=prediction.sigma,
            mu_q=prediction.mu_q,
            mu_p=prediction.mu_p,
            theta_deg=prediction.theta_deg,
            source="particle_filter_baseline",
            metadata=metadata,
        )

    def _predict_teacher(self, window: WindowFrame) -> NoisePrediction:
        if self.config.teacher_mode == "window_variance":
            return self._predict_from_window_variance(window)
        if self.config.teacher_mode == "ekf":
            return self._predict_from_ekf(window)
        if self.config.teacher_mode == "ukf":
            return self._predict_from_ukf(window)
        if self.config.teacher_mode == "particle_filter":
            return self._predict_from_particle_filter(window)
        raise SlowLoopRuntimeError(f"unsupported_teacher_mode:{self.config.teacher_mode}")

    def _predict_constant_residual_mu(self, window: WindowFrame, active_params: DecoderRuntimeParams) -> NoisePrediction:
        teacher_prediction = self._predict_teacher(window)
        teacher_params = self.param_mapper.map_prediction(teacher_prediction, previous_params=active_params)
        delta_mu_q = float(self.config.constant_delta_mu_q)
        delta_mu_p = float(self.config.constant_delta_mu_p)
        metadata = {
            "teacher_prediction": teacher_prediction.to_dict(),
            "teacher_params": teacher_params.to_dict(),
            "teacher_mode": self.config.teacher_mode,
            "constant_residual_mu_q": delta_mu_q,
            "constant_residual_mu_p": delta_mu_p,
            "applied_delta_mu_q": delta_mu_q,
            "applied_delta_mu_p": delta_mu_p,
        }
        return NoisePrediction(
            sigma=teacher_prediction.sigma,
            mu_q=teacher_prediction.mu_q + delta_mu_q,
            mu_p=teacher_prediction.mu_p + delta_mu_p,
            theta_deg=teacher_prediction.theta_deg,
            source=f"{self.config.teacher_mode}+constant_residual_mu",
            metadata=metadata,
        )

    def _predict_hybrid_residual_mu(self, window: WindowFrame, active_params: DecoderRuntimeParams) -> NoisePrediction:
        teacher_prediction = self._predict_teacher(window)
        teacher_params = self.param_mapper.map_prediction(teacher_prediction, previous_params=active_params)
        residual_prediction = self._predict_from_artifact(
            window,
            teacher_prediction=teacher_prediction,
            teacher_params=teacher_params,
        )
        delta_mu_q = float(
            np.clip(
                residual_prediction.mu_q * self.config.residual_scale_mu,
                -self.config.residual_clip_mu,
                self.config.residual_clip_mu,
            )
        )
        delta_mu_p = float(
            np.clip(
                residual_prediction.mu_p * self.config.residual_scale_mu,
                -self.config.residual_clip_mu,
                self.config.residual_clip_mu,
            )
        )
        metadata = {
            "teacher_prediction": teacher_prediction.to_dict(),
            "teacher_params": teacher_params.to_dict(),
            "residual_prediction": residual_prediction.to_dict(),
            "teacher_mode": self.config.teacher_mode,
            "context_windows": self.config.feature_config.context_windows,
            "residual_scale_mu": self.config.residual_scale_mu,
            "residual_clip_mu": self.config.residual_clip_mu,
            "applied_delta_mu_q": delta_mu_q,
            "applied_delta_mu_p": delta_mu_p,
        }
        return NoisePrediction(
            sigma=teacher_prediction.sigma,
            mu_q=teacher_prediction.mu_q + delta_mu_q,
            mu_p=teacher_prediction.mu_p + delta_mu_p,
            theta_deg=teacher_prediction.theta_deg,
            source=f"{self.config.teacher_mode}+residual_mu",
            metadata=metadata,
        )

    def _apply_postprocess(self, prediction: NoisePrediction) -> NoisePrediction:
        alpha = float(np.clip(self.config.ema_alpha, 0.0, 1.0))
        if alpha >= 1.0 or self._ema_prediction is None:
            self._ema_prediction = prediction
            return prediction

        prev = self._ema_prediction
        if self.config.ema_apply_to in {"mu", "mu_only"}:
            smoothed = NoisePrediction(
                sigma=prediction.sigma,
                mu_q=(1.0 - alpha) * prev.mu_q + alpha * prediction.mu_q,
                mu_p=(1.0 - alpha) * prev.mu_p + alpha * prediction.mu_p,
                theta_deg=prediction.theta_deg,
                source=prediction.source,
                metadata={
                    **dict(prediction.metadata or {}),
                    "postprocess": {
                        "type": "ema",
                        "alpha": alpha,
                        "apply_to": self.config.ema_apply_to,
                    },
                },
            )
        else:
            smoothed = NoisePrediction(
                sigma=(1.0 - alpha) * prev.sigma + alpha * prediction.sigma,
                mu_q=(1.0 - alpha) * prev.mu_q + alpha * prediction.mu_q,
                mu_p=(1.0 - alpha) * prev.mu_p + alpha * prediction.mu_p,
                theta_deg=(1.0 - alpha) * prev.theta_deg + alpha * prediction.theta_deg,
                source=prediction.source,
                metadata={
                    **dict(prediction.metadata or {}),
                    "postprocess": {
                        "type": "ema",
                        "alpha": alpha,
                        "apply_to": self.config.ema_apply_to,
                    },
                },
            )
        self._ema_prediction = smoothed
        return smoothed

    def _apply_b_postprocess(self, b_target: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.config.ema_alpha, 0.0, 1.0))
        b_target = np.asarray(b_target, dtype=float).reshape(2)
        if alpha >= 1.0 or self._ema_b is None or self.config.ema_apply_to not in {"b", "all"}:
            self._ema_b = b_target.copy()
            return b_target
        smoothed = (1.0 - alpha) * self._ema_b + alpha * b_target
        self._ema_b = smoothed.copy()
        return smoothed

    def _predict_hybrid_residual_b(self, window: WindowFrame, active_params: DecoderRuntimeParams) -> DecoderRuntimeParams:
        teacher_prediction = self._predict_teacher(window)
        teacher_params = self.param_mapper.map_prediction(teacher_prediction, previous_params=active_params)
        residual_prediction = self._predict_from_artifact(
            window,
            teacher_prediction=teacher_prediction,
            teacher_params=teacher_params,
        )
        raw_prediction = dict((residual_prediction.metadata or {}).get("raw_prediction", {}))
        raw_b_q = float(raw_prediction.get("b_q", raw_prediction.get("mu_q", 0.0)))
        raw_b_p = float(raw_prediction.get("b_p", raw_prediction.get("mu_p", 0.0)))
        delta_b = np.array(
            [
                np.clip(raw_b_q * self.config.residual_scale_b, -self.config.residual_clip_b, self.config.residual_clip_b),
                np.clip(raw_b_p * self.config.residual_scale_b, -self.config.residual_clip_b, self.config.residual_clip_b),
            ],
            dtype=float,
        )
        b_next = self._apply_b_postprocess(teacher_params.b + delta_b)
        aggressiveness = analyze_decoder_aggressiveness(
            teacher_params.K,
            b_next,
            gain_upper_bound=self.param_mapper.config.gain_clip[1],
            correction_limit=self.param_mapper.config.correction_limit,
            aggressive_gain_ratio=self.param_mapper.config.aggressive_gain_ratio,
            aggressive_bias_ratio=self.param_mapper.config.aggressive_bias_ratio,
        )
        metadata = {
            "prediction": teacher_prediction.to_dict(),
            "teacher_prediction": teacher_prediction.to_dict(),
            "teacher_params": teacher_params.to_dict(),
            "residual_prediction": residual_prediction.to_dict(),
            "runtime_mode": self.config.mode,
            "teacher_mode": self.config.teacher_mode,
            "context_windows": self.config.feature_config.context_windows,
            "residual_scale_b": self.config.residual_scale_b,
            "residual_clip_b": self.config.residual_clip_b,
            "applied_delta_b": delta_b.tolist(),
            "param_aggressiveness": aggressiveness,
            "feature_config": {
                "context_windows": self.config.feature_config.context_windows,
                "include_histogram_deltas": self.config.feature_config.include_histogram_deltas,
                "include_teacher_prediction": self.config.feature_config.include_teacher_prediction,
                "include_teacher_params": self.config.feature_config.include_teacher_params,
                "include_teacher_deltas": self.config.feature_config.include_teacher_deltas,
            },
        }
        if self.config.ema_apply_to in {"b", "all"}:
            metadata["postprocess"] = {
                "type": "ema",
                "alpha": float(np.clip(self.config.ema_alpha, 0.0, 1.0)),
                "apply_to": self.config.ema_apply_to,
            }
        return DecoderRuntimeParams(K=teacher_params.K, b=b_next, metadata=metadata)

    def _predict_rls_residual_b(self, window: WindowFrame, active_params: DecoderRuntimeParams) -> DecoderRuntimeParams:
        if self.rls_residual_b_baseline is None:
            raise SlowLoopRuntimeError("rls_residual_b_not_initialized")
        teacher_prediction = self._predict_teacher(window)
        teacher_params = self.param_mapper.map_prediction(teacher_prediction, previous_params=active_params)
        measurement_prediction = self._predict_from_window_variance(window)
        measurement_params = self.param_mapper.map_prediction(measurement_prediction, previous_params=active_params)
        residual = self.rls_residual_b_baseline.predict(
            teacher_prediction=teacher_prediction,
            measurement_prediction=measurement_prediction,
            teacher_b=teacher_params.b,
            measurement_b=measurement_params.b,
            window_id=window.window_id,
        )
        b_next = self._apply_b_postprocess(teacher_params.b + residual.delta_b_pred)
        aggressiveness = analyze_decoder_aggressiveness(
            teacher_params.K,
            b_next,
            gain_upper_bound=self.param_mapper.config.gain_clip[1],
            correction_limit=self.param_mapper.config.correction_limit,
            aggressive_gain_ratio=self.param_mapper.config.aggressive_gain_ratio,
            aggressive_bias_ratio=self.param_mapper.config.aggressive_bias_ratio,
        )
        metadata = {
            "prediction": teacher_prediction.to_dict(),
            "teacher_prediction": teacher_prediction.to_dict(),
            "teacher_params": teacher_params.to_dict(),
            "measurement_prediction": measurement_prediction.to_dict(),
            "measurement_params": measurement_params.to_dict(),
            "runtime_mode": self.config.mode,
            "teacher_mode": self.config.teacher_mode,
            "rls_residual": residual.metadata,
            "param_aggressiveness": aggressiveness,
        }
        if self.config.ema_apply_to in {"b", "all"}:
            metadata["postprocess"] = {
                "type": "ema",
                "alpha": float(np.clip(self.config.ema_alpha, 0.0, 1.0)),
                "apply_to": self.config.ema_apply_to,
            }
        return DecoderRuntimeParams(K=teacher_params.K, b=b_next, metadata=metadata)

    def _predict_particle_filter_residual_b(self, window: WindowFrame, active_params: DecoderRuntimeParams) -> DecoderRuntimeParams:
        if self.particle_filter_residual_b_baseline is None:
            raise SlowLoopRuntimeError("particle_filter_residual_b_not_initialized")
        teacher_prediction = self._predict_teacher(window)
        teacher_params = self.param_mapper.map_prediction(teacher_prediction, previous_params=active_params)
        measurement_prediction = self._predict_from_window_variance(window)
        measurement_params = self.param_mapper.map_prediction(measurement_prediction, previous_params=active_params)
        residual = self.particle_filter_residual_b_baseline.predict(
            teacher_b=teacher_params.b,
            measurement_b=measurement_params.b,
            window_id=window.window_id,
        )
        b_next = self._apply_b_postprocess(teacher_params.b + residual.delta_b_pred)
        aggressiveness = analyze_decoder_aggressiveness(
            teacher_params.K,
            b_next,
            gain_upper_bound=self.param_mapper.config.gain_clip[1],
            correction_limit=self.param_mapper.config.correction_limit,
            aggressive_gain_ratio=self.param_mapper.config.aggressive_gain_ratio,
            aggressive_bias_ratio=self.param_mapper.config.aggressive_bias_ratio,
        )
        metadata = {
            "prediction": teacher_prediction.to_dict(),
            "teacher_prediction": teacher_prediction.to_dict(),
            "teacher_params": teacher_params.to_dict(),
            "measurement_prediction": measurement_prediction.to_dict(),
            "measurement_params": measurement_params.to_dict(),
            "runtime_mode": self.config.mode,
            "teacher_mode": self.config.teacher_mode,
            "particle_filter_residual": residual.metadata,
            "param_aggressiveness": aggressiveness,
        }
        if self.config.ema_apply_to in {"b", "all"}:
            metadata["postprocess"] = {
                "type": "ema",
                "alpha": float(np.clip(self.config.ema_alpha, 0.0, 1.0)),
                "apply_to": self.config.ema_apply_to,
            }
        return DecoderRuntimeParams(K=teacher_params.K, b=b_next, metadata=metadata)

    def _predict(self, window: WindowFrame, active_params: DecoderRuntimeParams) -> NoisePrediction:
        if self.config.mode in {"mock", "oracle_delayed"}:
            return self._predict_from_delayed_oracle(window)
        if self.config.mode == "fixed_baseline":
            return self._fixed_baseline_prediction()
        if self.config.mode == "static_linear":
            return self._static_linear_prediction(source="static_linear")
        if self.config.mode == "window_variance":
            return self._predict_from_window_variance(window)
        if self.config.mode == "ekf":
            return self._predict_from_ekf(window)
        if self.config.mode == "particle_filter":
            return self._apply_postprocess(self._predict_from_particle_filter(window))
        if self.config.mode == "ukf":
            return self._predict_from_ukf(window)
        if self.config.mode == "model_artifact":
            return self._apply_postprocess(self._predict_from_artifact(window))
        if self.config.mode == "constant_residual_mu":
            return self._apply_postprocess(self._predict_constant_residual_mu(window, active_params))
        if self.config.mode == "hybrid_residual_mu":
            return self._apply_postprocess(self._predict_hybrid_residual_mu(window, active_params))
        raise SlowLoopRuntimeError(f"unsupported_slow_loop_mode:{self.config.mode}")

    def _maybe_fail(self) -> None:
        if self._rng.random() < self.config.dma_timeout_prob:
            raise SlowLoopRuntimeError("dma_timeout")
        if self.config.mode in {"model_artifact", "hybrid_residual_mu", "hybrid_residual_b"} and self._rng.random() < self.config.cnn_infer_fail_prob:
            raise SlowLoopRuntimeError("cnn_infer_fail")
        if self._rng.random() < self.config.axi_write_fail_prob:
            raise SlowLoopRuntimeError("axi_write_fail")

    def __call__(self, window: WindowFrame, active_params: DecoderRuntimeParams) -> DecoderRuntimeParams:
        self._maybe_fail()
        if self.config.mode == "hybrid_residual_b":
            proposed = self._predict_hybrid_residual_b(window, active_params)
            metadata = dict(proposed.metadata)
            metadata.update(
                {
                    "window_id": window.window_id,
                    "window_diagnostics": dict(window.payload.get("diagnostics", {})),
                }
            )
            return DecoderRuntimeParams(K=proposed.K, b=proposed.b, metadata=metadata)
        if self.config.mode == "rls_residual_b":
            proposed = self._predict_rls_residual_b(window, active_params)
            metadata = dict(proposed.metadata)
            metadata.update(
                {
                    "window_id": window.window_id,
                    "window_diagnostics": dict(window.payload.get("diagnostics", {})),
                }
            )
            return DecoderRuntimeParams(K=proposed.K, b=proposed.b, metadata=metadata)
        if self.config.mode == "particle_filter_residual_b":
            proposed = self._predict_particle_filter_residual_b(window, active_params)
            metadata = dict(proposed.metadata)
            metadata.update(
                {
                    "window_id": window.window_id,
                    "window_diagnostics": dict(window.payload.get("diagnostics", {})),
                }
            )
            return DecoderRuntimeParams(K=proposed.K, b=proposed.b, metadata=metadata)

        prediction = self._predict(window, active_params)
        mapped = self.param_mapper.map_prediction(prediction, previous_params=active_params)
        metadata = dict(mapped.metadata)
        metadata.update(
            {
                "runtime_mode": self.config.mode,
                "window_id": window.window_id,
                "window_diagnostics": dict(window.payload.get("diagnostics", {})),
            }
        )
        if self.config.mode in {"mock", "oracle_delayed"}:
            metadata["target_prediction"] = self._extract_target_prediction(window).to_dict()
        if self.config.mode == "fixed_baseline":
            metadata["fixed_prediction"] = self._fixed_baseline_prediction().to_dict()
        if self.config.mode == "static_linear":
            metadata["fixed_prediction"] = self._static_linear_prediction(source="static_linear").to_dict()
        if self.config.mode in {"hybrid_residual_mu", "constant_residual_mu"}:
            metadata["teacher_mode"] = self.config.teacher_mode
            metadata["context_windows"] = self.config.feature_config.context_windows
        return DecoderRuntimeParams(K=mapped.K, b=mapped.b, metadata=metadata)

    def close(self) -> None:
        if self.inference_service is not None:
            self.inference_service.close()

    @property
    def artifact_predictor(self) -> Optional[object]:
        # 中文注释：保持旧代码中 `artifact_predictor.artifact_path` 的兼容读取。
        if self.inference_model_path is None:
            return None

        class _CompatArtifactHandle:
            def __init__(self, artifact_path: Path) -> None:
                self.artifact_path = artifact_path

        return _CompatArtifactHandle(self.inference_model_path)
