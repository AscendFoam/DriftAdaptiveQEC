"""Bridge physically-motivated noise channels to runtime-effective noise parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _clip(value: float, minimum: Optional[float], maximum: Optional[float]) -> float:
    if minimum is not None:
        value = max(float(minimum), float(value))
    if maximum is not None:
        value = min(float(maximum), float(value))
    return float(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _evolve_scalar(
    kind: str,
    *,
    epoch_id: int,
    current: float,
    initial: float,
    alpha: float,
    amplitude: float,
    frequency: float,
    step_std: float,
    jump_round: int,
    delta: float,
    minimum: Optional[float],
    maximum: Optional[float],
    rng: np.random.Generator,
) -> float:
    t = float(epoch_id - 1)
    kind = str(kind).lower()
    if kind in {"static", "constant"}:
        value = initial
    elif kind == "linear":
        value = initial + alpha * t
    elif kind == "step":
        value = initial + (delta if (jump_round > 0 and epoch_id >= jump_round) else 0.0)
    elif kind in {"sin", "sine", "periodic"}:
        value = initial + amplitude * np.sin(2.0 * np.pi * frequency * t)
    else:
        value = current + rng.normal(0.0, step_std)
    return _clip(value, minimum, maximum)


@dataclass(frozen=True)
class PhysicalNoiseBridgeConfig:
    """Configuration for a lightweight physical-to-effective noise bridge."""

    enabled: bool = False
    mapping_mode: str = "hybrid_additive"
    use_base_signal: bool = True
    sigma_combine_mode: str = "quadrature"
    sigma_scale: float = 1.0
    mu_q_scale: float = 1.0
    mu_p_scale: float = 1.0
    phase_to_theta_gain_deg: float = 180.0 / np.pi
    theta_bias_deg: float = 0.0
    theta_clip_min_deg: float = -30.0
    theta_clip_max_deg: float = 30.0
    sigma_reference: float = 0.12
    mu_reference: float = 0.02
    theta_reference_deg: float = 6.0
    gamma0: float = 0.0
    gamma_min: float = 0.0
    gamma_max: float = 0.25
    gamma_alpha: float = 0.0
    gamma_amplitude: float = 0.0
    gamma_frequency: float = 0.0
    gamma_step_std: float = 0.0
    delta_gamma: float = 0.0
    n_bar0: float = 0.0
    n_bar_min: float = 0.0
    n_bar_max: float = 0.2
    n_bar_alpha: float = 0.0
    n_bar_amplitude: float = 0.0
    n_bar_frequency: float = 0.0
    n_bar_step_std: float = 0.0
    delta_n_bar: float = 0.0
    sigma_displacement0: float = 0.0
    sigma_displacement_min: float = 0.0
    sigma_displacement_max: float = 0.7
    sigma_displacement_alpha: float = 0.0
    sigma_displacement_amplitude: float = 0.0
    sigma_displacement_frequency: float = 0.0
    sigma_displacement_step_std: float = 0.0
    delta_sigma_displacement: float = 0.0
    sigma_phase0: float = 0.0
    sigma_phase_min: float = 0.0
    sigma_phase_max: float = np.pi / 4.0
    sigma_phase_alpha: float = 0.0
    sigma_phase_amplitude: float = 0.0
    sigma_phase_frequency: float = 0.0
    sigma_phase_step_std: float = 0.0
    delta_sigma_phase: float = 0.0
    mu_q_bias0: float = 0.0
    mu_q_bias_min: Optional[float] = None
    mu_q_bias_max: Optional[float] = None
    mu_q_bias_alpha: float = 0.0
    mu_q_bias_amplitude: float = 0.0
    mu_q_bias_frequency: float = 0.0
    mu_q_bias_step_std: float = 0.0
    delta_mu_q_bias: float = 0.0
    mu_p_bias0: float = 0.0
    mu_p_bias_min: Optional[float] = None
    mu_p_bias_max: Optional[float] = None
    mu_p_bias_alpha: float = 0.0
    mu_p_bias_amplitude: float = 0.0
    mu_p_bias_frequency: float = 0.0
    mu_p_bias_step_std: float = 0.0
    delta_mu_p_bias: float = 0.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PhysicalNoiseBridgeConfig":
        signal_cfg = dict(config.get("mock_signal", {}))
        raw = dict(signal_cfg.get("physical_bridge", {}) or {})
        theta_clip = raw.get("theta_clip_deg", [-30.0, 30.0])
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mapping_mode=str(raw.get("mapping_mode", "hybrid_additive")).lower(),
            use_base_signal=bool(raw.get("use_base_signal", True)),
            sigma_combine_mode=str(raw.get("sigma_combine_mode", "quadrature")).lower(),
            sigma_scale=float(raw.get("sigma_scale", 1.0)),
            mu_q_scale=float(raw.get("mu_q_scale", 1.0)),
            mu_p_scale=float(raw.get("mu_p_scale", 1.0)),
            phase_to_theta_gain_deg=float(raw.get("phase_to_theta_gain_deg", 180.0 / np.pi)),
            theta_bias_deg=float(raw.get("theta_bias_deg", 0.0)),
            theta_clip_min_deg=float(theta_clip[0]),
            theta_clip_max_deg=float(theta_clip[1]),
            sigma_reference=float(raw.get("sigma_reference", 0.12)),
            mu_reference=float(raw.get("mu_reference", 0.02)),
            theta_reference_deg=float(raw.get("theta_reference_deg", 6.0)),
            gamma0=float(raw.get("gamma0", 0.0)),
            gamma_min=float(raw.get("gamma_min", 0.0)),
            gamma_max=float(raw.get("gamma_max", 0.25)),
            gamma_alpha=float(raw.get("gamma_alpha", 0.0)),
            gamma_amplitude=float(raw.get("gamma_amplitude", 0.0)),
            gamma_frequency=float(raw.get("gamma_frequency", 0.0)),
            gamma_step_std=float(raw.get("gamma_step_std", 0.0)),
            delta_gamma=float(raw.get("delta_gamma", 0.0)),
            n_bar0=float(raw.get("n_bar0", 0.0)),
            n_bar_min=float(raw.get("n_bar_min", 0.0)),
            n_bar_max=float(raw.get("n_bar_max", 0.2)),
            n_bar_alpha=float(raw.get("n_bar_alpha", 0.0)),
            n_bar_amplitude=float(raw.get("n_bar_amplitude", 0.0)),
            n_bar_frequency=float(raw.get("n_bar_frequency", 0.0)),
            n_bar_step_std=float(raw.get("n_bar_step_std", 0.0)),
            delta_n_bar=float(raw.get("delta_n_bar", 0.0)),
            sigma_displacement0=float(raw.get("sigma_displacement0", 0.0)),
            sigma_displacement_min=float(raw.get("sigma_displacement_min", 0.0)),
            sigma_displacement_max=float(raw.get("sigma_displacement_max", 0.7)),
            sigma_displacement_alpha=float(raw.get("sigma_displacement_alpha", 0.0)),
            sigma_displacement_amplitude=float(raw.get("sigma_displacement_amplitude", 0.0)),
            sigma_displacement_frequency=float(raw.get("sigma_displacement_frequency", 0.0)),
            sigma_displacement_step_std=float(raw.get("sigma_displacement_step_std", 0.0)),
            delta_sigma_displacement=float(raw.get("delta_sigma_displacement", 0.0)),
            sigma_phase0=float(raw.get("sigma_phase0", 0.0)),
            sigma_phase_min=float(raw.get("sigma_phase_min", 0.0)),
            sigma_phase_max=float(raw.get("sigma_phase_max", np.pi / 4.0)),
            sigma_phase_alpha=float(raw.get("sigma_phase_alpha", 0.0)),
            sigma_phase_amplitude=float(raw.get("sigma_phase_amplitude", 0.0)),
            sigma_phase_frequency=float(raw.get("sigma_phase_frequency", 0.0)),
            sigma_phase_step_std=float(raw.get("sigma_phase_step_std", 0.0)),
            delta_sigma_phase=float(raw.get("delta_sigma_phase", 0.0)),
            mu_q_bias0=float(raw.get("mu_q_bias0", 0.0)),
            mu_q_bias_min=None if raw.get("mu_q_bias_min") is None else float(raw.get("mu_q_bias_min")),
            mu_q_bias_max=None if raw.get("mu_q_bias_max") is None else float(raw.get("mu_q_bias_max")),
            mu_q_bias_alpha=float(raw.get("mu_q_bias_alpha", 0.0)),
            mu_q_bias_amplitude=float(raw.get("mu_q_bias_amplitude", 0.0)),
            mu_q_bias_frequency=float(raw.get("mu_q_bias_frequency", 0.0)),
            mu_q_bias_step_std=float(raw.get("mu_q_bias_step_std", 0.0)),
            delta_mu_q_bias=float(raw.get("delta_mu_q_bias", 0.0)),
            mu_p_bias0=float(raw.get("mu_p_bias0", 0.0)),
            mu_p_bias_min=None if raw.get("mu_p_bias_min") is None else float(raw.get("mu_p_bias_min")),
            mu_p_bias_max=None if raw.get("mu_p_bias_max") is None else float(raw.get("mu_p_bias_max")),
            mu_p_bias_alpha=float(raw.get("mu_p_bias_alpha", 0.0)),
            mu_p_bias_amplitude=float(raw.get("mu_p_bias_amplitude", 0.0)),
            mu_p_bias_frequency=float(raw.get("mu_p_bias_frequency", 0.0)),
            mu_p_bias_step_std=float(raw.get("mu_p_bias_step_std", 0.0)),
            delta_mu_p_bias=float(raw.get("delta_mu_p_bias", 0.0)),
        )


class PhysicalNoiseBridge:
    """Stateful physical-channel augmentation for runtime noise providers."""

    def __init__(self, config: PhysicalNoiseBridgeConfig, *, kind: str, jump_round: int, seed: Optional[int] = None) -> None:
        self.config = config
        self.kind = str(kind).lower()
        self.jump_round = int(jump_round)
        self._rng = np.random.default_rng(seed)
        self._gamma = config.gamma0
        self._n_bar = config.n_bar0
        self._sigma_displacement = config.sigma_displacement0
        self._sigma_phase = config.sigma_phase0
        self._mu_q_bias = config.mu_q_bias0
        self._mu_p_bias = config.mu_p_bias0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _effective_components(self) -> Dict[str, float]:
        sigma_loss = float(np.sqrt(max(0.0, self._gamma) / 2.0))
        sigma_thermal = float(np.sqrt(max(0.0, self._n_bar)))
        sigma_disp = float(max(0.0, self._sigma_displacement))
        sigma_total = float(np.sqrt(sigma_loss**2 + sigma_thermal**2 + sigma_disp**2))
        theta_physical_deg = float(self.config.theta_bias_deg + self._sigma_phase * self.config.phase_to_theta_gain_deg)
        mu_q_physical = float(self._mu_q_bias * self.config.mu_q_scale)
        mu_p_physical = float(self._mu_p_bias * self.config.mu_p_scale)
        return {
            "sigma_loss": sigma_loss,
            "sigma_thermal": sigma_thermal,
            "sigma_displacement": sigma_disp,
            "sigma_total": sigma_total,
            "theta_deg": theta_physical_deg,
            "mu_q": mu_q_physical,
            "mu_p": mu_p_physical,
        }

    def _combine_sigma(self, base_sigma: float, sigma_delta: float, *, use_base_signal: bool) -> float:
        mode = self.config.sigma_combine_mode
        if use_base_signal and mode == "add":
            return float(base_sigma + sigma_delta)
        if use_base_signal and mode == "override":
            return float(sigma_delta)
        if use_base_signal:
            return float(np.sqrt(base_sigma**2 + sigma_delta**2))
        return float(sigma_delta)

    def _bounded(self, value: float, reference: float) -> float:
        ref = max(1.0e-8, float(reference))
        return float(ref * np.tanh(float(value) / ref))

    def _update_state(self, epoch_id: int) -> None:
        cfg = self.config
        self._gamma = _evolve_scalar(
            self.kind,
            epoch_id=epoch_id,
            current=self._gamma,
            initial=cfg.gamma0,
            alpha=cfg.gamma_alpha,
            amplitude=cfg.gamma_amplitude,
            frequency=cfg.gamma_frequency,
            step_std=cfg.gamma_step_std,
            jump_round=self.jump_round,
            delta=cfg.delta_gamma,
            minimum=cfg.gamma_min,
            maximum=cfg.gamma_max,
            rng=self._rng,
        )
        self._n_bar = _evolve_scalar(
            self.kind,
            epoch_id=epoch_id,
            current=self._n_bar,
            initial=cfg.n_bar0,
            alpha=cfg.n_bar_alpha,
            amplitude=cfg.n_bar_amplitude,
            frequency=cfg.n_bar_frequency,
            step_std=cfg.n_bar_step_std,
            jump_round=self.jump_round,
            delta=cfg.delta_n_bar,
            minimum=cfg.n_bar_min,
            maximum=cfg.n_bar_max,
            rng=self._rng,
        )
        self._sigma_displacement = _evolve_scalar(
            self.kind,
            epoch_id=epoch_id,
            current=self._sigma_displacement,
            initial=cfg.sigma_displacement0,
            alpha=cfg.sigma_displacement_alpha,
            amplitude=cfg.sigma_displacement_amplitude,
            frequency=cfg.sigma_displacement_frequency,
            step_std=cfg.sigma_displacement_step_std,
            jump_round=self.jump_round,
            delta=cfg.delta_sigma_displacement,
            minimum=cfg.sigma_displacement_min,
            maximum=cfg.sigma_displacement_max,
            rng=self._rng,
        )
        self._sigma_phase = _evolve_scalar(
            self.kind,
            epoch_id=epoch_id,
            current=self._sigma_phase,
            initial=cfg.sigma_phase0,
            alpha=cfg.sigma_phase_alpha,
            amplitude=cfg.sigma_phase_amplitude,
            frequency=cfg.sigma_phase_frequency,
            step_std=cfg.sigma_phase_step_std,
            jump_round=self.jump_round,
            delta=cfg.delta_sigma_phase,
            minimum=cfg.sigma_phase_min,
            maximum=cfg.sigma_phase_max,
            rng=self._rng,
        )
        self._mu_q_bias = _evolve_scalar(
            self.kind,
            epoch_id=epoch_id,
            current=self._mu_q_bias,
            initial=self.config.mu_q_bias0,
            alpha=self.config.mu_q_bias_alpha,
            amplitude=self.config.mu_q_bias_amplitude,
            frequency=self.config.mu_q_bias_frequency,
            step_std=self.config.mu_q_bias_step_std,
            jump_round=self.jump_round,
            delta=self.config.delta_mu_q_bias,
            minimum=self.config.mu_q_bias_min,
            maximum=self.config.mu_q_bias_max,
            rng=self._rng,
        )
        self._mu_p_bias = _evolve_scalar(
            self.kind,
            epoch_id=epoch_id,
            current=self._mu_p_bias,
            initial=self.config.mu_p_bias0,
            alpha=self.config.mu_p_bias_alpha,
            amplitude=self.config.mu_p_bias_amplitude,
            frequency=self.config.mu_p_bias_frequency,
            step_std=self.config.mu_p_bias_step_std,
            jump_round=self.jump_round,
            delta=self.config.delta_mu_p_bias,
            minimum=self.config.mu_p_bias_min,
            maximum=self.config.mu_p_bias_max,
            rng=self._rng,
        )

    def apply(self, epoch_id: int, base_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        self._update_state(epoch_id)
        components = self._effective_components()
        sigma_physical = float(components["sigma_total"] * self.config.sigma_scale)
        theta_physical_deg = float(components["theta_deg"])
        mu_q_physical = float(components["mu_q"])
        mu_p_physical = float(components["mu_p"])

        base_sigma = _as_float(None if base_payload is None else base_payload.get("sigma"), 0.0)
        base_mu_q = _as_float(None if base_payload is None else base_payload.get("mu_q"), 0.0)
        base_mu_p = _as_float(None if base_payload is None else base_payload.get("mu_p"), 0.0)
        base_theta = _as_float(None if base_payload is None else base_payload.get("theta_deg"), 0.0)

        mapping_mode = self.config.mapping_mode
        if mapping_mode == "hybrid_additive":
            use_base_signal = self.config.use_base_signal
            sigma_delta = sigma_physical
            mu_q_delta = mu_q_physical
            mu_p_delta = mu_p_physical
            theta_delta = theta_physical_deg
        elif mapping_mode == "bounded_residual":
            use_base_signal = self.config.use_base_signal
            sigma_delta = self._bounded(sigma_physical, self.config.sigma_reference)
            mu_q_delta = self._bounded(mu_q_physical, self.config.mu_reference)
            mu_p_delta = self._bounded(mu_p_physical, self.config.mu_reference)
            theta_delta = self._bounded(theta_physical_deg, self.config.theta_reference_deg)
        elif mapping_mode == "sigma_theta_only":
            use_base_signal = True
            sigma_delta = self._bounded(sigma_physical, self.config.sigma_reference)
            mu_q_delta = 0.0
            mu_p_delta = 0.0
            theta_delta = self._bounded(theta_physical_deg, self.config.theta_reference_deg)
        else:
            raise ValueError(f"Unsupported physical_bridge.mapping_mode: {mapping_mode}")

        sigma_value = self._combine_sigma(base_sigma, sigma_delta, use_base_signal=use_base_signal)
        mu_q_value = base_mu_q + mu_q_delta if use_base_signal else mu_q_delta
        mu_p_value = base_mu_p + mu_p_delta if use_base_signal else mu_p_delta
        theta_value = base_theta + theta_delta if use_base_signal else theta_delta
        theta_value = float(np.clip(theta_value, self.config.theta_clip_min_deg, self.config.theta_clip_max_deg))

        metadata = {}
        if base_payload is not None:
            metadata.update(dict(base_payload.get("metadata", {})))
        metadata.update(
            {
                "signal_type": self.kind,
                "effective_source": f"physical_bridge:{mapping_mode}",
                "physical_bridge": {
                    "enabled": True,
                    "mapping_mode": mapping_mode,
                    "use_base_signal": bool(use_base_signal),
                    "sigma_combine_mode": self.config.sigma_combine_mode,
                    "gamma": float(self._gamma),
                    "n_bar": float(self._n_bar),
                    "sigma_displacement": float(self._sigma_displacement),
                    "sigma_phase": float(self._sigma_phase),
                    "mu_q_bias": float(self._mu_q_bias),
                    "mu_p_bias": float(self._mu_p_bias),
                    "sigma_physical": sigma_physical,
                    "theta_physical_deg": theta_physical_deg,
                    "mu_q_physical": mu_q_physical,
                    "mu_p_physical": mu_p_physical,
                    "sigma_delta_applied": float(sigma_delta),
                    "theta_delta_applied": float(theta_delta),
                    "mu_q_delta_applied": float(mu_q_delta),
                    "mu_p_delta_applied": float(mu_p_delta),
                    "base_sigma": base_sigma,
                    "base_mu_q": base_mu_q,
                    "base_mu_p": base_mu_p,
                    "base_theta_deg": base_theta,
                    **components,
                },
            }
        )
        return {
            "sigma": float(sigma_value),
            "mu_q": float(mu_q_value),
            "mu_p": float(mu_p_value),
            "theta_deg": float(theta_value),
            "metadata": metadata,
        }


def maybe_build_physical_noise_bridge(config: Dict[str, Any], *, seed: Optional[int] = None) -> Optional[PhysicalNoiseBridge]:
    bridge_cfg = PhysicalNoiseBridgeConfig.from_config(config)
    if not bridge_cfg.enabled:
        return None
    signal_cfg = dict(config.get("mock_signal", {}))
    return PhysicalNoiseBridge(
        bridge_cfg,
        kind=str(signal_cfg.get("type", "random_walk")).lower(),
        jump_round=int(signal_cfg.get("jump_round", 0)),
        seed=seed,
    )
