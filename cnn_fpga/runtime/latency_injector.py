"""Latency sampling utilities for dual-loop runtime emulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class StageLatencySpec:
    """Sampling rule for one latency stage."""

    mean_us: float
    std_us: float = 0.0
    distribution: str = "normal"
    min_us: float = 0.0
    max_us: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mean_us < 0:
            raise ValueError("mean_us must be non-negative")
        if self.std_us < 0:
            raise ValueError("std_us must be non-negative")
        if self.min_us < 0:
            raise ValueError("min_us must be non-negative")
        if self.max_us is not None and self.max_us < self.min_us:
            raise ValueError("max_us must be >= min_us")

    def sample(self, rng: np.random.Generator) -> float:
        distribution = self.distribution.lower()
        if distribution in {"constant", "fixed"} or self.std_us == 0.0:
            value = self.mean_us
        elif distribution in {"normal", "gaussian"}:
            value = float(rng.normal(self.mean_us, self.std_us))
        else:
            raise ValueError(f"Unsupported latency distribution: {self.distribution}")

        value = max(self.min_us, value)
        if self.max_us is not None:
            value = min(self.max_us, value)
        return float(value)


@dataclass
class LatencySample:
    """Concrete latency sample for one slow-loop update."""

    dma_us: float
    preprocess_us: float
    inference_us: float
    writeback_us: float
    commit_ack_us: float = 0.0
    fast_cycle_us: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_us(self) -> float:
        return (
            self.dma_us
            + self.preprocess_us
            + self.inference_us
            + self.writeback_us
            + self.commit_ack_us
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dma_us": self.dma_us,
            "preprocess_us": self.preprocess_us,
            "inference_us": self.inference_us,
            "writeback_us": self.writeback_us,
            "commit_ack_us": self.commit_ack_us,
            "fast_cycle_us": self.fast_cycle_us,
            "total_us": self.total_us,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LatencyContext:
    """Runtime state used by load-aware latency sampling."""

    pending_windows: int = 0
    slow_job_inflight: bool = False
    pending_commit: bool = False
    recent_slow_budget_violations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pending_windows": int(self.pending_windows),
            "slow_job_inflight": bool(self.slow_job_inflight),
            "pending_commit": bool(self.pending_commit),
            "recent_slow_budget_violations": int(self.recent_slow_budget_violations),
        }


class LatencyInjector:
    """Samples fast-path and slow-path latencies from runtime profiles."""

    def __init__(
        self,
        *,
        dma: Optional[StageLatencySpec] = None,
        preprocess: Optional[StageLatencySpec] = None,
        inference: Optional[StageLatencySpec] = None,
        writeback: Optional[StageLatencySpec] = None,
        commit_ack: Optional[StageLatencySpec] = None,
        fast_cycle: Optional[StageLatencySpec] = None,
        load_aware_enabled: bool = False,
        load_mean_scale_per_pending_window: float = 0.0,
        load_std_scale_per_pending_window: float = 0.0,
        inflight_mean_scale: float = 0.0,
        inflight_std_scale: float = 0.0,
        commit_pending_mean_scale: float = 0.0,
        violation_mean_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.dma = dma or StageLatencySpec(mean_us=10.0, std_us=2.0)
        self.preprocess = preprocess or StageLatencySpec(mean_us=50.0, std_us=10.0)
        self.inference = inference or StageLatencySpec(mean_us=800.0, std_us=120.0)
        self.writeback = writeback or StageLatencySpec(mean_us=20.0, std_us=5.0)
        self.commit_ack = commit_ack or StageLatencySpec(mean_us=5.0, std_us=1.0)
        self.fast_cycle = fast_cycle or StageLatencySpec(mean_us=1.0, std_us=0.15)
        self.load_aware_enabled = bool(load_aware_enabled)
        self.load_mean_scale_per_pending_window = float(load_mean_scale_per_pending_window)
        self.load_std_scale_per_pending_window = float(load_std_scale_per_pending_window)
        self.inflight_mean_scale = float(inflight_mean_scale)
        self.inflight_std_scale = float(inflight_std_scale)
        self.commit_pending_mean_scale = float(commit_pending_mean_scale)
        self.violation_mean_scale = float(violation_mean_scale)

    @classmethod
    def from_config(cls, config: Dict[str, Any], seed: Optional[int] = None) -> "LatencyInjector":
        latency_cfg = config.get("latency_model", {})
        fast_cfg = config.get("fast_path_model", {})
        load_cfg = dict(latency_cfg.get("load_aware", {}) or {})

        def _stage(prefix: str, default_mean: float, default_std: float) -> StageLatencySpec:
            return StageLatencySpec(
                mean_us=float(latency_cfg.get(f"{prefix}_mean_us", default_mean)),
                std_us=float(latency_cfg.get(f"{prefix}_std_us", default_std)),
            )

        return cls(
            dma=_stage("dma", 10.0, 2.0),
            preprocess=_stage("preprocess", 50.0, 10.0),
            inference=_stage("inference", 800.0, 120.0),
            writeback=_stage("writeback", 20.0, 5.0),
            commit_ack=StageLatencySpec(
                mean_us=float(latency_cfg.get("commit_ack_mean_us", 5.0)),
                std_us=float(latency_cfg.get("commit_ack_std_us", 1.0)),
            ),
            fast_cycle=StageLatencySpec(
                mean_us=float(fast_cfg.get("latency_mean_us", 1.0)),
                std_us=float(fast_cfg.get("latency_std_us", 0.15)),
            ),
            load_aware_enabled=bool(load_cfg.get("enabled", False)),
            load_mean_scale_per_pending_window=float(load_cfg.get("mean_scale_per_pending_window", 0.0)),
            load_std_scale_per_pending_window=float(load_cfg.get("std_scale_per_pending_window", 0.0)),
            inflight_mean_scale=float(load_cfg.get("inflight_mean_scale", 0.0)),
            inflight_std_scale=float(load_cfg.get("inflight_std_scale", 0.0)),
            commit_pending_mean_scale=float(load_cfg.get("commit_pending_mean_scale", 0.0)),
            violation_mean_scale=float(load_cfg.get("violation_mean_scale", 0.0)),
            seed=seed,
        )

    def sample_fast_cycle(self) -> float:
        return self.fast_cycle.sample(self._rng)

    def _sample_stage_with_context(
        self,
        spec: StageLatencySpec,
        *,
        context: Optional[LatencyContext],
    ) -> float:
        if context is None or not self.load_aware_enabled:
            return spec.sample(self._rng)
        pending = max(0, int(context.pending_windows))
        mean_multiplier = 1.0 + pending * self.load_mean_scale_per_pending_window
        std_multiplier = 1.0 + pending * self.load_std_scale_per_pending_window
        if context.slow_job_inflight:
            mean_multiplier += self.inflight_mean_scale
            std_multiplier += self.inflight_std_scale
        if context.pending_commit:
            mean_multiplier += self.commit_pending_mean_scale
        if context.recent_slow_budget_violations > 0:
            mean_multiplier += context.recent_slow_budget_violations * self.violation_mean_scale

        derived = StageLatencySpec(
            mean_us=max(spec.min_us, spec.mean_us * mean_multiplier),
            std_us=max(0.0, spec.std_us * max(0.0, std_multiplier)),
            distribution=spec.distribution,
            min_us=spec.min_us,
            max_us=spec.max_us,
        )
        return derived.sample(self._rng)

    def sample_slow_update(self, *, context: Optional[LatencyContext] = None) -> LatencySample:
        metadata = {}
        if context is not None:
            metadata["load_context"] = context.to_dict()
            metadata["load_aware_enabled"] = bool(self.load_aware_enabled)
        return LatencySample(
            dma_us=self._sample_stage_with_context(self.dma, context=context),
            preprocess_us=self._sample_stage_with_context(self.preprocess, context=context),
            inference_us=self._sample_stage_with_context(self.inference, context=context),
            writeback_us=self._sample_stage_with_context(self.writeback, context=context),
            commit_ack_us=self._sample_stage_with_context(self.commit_ack, context=context),
            metadata=metadata,
        )
