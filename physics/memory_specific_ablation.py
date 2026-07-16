"""Causal frozen-history interventions for the T3.2.11 memory audit.

The wrappers reuse a trained :class:`PaperScaleNMFPolicy` without changing any
weights.  They alter only which already-observed prefix tokens are replayed
into the GRU, so future outcomes and evaluator truth are never available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .nmf_directional_ranking import PaperScaleNMFPolicy

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


InterventionMode = Literal[
    "full_history",
    "history_shuffle",
    "history_truncation",
    "periodic_hidden_reset",
    "last_outcome_only",
]
INTERVENTION_MODES = (
    "full_history",
    "history_shuffle",
    "history_truncation",
    "periodic_hidden_reset",
    "last_outcome_only",
)


@dataclass(frozen=True)
class MemoryInterventionSpec:
    mode: InterventionMode
    history_length: int | None = None
    reset_period: int | None = None
    shuffle_seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in INTERVENTION_MODES:
            raise ValueError(f"mode must be one of {INTERVENTION_MODES}")
        required = {
            "history_length": self.mode == "history_truncation",
            "reset_period": self.mode == "periodic_hidden_reset",
            "shuffle_seed": self.mode == "history_shuffle",
        }
        for name, needed in required.items():
            value = getattr(self, name)
            if needed:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                    raise ValueError(f"{name} must be a positive integer for {self.mode}")
                object.__setattr__(self, name, int(value))
            elif value is not None:
                raise ValueError(f"{name} is only valid for its matching intervention mode")

    @property
    def identifier(self) -> str:
        if self.mode == "history_truncation":
            return f"history_truncation_L{self.history_length}"
        if self.mode == "periodic_hidden_reset":
            return f"periodic_hidden_reset_R{self.reset_period}"
        if self.mode == "history_shuffle":
            return f"history_shuffle_seed{self.shuffle_seed}"
        return self.mode


if torch is not None:

    class FrozenMemoryInterventionPolicy(torch.nn.Module):
        """Read-only view of a trained NMF policy under one causal intervention."""

        strategy = "nmf_frozen_memory_intervention"

        def __init__(self, parent: PaperScaleNMFPolicy, spec: MemoryInterventionSpec) -> None:
            super().__init__()
            if not isinstance(parent, PaperScaleNMFPolicy):
                raise TypeError("parent must be a trained PaperScaleNMFPolicy")
            if not isinstance(spec, MemoryInterventionSpec):
                raise TypeError("spec must be MemoryInterventionSpec")
            self.parent = parent
            self.spec = spec

        @property
        def parameter_count(self) -> int:
            return self.parent.parameter_count

        @staticmethod
        def _shuffle_key(position: int, seed: int) -> int:
            """Stable SplitMix64 key for prefix-consistent causal shuffling."""

            mask = (1 << 64) - 1
            value = (int(seed) + 0x9E3779B97F4A7C15 * (int(position) + 1)) & mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
            return int((value ^ (value >> 31)) & mask)

        def history_indices(self, half_index: int) -> tuple[int, ...]:
            if isinstance(half_index, bool) or not isinstance(half_index, int) or half_index < 0:
                raise ValueError("half_index must be a nonnegative integer")
            if self.spec.mode == "full_history":
                return tuple(range(half_index))
            if self.spec.mode == "last_outcome_only":
                return () if half_index == 0 else (half_index - 1,)
            if self.spec.mode == "history_truncation":
                length = int(self.spec.history_length)
                return tuple(range(max(0, half_index - length), half_index))
            if self.spec.mode == "periodic_hidden_reset":
                if half_index == 0:
                    return ()
                period = int(self.spec.reset_period)
                start = ((half_index - 1) // period) * period
                return tuple(range(start, half_index))
            if half_index <= 1:
                return tuple(range(half_index))
            # Assign each absolute history position one fixed random priority.
            # Restricting that global ordering to [0, half_index) is causal and
            # preserves the relative shuffle order of all previously observed
            # tokens when the prefix grows.
            seed = int(self.spec.shuffle_seed)
            return tuple(
                sorted(
                    range(half_index),
                    key=lambda index: (self._shuffle_key(index, seed), index),
                )
            )

        def transformed_history(self, history: Any, half_index: int) -> Any:
            if not isinstance(history, torch.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            indices = self.history_indices(half_index)
            if not indices:
                return history[:, :0]
            index = torch.tensor(indices, dtype=torch.int64, device=history.device)
            return history.index_select(1, index)

        def forward(self, history: Any, half_index: int) -> Any:
            selected = self.transformed_history(history, half_index)
            reference = next(self.parent.parameters())
            hidden = torch.zeros(
                (history.shape[0], 10), dtype=reference.dtype, device=reference.device
            )
            for index in range(selected.shape[1]):
                value = (2.0 * selected[:, index : index + 1] - 1.0).to(reference.dtype)
                hidden = self.parent.gru(value, hidden)
            value = torch.tanh(self.parent.dense1(hidden))
            value = torch.tanh(self.parent.dense2(value))
            return self.parent.output(value)


else:  # pragma: no cover

    class FrozenMemoryInterventionPolicy:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("memory-specific ablations require PyTorch")


__all__ = [
    "INTERVENTION_MODES",
    "InterventionMode",
    "MemoryInterventionSpec",
    "FrozenMemoryInterventionPolicy",
]
