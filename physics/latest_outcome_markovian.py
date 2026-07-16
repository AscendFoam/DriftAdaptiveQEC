"""Budget-matched latest-outcome Markovian feedback for T3.2.7.

The physical simulator exposes a history tensor because recurrent policies need
it.  :class:`BudgetMatchedMarkovianPolicy` deliberately narrows that interface:
its learned path receives only the most recent observed token (``g``, ``e`` or
``leakage``), or an all-zero start token before the first observation.

The static front-end has exactly the same 390 trainable parameters and 330
dense MACs as the GRU(1, 10) front-end used by the frozen T2.3.7 history model.
The downstream 10--256--256--15 action trunk is identical in shape.  Therefore
both policies have 72,853 parameters and 72,266 dense MACs per half-cycle;
capacity is not hidden in an unused padding tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

# Import the numpy/scipy-backed physics layer before torch.  This matches the
# established DLEnv import order and avoids loading two OpenMP runtimes in the
# opposite order on Windows.
from physics.differentiable_sbs_trajectory import PARAMETER_NAMES

try:  # Keep the rest of physics importable in environments without torch.
    import torch
except ImportError:  # pragma: no cover - exercised by the recovery interpreter.
    torch = None


OUTCOME_ALPHABET_ID = "T327-OBSERVED-LATEST-G-E-LEAKAGE-V1"
ARCHITECTURE_ID = "LATEST3-STATIC390-DENSE256-DENSE256-OUT15"
HISTORY_REFERENCE_ARCHITECTURE_ID = "GRU1X10-DENSE256-DENSE256-OUT15"
ACTION_CONTRACT_ID = "T237-NOMINAL-SBS-RESIDUAL-15"
TRAINING_PROTOCOL_ID = "T237-FEEDBACK-GRAPE-TRAIN-ONLY-EMA-BASELINE-V1"


class ObservedOutcome(IntEnum):
    G = 0
    E = 1
    LEAKAGE = 2


@dataclass(frozen=True)
class ComputeContract:
    input_classes: int = 3
    static_feature_width: int = 10
    adapter_width: int = 15
    hidden_width: int = 256
    output_width: int = len(PARAMETER_NAMES)
    front_parameter_count: int = 390
    downstream_parameter_count: int = 72_463
    total_parameter_count: int = 72_853
    front_dense_macs: int = 330
    downstream_dense_macs: int = 71_936
    total_dense_macs: int = 72_266

    def __post_init__(self) -> None:
        if self.output_width != 15:
            raise ValueError("the frozen sBs residual action must have 15 outputs")
        if self.front_parameter_count + self.downstream_parameter_count != self.total_parameter_count:
            raise ValueError("parameter-count decomposition is inconsistent")
        if self.front_dense_macs + self.downstream_dense_macs != self.total_dense_macs:
            raise ValueError("MAC decomposition is inconsistent")


COMPUTE_CONTRACT = ComputeContract()


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for the Markovian feedback policy")
    return torch


if torch is not None:

    class BudgetMatchedMarkovianPolicy(torch.nn.Module):
        """FNN policy whose learned state is exactly the latest observed token.

        ``forward`` is the simulator adapter.  ``forward_latest`` is the
        canonical learned interface and has no history argument or recurrent
        state.  The third leakage token is implemented even though the frozen
        two-level T2.3.7 simulator emits only g/e; that evidence boundary is
        recorded by the T3.2.7 benchmark rather than concealed.
        """

        strategy = "budget_matched_markovian"

        def __init__(self, *, device: str, dtype: Any, seed: int) -> None:
            super().__init__()
            th = _require_torch()

            # 40 parameters.
            self.outcome_encoder = th.nn.Linear(3, 10)
            # 165 + 160 + 20 + 5 = 350 parameters.  Together with the
            # encoder this static front-end exactly replaces the 390-parameter
            # GRU front-end.  Every tensor is used in the forward path.
            self.adapter_down = th.nn.Linear(10, 15)
            self.adapter_up = th.nn.Linear(15, 10)
            self.adapter_norm = th.nn.LayerNorm(10)
            self.feature_pair_scale = th.nn.Parameter(th.ones(5))

            self.dense1 = th.nn.Linear(10, 256)
            self.dense2 = th.nn.Linear(256, 256)
            self.output = th.nn.Linear(256, len(PARAMETER_NAMES))

            generator = th.Generator(device="cpu")
            generator.manual_seed(int(seed))
            for layer, gain in (
                (self.outcome_encoder, 1.0),
                (self.adapter_down, 0.5),
                (self.adapter_up, 0.5),
                (self.dense1, 0.5),
                (self.dense2, 0.5),
                (self.output, 0.05),
            ):
                th.nn.init.xavier_uniform_(layer.weight, gain=gain, generator=generator)
                th.nn.init.uniform_(layer.bias, -0.01, 0.01, generator=generator)
            th.nn.init.zeros_(self.output.bias)
            th.nn.init.ones_(self.adapter_norm.weight)
            th.nn.init.zeros_(self.adapter_norm.bias)
            self.to(device=device, dtype=dtype)

            if self.parameter_count != COMPUTE_CONTRACT.total_parameter_count:
                raise RuntimeError(
                    f"parameter budget drifted: {self.parameter_count} != "
                    f"{COMPUTE_CONTRACT.total_parameter_count}"
                )

        @property
        def parameter_count(self) -> int:
            return int(sum(parameter.numel() for parameter in self.parameters()))

        @property
        def front_parameter_count(self) -> int:
            names = (
                "outcome_encoder",
                "adapter_down",
                "adapter_up",
                "adapter_norm",
            )
            count = sum(
                parameter.numel()
                for name in names
                for parameter in getattr(self, name).parameters()
            )
            return int(count + self.feature_pair_scale.numel())

        @property
        def dense_mac_count(self) -> int:
            return COMPUTE_CONTRACT.total_dense_macs

        def _encode_tokens(self, latest_outcomes: Any | None, *, batch_size: int | None = None) -> Any:
            th = _require_torch()
            reference = next(self.parameters())
            if latest_outcomes is None:
                if batch_size is None or int(batch_size) <= 0:
                    raise ValueError("positive batch_size is required for the start token")
                return th.zeros(
                    (int(batch_size), 3), dtype=reference.dtype, device=reference.device
                )
            if not isinstance(latest_outcomes, th.Tensor):
                raise TypeError("latest_outcomes must be a torch.Tensor or None")
            if latest_outcomes.ndim != 1:
                raise ValueError("latest_outcomes must be a rank-one token vector")
            if latest_outcomes.numel() == 0:
                raise ValueError("latest_outcomes must be nonempty")
            if latest_outcomes.dtype == th.bool or th.is_floating_point(latest_outcomes):
                raise TypeError("latest_outcomes must use an integer dtype")
            tokens = latest_outcomes.to(device=reference.device, dtype=th.int64)
            if bool(th.any((tokens < 0) | (tokens >= 3)).detach().cpu()):
                raise ValueError("latest_outcomes must be g=0, e=1, or leakage=2")
            return th.nn.functional.one_hot(tokens, num_classes=3).to(reference.dtype)

        def forward_latest(
            self, latest_outcomes: Any | None, *, batch_size: int | None = None
        ) -> Any:
            one_hot = self._encode_tokens(latest_outcomes, batch_size=batch_size)
            base = self.outcome_encoder(one_hot)
            adapted = self.adapter_up(torch.tanh(self.adapter_down(base)))
            features = self.adapter_norm(base + adapted)
            features = features * torch.repeat_interleave(self.feature_pair_scale, 2)[None, :]
            value = torch.tanh(self.dense1(features))
            value = torch.tanh(self.dense2(value))
            return self.output(value)

        def forward(self, history: Any, half_index: int) -> Any:
            """Simulator adapter; only ``history[:, -1]`` enters the model."""

            th = _require_torch()
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if not isinstance(half_index, int) or half_index < 0:
                raise ValueError("half_index must be a nonnegative integer")
            if history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            if half_index == 0:
                return self.forward_latest(None, batch_size=int(history.shape[0]))
            return self.forward_latest(history[:, -1])


else:

    class BudgetMatchedMarkovianPolicy:  # pragma: no cover - recovery interpreter.
        def __init__(self, **_: Any) -> None:
            _require_torch()


def build_budget_matched_policy(*, device: str, dtype: Any, seed: int) -> Any:
    return BudgetMatchedMarkovianPolicy(device=device, dtype=dtype, seed=int(seed))


def audit_latest_only_behavior(model: Any) -> Mapping[str, bool | float]:
    """Run behavioral causality and all-token audits on a frozen policy."""

    th = _require_torch()
    reference = next(model.parameters())
    first = th.tensor(
        [[0, 0, 1], [1, 0, 1]], dtype=th.int64, device=reference.device
    )
    with th.no_grad():
        outputs = model(first, 3)
        tokens = th.tensor([0, 1, 2], dtype=th.int64, device=reference.device)
        token_outputs = model.forward_latest(tokens)
        repeated = model(first, 3)
    earlier_history_invariant = bool(th.equal(outputs[0], outputs[1]))
    stateless_repeat_exact = bool(th.equal(outputs, repeated))
    pairwise = [
        float(th.max(th.abs(token_outputs[left] - token_outputs[right])).detach().cpu())
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    return {
        "earlier_history_invariant_bit_exact": earlier_history_invariant,
        "stateless_repeat_bit_exact": stateless_repeat_exact,
        "all_three_tokens_have_distinct_outputs": all(value > 0.0 for value in pairwise),
        "minimum_pairwise_token_output_difference": min(pairwise),
        "has_no_recurrent_state_attribute": not any(
            key in model.__dict__ for key in ("hidden", "_hidden", "_rollout_hidden")
        ),
    }


__all__ = [
    "ACTION_CONTRACT_ID",
    "ARCHITECTURE_ID",
    "BudgetMatchedMarkovianPolicy",
    "COMPUTE_CONTRACT",
    "ComputeContract",
    "HISTORY_REFERENCE_ARCHITECTURE_ID",
    "OUTCOME_ALPHABET_ID",
    "ObservedOutcome",
    "TRAINING_PROTOCOL_ID",
    "audit_latest_only_behavior",
    "build_budget_matched_policy",
]
