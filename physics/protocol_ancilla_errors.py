"""sBs 与 sharpen--trim 的协议原生 ancilla/readout/reset error layer。

本模块刻意不共享两个协议的观测字母表：

* sBs 复用 :mod:`physics.sbs_observation_reset` 的 ``g/e/leakage`` hidden-
  observed-reset kernel，并叠加 stage-resolved bit/phase flips；
* sharpen--trim 使用原生 ``+y/-y`` readout、四轮 action、conditional ``pi/2``
  reset 和独立 hidden carry。

Steane/Knill/qunaught/P-Steane 只保留不可执行的 theory/noise-shaping registry。
任何 fault probability、confusion matrix、reset success 或 backaction scale 都必须带
显式 provenance；这里没有 device-calibrated 默认值。
"""

from __future__ import annotations

import argparse
import json

from ._protocol_ancilla.common import (
    SBS_CONSTITUENTS,
    SBS_FAULT_STAGES,
    SHARPEN_TRIM_CARRY_STATES,
    SHARPEN_TRIM_HIDDEN_STATES,
    SHARPEN_TRIM_OBSERVED_CLASSES,
    SHARPEN_TRIM_PROTOCOL_ID,
    SHARPEN_TRIM_ROUND_TYPES,
)
from ._protocol_ancilla.fault_overlay import (
    SBSAncillaFaultEvent,
    SBSAncillaFaultOverlay,
    SBSFaultOverlayConfig,
    SBSFaultOverlayStep,
    SBSFaultOverlayTrajectory,
    SBSFaultOverlayTruth,
)
from ._protocol_ancilla.sharpen_trim import (
    SharpenTrimAncillaConfig,
    SharpenTrimAncillaModel,
    SharpenTrimMemory,
    SharpenTrimObservedRound,
    SharpenTrimStep,
    SharpenTrimTrajectory,
    SharpenTrimTruthRound,
)
from ._protocol_ancilla.validation import (
    SECONDARY_PROTOCOL_NOISE_REGISTRY,
    ProtocolAncillaValidationResult,
    SecondaryProtocolNoiseSpec,
    _perfect_sbs_base_model,
    run_protocol_ancilla_validation,
    secondary_protocol_noise_specs,
    write_protocol_ancilla_validation,
)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=2026071422)
    arguments = parser.parse_args()
    result = run_protocol_ancilla_validation(
        samples=arguments.samples,
        seed=arguments.seed,
    )
    write_protocol_ancilla_validation(result, arguments.output)
    print(json.dumps(result.as_dict()["checks"], ensure_ascii=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "SBS_FAULT_STAGES",
    "SBS_CONSTITUENTS",
    "SHARPEN_TRIM_PROTOCOL_ID",
    "SHARPEN_TRIM_ROUND_TYPES",
    "SHARPEN_TRIM_HIDDEN_STATES",
    "SHARPEN_TRIM_OBSERVED_CLASSES",
    "SBSFaultOverlayConfig",
    "SBSAncillaFaultEvent",
    "SBSFaultOverlayTruth",
    "SBSFaultOverlayStep",
    "SBSFaultOverlayTrajectory",
    "SBSAncillaFaultOverlay",
    "SharpenTrimAncillaConfig",
    "SharpenTrimMemory",
    "SharpenTrimObservedRound",
    "SharpenTrimTruthRound",
    "SharpenTrimStep",
    "SharpenTrimTrajectory",
    "SharpenTrimAncillaModel",
    "SecondaryProtocolNoiseSpec",
    "SECONDARY_PROTOCOL_NOISE_REGISTRY",
    "secondary_protocol_noise_specs",
    "ProtocolAncillaValidationResult",
    "run_protocol_ancilla_validation",
    "write_protocol_ancilla_validation",
]
