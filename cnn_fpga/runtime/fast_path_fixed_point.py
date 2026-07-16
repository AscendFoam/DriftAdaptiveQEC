"""T4.2.4 bit-accurate composition of MAP, health, event, and frame paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from cnn_fpga.runtime.conservative_fallback import (
    ConservativeFallbackAction,
    ConservativeFallbackConfig,
    ConservativeFallbackController,
    ConservativeFallbackInput,
    TrustedParameterImage,
)
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTDecision,
    ParametricMAPLUTImage,
    ParametricMAPLUTInput,
    ParametricMAPLUTRuntime,
)


MODEL_SCOPE = "bit_accurate_integer_fast_path_software_reference_not_rtl_or_board"


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def encode_syndrome_replay(
    syndrome: float,
    image: ParametricMAPLUTImage,
) -> tuple[int, bool, bool]:
    """Offline float replay boundary; online execution starts from the returned code."""

    value = float(syndrome)
    config = image.config
    if not math.isfinite(value):
        return 0, True, False
    half = 0.5 * config.lattice
    valid = -half <= value < half
    clipped = min(max(value, -half), math.nextafter(half, -half))
    normalized = (clipped + half) / config.lattice
    code = min(config.adc_levels - 1, int(math.floor(normalized * config.adc_levels)))
    return code, not valid, valid


def encode_unit_interval_replay(value: float, bits: int) -> tuple[int, bool]:
    """Round-to-nearest-even unit-interval replay encoder with explicit saturation."""

    width = _integer(bits, "bits", 2)
    if width > 16:
        raise ValueError("bits must not exceed 16")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("unit-interval value must be finite")
    saturated = score < 0.0 or score > 1.0
    clipped = min(max(score, 0.0), 1.0)
    maximum = (1 << width) - 1
    return int(round(clipped * maximum)), saturated


def encode_unsigned_age_replay(value: int, bits: int) -> tuple[int, bool]:
    width = _integer(bits, "bits", 2)
    if width > 32:
        raise ValueError("bits must not exceed 32")
    age = _integer(value, "value")
    maximum = (1 << width) - 1
    return min(age, maximum), age > maximum


@dataclass(frozen=True)
class FastPathFixedPointContract:
    adc_bits: int = 10
    address_bits: int = 8
    interpolation_fraction_bits: int = 2
    llr_integer_bits: int = 9
    llr_fractional_bits: int = 12
    event_counter_bits: int = 3
    event_mode_bits: int = 3
    pauli_frame_bits: int = 2
    phase_frame_bits_each: int = 8
    ood_score_bits: int = 8
    parameter_age_bits: int = 16
    active_version_bits: int = 16
    fault_mask_bits: int = 14
    health_counter_bits_each: int = 8
    image_crc_bits: int = 32
    image_sha_bits: int = 256
    map_pipeline_cycles: int = 5
    health_event_register_cycles: int = 1
    initiation_interval_cycles: int = 1
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        for name in (
            "adc_bits",
            "address_bits",
            "interpolation_fraction_bits",
            "llr_integer_bits",
            "llr_fractional_bits",
            "event_counter_bits",
            "event_mode_bits",
            "pauli_frame_bits",
            "phase_frame_bits_each",
            "ood_score_bits",
            "parameter_age_bits",
            "active_version_bits",
            "fault_mask_bits",
            "health_counter_bits_each",
            "image_crc_bits",
            "image_sha_bits",
            "map_pipeline_cycles",
            "health_event_register_cycles",
            "initiation_interval_cycles",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if self.interpolation_fraction_bits != self.adc_bits - self.address_bits:
            raise ValueError("interpolation_fraction_bits must equal adc_bits-address_bits")
        if self.event_mode_bits != 3 or self.pauli_frame_bits != 2:
            raise ValueError("six event modes and two Pauli-frame bits are frozen")
        if self.fault_mask_bits != 14:
            raise ValueError("T4.2.3 fault mask is exactly 14 bits")
        if self.image_crc_bits != 32 or self.image_sha_bits != 256:
            raise ValueError("image integrity widths must remain CRC32/SHA256")
        if self.map_pipeline_cycles != 5 or self.health_event_register_cycles != 1:
            raise ValueError("registered source-to-action latency is 5+1 cycles")
        if self.initiation_interval_cycles != 1:
            raise ValueError("initiation interval must be one")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")

    @property
    def llr_word_bits(self) -> int:
        return 1 + self.llr_integer_bits + self.llr_fractional_bits

    @property
    def source_to_action_cycles(self) -> int:
        return self.map_pipeline_cycles + self.health_event_register_cycles

    def representation_proxy(self, *, registered_banks: int = 8) -> dict[str, int]:
        banks = _integer(registered_banks, "registered_banks", 1)
        entries = (1 << self.address_bits) + 1
        rom_bits_per_bank = 2 * entries * self.llr_word_bits
        live_event_bits = (
            self.event_mode_bits
            + 6 * self.event_counter_bits
            + self.pauli_frame_bits
            + 2 * self.phase_frame_bits_each
            + self.active_version_bits
        )
        live_health_bits = (
            self.ood_score_bits
            + self.parameter_age_bits
            + self.fault_mask_bits
            + (4 + self.fault_mask_bits) * self.health_counter_bits_each
        )
        return {
            "rom_entries_per_phase": entries,
            "rom_bits_per_bank": rom_bits_per_bank,
            "double_bank_rom_bits": 2 * rom_bits_per_bank,
            "registered_eight_bank_artifact_bits": banks * rom_bits_per_bank,
            "live_event_state_bits": live_event_bits,
            "live_health_state_and_input_bits": live_health_bits,
            "integrity_metadata_bits_per_image": self.image_crc_bits + self.image_sha_bits,
        }


@dataclass(frozen=True)
class FastPathCodeInput:
    cycle_index: int
    syndrome_code: int
    syndrome_x: str
    syndrome_z: str
    quadrature_phase_bit: int
    expected_active_bank_version: int
    reported_image_crc32: str
    reported_image_sha256: str
    parameter_age_code: int
    ood_score_code: int
    reset_ack: bool = False
    observation_valid: bool = True
    input_crc_ok: bool = True
    deadline_ok: bool = True

    def __post_init__(self) -> None:
        for name in (
            "cycle_index",
            "syndrome_code",
            "quadrature_phase_bit",
            "expected_active_bank_version",
            "parameter_age_code",
            "ood_score_code",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.quadrature_phase_bit not in (0, 1):
            raise ValueError("quadrature_phase_bit must be 0 or 1")
        for name in ("syndrome_x", "syndrome_z"):
            if getattr(self, name) not in ("g", "e", "leakage"):
                raise ValueError(f"{name} must be g, e, or leakage")
        for name in ("reset_ack", "observation_valid", "input_crc_ok", "deadline_ok"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))


@dataclass(frozen=True)
class FastPathBitAccurateResult:
    code_input: FastPathCodeInput
    map_decision: ParametricMAPLUTDecision | None
    fallback_action: ConservativeFallbackAction
    model_scope: str = MODEL_SCOPE


class BitAccurateFastPath:
    """Integer/boolean online composition; no float syndrome or OOD input is accepted."""

    def __init__(
        self,
        images: Sequence[ParametricMAPLUTImage],
        *,
        parameter_age_bits: int = 16,
        max_parameter_age_cycles: int = 64,
    ) -> None:
        registered = tuple(images)
        if not registered or not all(isinstance(image, ParametricMAPLUTImage) for image in registered):
            raise ValueError("images must contain ParametricMAPLUTImage records")
        versions = [image.active_bank_version for image in registered]
        if len(set(versions)) != len(versions):
            raise ValueError("image versions must be unique")
        for image in registered:
            image.verify()
        first = registered[0].config
        if any(image.config != first for image in registered):
            raise ValueError("all registered images must share one fixed-point config")
        self.contract = FastPathFixedPointContract(
            adc_bits=first.adc_bits,
            address_bits=first.address_bits,
            interpolation_fraction_bits=first.fraction_bits,
            llr_integer_bits=first.llr_integer_bits,
            llr_fractional_bits=first.llr_fractional_bits,
            parameter_age_bits=parameter_age_bits,
        )
        if max(versions) >= 1 << self.contract.active_version_bits:
            raise ValueError("image version exceeds active-version word width")
        maximum_age = _integer(
            max_parameter_age_cycles, "max_parameter_age_cycles", 1
        )
        if maximum_age >= 1 << self.contract.parameter_age_bits:
            raise ValueError("max_parameter_age_cycles exceeds parameter-age word width")
        self._images = {image.active_bank_version: image for image in registered}
        self._runtimes = {
            version: ParametricMAPLUTRuntime(image) for version, image in self._images.items()
        }
        trusted = tuple(
            TrustedParameterImage(version, image.image_crc32, image.image_sha256)
            for version, image in sorted(self._images.items())
        )
        self._controller = ConservativeFallbackController(
            trusted,
            ConservativeFallbackConfig(
                ood_score_bits=self.contract.ood_score_bits,
                health_counter_bits=self.contract.health_counter_bits_each,
                initial_active_bank_version=min(versions),
                max_parameter_age_cycles=maximum_age,
            ),
        )
        self._history: list[FastPathBitAccurateResult] = []

    @property
    def state(self):
        return self._controller.state

    @property
    def history(self) -> tuple[FastPathBitAccurateResult, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._controller.reset()
        self._history: list[FastPathBitAccurateResult] = []

    def step_codes(self, event: FastPathCodeInput) -> FastPathBitAccurateResult:
        if not isinstance(event, FastPathCodeInput):
            raise TypeError("event must be FastPathCodeInput")
        if event.syndrome_code >= 1 << self.contract.adc_bits:
            raise ValueError("syndrome_code exceeds ADC word width")
        if event.ood_score_code >= 1 << self.contract.ood_score_bits:
            raise ValueError("ood_score_code exceeds OOD word width")
        if event.parameter_age_code >= 1 << self.contract.parameter_age_bits:
            raise ValueError("parameter_age_code exceeds age word width")
        if event.expected_active_bank_version >= 1 << self.contract.active_version_bits:
            raise ValueError("expected_active_bank_version exceeds version word width")
        image = self._images.get(event.expected_active_bank_version)
        decision = None
        if image is not None and event.observation_valid:
            decision = self._runtimes[event.expected_active_bank_version].decode_code(
                ParametricMAPLUTInput(
                    event.cycle_index - self.contract.map_pipeline_cycles,
                    event.syndrome_code,
                    event.quadrature_phase_bit,
                    event.expected_active_bank_version,
                )
            )
        fallback = self._controller.step(
            ConservativeFallbackInput(
                cycle_index=event.cycle_index,
                syndrome_x=event.syndrome_x,
                syndrome_z=event.syndrome_z,
                quadrature_phase_bit=event.quadrature_phase_bit,
                map_decision=decision,
                expected_active_bank_version=event.expected_active_bank_version,
                reported_image_crc32=event.reported_image_crc32,
                reported_image_sha256=event.reported_image_sha256,
                parameter_age_cycles=event.parameter_age_code,
                ood_score_code=event.ood_score_code,
                reset_ack=event.reset_ack,
                observation_valid=event.observation_valid,
                input_crc_ok=event.input_crc_ok,
                deadline_ok=event.deadline_ok,
            )
        )
        result = FastPathBitAccurateResult(event, decision, fallback)
        self._history.append(result)
        return result


def build_code_input_from_replay(
    *,
    cycle_index: int,
    syndrome: float,
    syndrome_x: str,
    syndrome_z: str,
    quadrature_phase_bit: int,
    image: ParametricMAPLUTImage,
    parameter_age_cycles: int,
    ood_score: float,
    reset_ack: bool = False,
    observation_valid: bool = True,
    input_crc_ok: bool = True,
    deadline_ok: bool = True,
) -> tuple[FastPathCodeInput, dict[str, bool]]:
    """Offline-only float replay adapter with explicit saturation provenance."""

    syndrome_code, syndrome_saturated, syndrome_finite_cell = encode_syndrome_replay(
        syndrome, image
    )
    ood_code, ood_saturated = encode_unit_interval_replay(ood_score, 8)
    age_code, age_saturated = encode_unsigned_age_replay(parameter_age_cycles, 16)
    code_input = FastPathCodeInput(
        cycle_index=cycle_index,
        syndrome_code=syndrome_code,
        syndrome_x=syndrome_x,
        syndrome_z=syndrome_z,
        quadrature_phase_bit=quadrature_phase_bit,
        expected_active_bank_version=image.active_bank_version,
        reported_image_crc32=image.image_crc32,
        reported_image_sha256=image.image_sha256,
        parameter_age_code=age_code,
        ood_score_code=ood_code,
        reset_ack=reset_ack,
        observation_valid=observation_valid and syndrome_finite_cell,
        input_crc_ok=input_crc_ok,
        deadline_ok=deadline_ok,
    )
    return code_input, {
        "syndrome_saturated": syndrome_saturated,
        "ood_saturated": ood_saturated,
        "age_saturated": age_saturated,
    }


__all__ = [
    "MODEL_SCOPE",
    "BitAccurateFastPath",
    "FastPathBitAccurateResult",
    "FastPathCodeInput",
    "FastPathFixedPointContract",
    "build_code_input_from_replay",
    "encode_syndrome_replay",
    "encode_unit_interval_replay",
    "encode_unsigned_age_replay",
]
