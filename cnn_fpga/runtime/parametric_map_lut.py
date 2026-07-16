"""Integer-only fast-path runtime for a version-bound parametric MAP LUT."""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, replace
from typing import Any


SCHEMA_VERSION = "t4.2.1-parametric-map-lut-image-v1"
ONLINE_SCOPE = "integer_code_phase_and_latched_bank_only"
PHASE_LABELS = ("X", "Z")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


@dataclass(frozen=True)
class ParametricMAPLUTConfig:
    adc_bits: int = 10
    address_bits: int = 8
    llr_integer_bits: int = 9
    llr_fractional_bits: int = 12
    lattice: float = math.sqrt(2.0 * math.pi)
    pipeline_latency_cycles: int = 5
    initiation_interval_cycles: int = 1

    def __post_init__(self) -> None:
        for name in (
            "adc_bits",
            "address_bits",
            "llr_integer_bits",
            "llr_fractional_bits",
            "pipeline_latency_cycles",
            "initiation_interval_cycles",
        ):
            value = _integer(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if not 4 <= self.adc_bits <= 20:
            raise ValueError("adc_bits must lie in [4,20]")
        if not 2 <= self.address_bits <= self.adc_bits:
            raise ValueError("address_bits must lie in [2,adc_bits]")
        if self.llr_word_bits > 31:
            raise ValueError("llr word width above 31 bits is unsupported")
        lattice = float(self.lattice)
        if not math.isfinite(lattice) or lattice <= 0.0:
            raise ValueError("lattice must be finite and positive")
        object.__setattr__(self, "lattice", lattice)
        if self.pipeline_latency_cycles != 5:
            raise ValueError("T4.2.1 pipeline contract requires exactly five stages")
        if self.initiation_interval_cycles != 1:
            raise ValueError("T4.2.1 pipeline contract requires initiation interval one")

    @property
    def adc_levels(self) -> int:
        return 1 << self.adc_bits

    @property
    def table_intervals(self) -> int:
        return 1 << self.address_bits

    @property
    def table_entries(self) -> int:
        return self.table_intervals + 1

    @property
    def fraction_bits(self) -> int:
        return self.adc_bits - self.address_bits

    @property
    def llr_word_bits(self) -> int:
        return 1 + self.llr_integer_bits + self.llr_fractional_bits

    @property
    def llr_min_code(self) -> int:
        return -(1 << (self.llr_word_bits - 1))

    @property
    def llr_max_code(self) -> int:
        return (1 << (self.llr_word_bits - 1)) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "adc_bits": self.adc_bits,
            "address_bits": self.address_bits,
            "fraction_bits": self.fraction_bits,
            "llr_integer_bits": self.llr_integer_bits,
            "llr_fractional_bits": self.llr_fractional_bits,
            "llr_word_bits": self.llr_word_bits,
            "lattice": self.lattice,
            "pipeline_latency_cycles": self.pipeline_latency_cycles,
            "initiation_interval_cycles": self.initiation_interval_cycles,
        }


@dataclass(frozen=True)
class ParametricMAPLUTImage:
    config: ParametricMAPLUTConfig
    active_bank_version: int
    source_params_sha256: str
    model_mean: tuple[float, float]
    model_sigma: tuple[float, float]
    table_codes: tuple[tuple[int, ...], tuple[int, ...]]
    llr_saturation_count: int
    image_crc32: str = ""
    image_sha256: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.config, ParametricMAPLUTConfig):
            raise TypeError("config must be ParametricMAPLUTConfig")
        version = _integer(self.active_bank_version, "active_bank_version")
        if version < 0:
            raise ValueError("active_bank_version must be non-negative")
        object.__setattr__(self, "active_bank_version", version)
        if len(self.source_params_sha256) != 64:
            raise ValueError("source_params_sha256 must be a SHA-256 hex digest")
        try:
            int(self.source_params_sha256, 16)
        except ValueError as exc:
            raise ValueError("source_params_sha256 must be hexadecimal") from exc
        if len(self.model_mean) != 2 or len(self.model_sigma) != 2:
            raise ValueError("model_mean/model_sigma must each contain two values")
        if not all(math.isfinite(float(value)) for value in self.model_mean):
            raise ValueError("model_mean must be finite")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in self.model_sigma
        ):
            raise ValueError("model_sigma must be finite and positive")
        if len(self.table_codes) != 2:
            raise ValueError("table_codes must contain X and Z phase tables")
        checked_tables: list[tuple[int, ...]] = []
        for phase, table in enumerate(self.table_codes):
            if len(table) != self.config.table_entries:
                raise ValueError(
                    f"phase {phase} table must contain {self.config.table_entries} entries"
                )
            checked = tuple(_integer(value, f"table_codes[{phase}]") for value in table)
            if (
                min(checked) < self.config.llr_min_code
                or max(checked) > self.config.llr_max_code
            ):
                raise ValueError("table code lies outside configured LLR word width")
            checked_tables.append(checked)
        object.__setattr__(self, "table_codes", (checked_tables[0], checked_tables[1]))
        count = _integer(self.llr_saturation_count, "llr_saturation_count")
        if count < 0:
            raise ValueError("llr_saturation_count must be non-negative")
        object.__setattr__(self, "llr_saturation_count", count)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "online_scope": ONLINE_SCOPE,
            "config": self.config.to_dict(),
            "active_bank_version": self.active_bank_version,
            "source_params_sha256": self.source_params_sha256,
            "model_mean": list(self.model_mean),
            "model_sigma": list(self.model_sigma),
            "table_codes": [list(table) for table in self.table_codes],
            "llr_saturation_count": self.llr_saturation_count,
        }

    @classmethod
    def create(
        cls,
        *,
        config: ParametricMAPLUTConfig,
        active_bank_version: int,
        source_params_sha256: str,
        model_mean: tuple[float, float],
        model_sigma: tuple[float, float],
        table_codes: tuple[tuple[int, ...], tuple[int, ...]],
        llr_saturation_count: int,
    ) -> "ParametricMAPLUTImage":
        provisional = cls(
            config=config,
            active_bank_version=active_bank_version,
            source_params_sha256=source_params_sha256,
            model_mean=model_mean,
            model_sigma=model_sigma,
            table_codes=table_codes,
            llr_saturation_count=llr_saturation_count,
        )
        payload = _canonical_bytes(provisional.unsigned_payload())
        return replace(
            provisional,
            image_crc32=f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
            image_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def verify(self) -> None:
        payload = _canonical_bytes(self.unsigned_payload())
        expected_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
        expected_sha = hashlib.sha256(payload).hexdigest()
        if self.image_crc32 != expected_crc:
            raise ValueError("parametric MAP-LUT image CRC mismatch")
        if self.image_sha256 != expected_sha:
            raise ValueError("parametric MAP-LUT image SHA-256 mismatch")

    def to_dict(self, *, include_tables: bool = True) -> dict[str, Any]:
        payload = self.unsigned_payload()
        if not include_tables:
            payload.pop("table_codes")
            payload["table_shape"] = [2, self.config.table_entries]
        payload.update(
            {"image_crc32": self.image_crc32, "image_sha256": self.image_sha256}
        )
        return payload


@dataclass(frozen=True)
class ParametricMAPLUTInput:
    cycle_index: int
    syndrome_code: int
    quadrature_phase_bit: int
    active_bank_version: int

    def __post_init__(self) -> None:
        cycle = _integer(self.cycle_index, "cycle_index")
        code = _integer(self.syndrome_code, "syndrome_code")
        phase = _integer(self.quadrature_phase_bit, "quadrature_phase_bit")
        version = _integer(self.active_bank_version, "active_bank_version")
        if cycle < 0 or version < 0:
            raise ValueError("cycle_index and active_bank_version must be non-negative")
        if phase not in (0, 1):
            raise ValueError("quadrature_phase_bit must be 0 (X) or 1 (Z)")
        object.__setattr__(self, "cycle_index", cycle)
        object.__setattr__(self, "syndrome_code", code)
        object.__setattr__(self, "quadrature_phase_bit", phase)
        object.__setattr__(self, "active_bank_version", version)


@dataclass(frozen=True)
class ParametricMAPLUTDecision:
    input_cycle: int
    valid_cycle: int
    syndrome_code: int
    quadrature_phase_bit: int
    phase_label: str
    address: int
    fraction_code: int
    llr_code: int
    logical_flip: bool
    logical_action: str
    active_bank_version: int
    image_sha256: str


def software_encode_syndrome_for_replay(
    syndrome: float, config: ParametricMAPLUTConfig
) -> int:
    value = float(syndrome)
    if not math.isfinite(value):
        raise ValueError("syndrome must be finite")
    half = 0.5 * config.lattice
    if value < -half or value >= half:
        raise ValueError("syndrome must lie in the half-open decoder cell")
    normalized = (value + half) / config.lattice
    return min(config.adc_levels - 1, int(math.floor(normalized * config.adc_levels)))


def software_decode_syndrome_code(
    syndrome_code: int, config: ParametricMAPLUTConfig
) -> float:
    code = _integer(syndrome_code, "syndrome_code")
    if code < 0 or code >= config.adc_levels:
        raise ValueError("syndrome_code lies outside configured ADC width")
    return -0.5 * config.lattice + (code + 0.5) * config.lattice / config.adc_levels


def _rounded_signed_shift(value: int, bits: int) -> int:
    """Round signed integer / 2**bits to nearest, ties to even, without division."""

    if bits == 0:
        return value
    magnitude = abs(value)
    quotient = magnitude >> bits
    remainder = magnitude & ((1 << bits) - 1)
    halfway = 1 << (bits - 1)
    if remainder > halfway or (remainder == halfway and (quotient & 1) == 1):
        quotient += 1
    return -quotient if value < 0 else quotient


class ParametricMAPLUTRuntime:
    """Loaded-image integer execution kernel; integrity is checked on load."""

    def __init__(self, image: ParametricMAPLUTImage) -> None:
        self.load_image(image)

    @property
    def image(self) -> ParametricMAPLUTImage:
        return self._image

    def load_image(self, image: ParametricMAPLUTImage) -> None:
        if not isinstance(image, ParametricMAPLUTImage):
            raise TypeError("image must be ParametricMAPLUTImage")
        image.verify()
        self._image = image

    def decode_code(self, request: ParametricMAPLUTInput) -> ParametricMAPLUTDecision:
        if not isinstance(request, ParametricMAPLUTInput):
            raise TypeError("request must be ParametricMAPLUTInput")
        config = self._image.config
        if request.syndrome_code < 0 or request.syndrome_code >= config.adc_levels:
            raise ValueError("syndrome_code lies outside configured ADC width")
        if request.active_bank_version != self._image.active_bank_version:
            raise ValueError("request/image active-bank version mismatch")

        fraction_bits = config.fraction_bits
        address = request.syndrome_code >> fraction_bits
        fraction_mask = (1 << fraction_bits) - 1
        fraction = request.syndrome_code & fraction_mask
        table = self._image.table_codes[request.quadrature_phase_bit]
        y0 = table[address]
        y1 = table[address + 1]
        # ADC code ``c`` represents the centre of its quantization bin.  The
        # odd numerator below is therefore (low_bits + 1/2), expressed exactly
        # over 2**(fraction_bits+1); omitting the half-bin creates a systematic
        # address bias even when address_bits == adc_bits.
        fraction_numerator = (fraction << 1) + 1
        llr_code = y0 + _rounded_signed_shift(
            (y1 - y0) * fraction_numerator, fraction_bits + 1
        )
        llr_code = max(config.llr_min_code, min(config.llr_max_code, llr_code))
        logical_flip = llr_code < 0
        phase_label = PHASE_LABELS[request.quadrature_phase_bit]
        return ParametricMAPLUTDecision(
            input_cycle=request.cycle_index,
            valid_cycle=request.cycle_index + config.pipeline_latency_cycles,
            syndrome_code=request.syndrome_code,
            quadrature_phase_bit=request.quadrature_phase_bit,
            phase_label=phase_label,
            address=address,
            fraction_code=fraction,
            llr_code=llr_code,
            logical_flip=logical_flip,
            logical_action=phase_label if logical_flip else "I",
            active_bank_version=request.active_bank_version,
            image_sha256=self._image.image_sha256,
        )


class ParametricMAPLUTPipeline:
    """Observable five-cycle pipeline with II=1 and per-request image latching."""

    def __init__(self, image: ParametricMAPLUTImage) -> None:
        self._runtime = ParametricMAPLUTRuntime(image)
        self._last_cycle = -1
        self._pending: dict[int, ParametricMAPLUTDecision] = {}

    @property
    def loaded_image(self) -> ParametricMAPLUTImage:
        return self._runtime.image

    def load_image(self, image: ParametricMAPLUTImage) -> None:
        self._runtime.load_image(image)

    def step(
        self,
        cycle_index: int,
        request: ParametricMAPLUTInput | None = None,
    ) -> ParametricMAPLUTDecision | None:
        cycle = _integer(cycle_index, "cycle_index")
        if cycle != self._last_cycle + 1:
            raise ValueError("pipeline cycle_index must advance by exactly one")
        if request is not None:
            if request.cycle_index != cycle:
                raise ValueError("request.cycle_index must equal current pipeline cycle")
            decision = self._runtime.decode_code(request)
            if decision.valid_cycle in self._pending:
                raise RuntimeError("pipeline initiation interval was violated")
            self._pending[decision.valid_cycle] = decision
        self._last_cycle = cycle
        return self._pending.pop(cycle, None)


def resource_contract(config: ParametricMAPLUTConfig) -> dict[str, Any]:
    table_bits_single_bank = 2 * config.table_entries * config.llr_word_bits
    return {
        "identity": "exact_image_and_pipeline_contract_not_synthesis",
        "phase_tables": 2,
        "entries_per_phase_with_guard": config.table_entries,
        "llr_word_bits": config.llr_word_bits,
        "single_bank_table_bits": table_bits_single_bank,
        "dual_bank_table_bits": 2 * table_bits_single_bank,
        "pipeline_stages": (
            "S0_latch_validate_version",
            "S1_address_phase_select",
            "S2_dual_rom_read",
            "S3_integer_linear_interpolation",
            "S4_sign_action_register",
        ),
        "worst_case_latency_cycles": config.pipeline_latency_cycles,
        "initiation_interval_cycles": config.initiation_interval_cycles,
        "runtime_dividers": 0,
        "runtime_exp_log_units": 0,
        "target_lut_count": None,
        "target_ff_count": None,
        "target_bram_count": None,
        "target_dsp_count": None,
        "fmax_mhz": None,
        "rtl_measured": False,
        "board_measured": False,
    }


__all__ = [
    "ONLINE_SCOPE",
    "PHASE_LABELS",
    "SCHEMA_VERSION",
    "ParametricMAPLUTConfig",
    "ParametricMAPLUTDecision",
    "ParametricMAPLUTImage",
    "ParametricMAPLUTInput",
    "ParametricMAPLUTPipeline",
    "ParametricMAPLUTRuntime",
    "resource_contract",
    "software_decode_syndrome_code",
    "software_encode_syndrome_for_replay",
]
