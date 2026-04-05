"""AXI-Lite register map helpers for the CNN-FPGA HIL layer."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict

import numpy as np

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


@dataclass(frozen=True)
class FixedPointFormat:
    """Minimal fixed-point helper kept local to avoid hwio/decoder import cycles."""

    integer_bits: int = 4
    fractional_bits: int = 20

    @classmethod
    def from_spec(cls, spec: str) -> "FixedPointFormat":
        match = re.fullmatch(r"Q(\d+)\.(\d+)", spec.strip(), flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Unsupported fixed-point spec: {spec}")
        return cls(integer_bits=int(match.group(1)), fractional_bits=int(match.group(2)))

    @property
    def step(self) -> float:
        return 2.0 ** (-self.fractional_bits)

    @property
    def min_value(self) -> float:
        return -(2.0 ** self.integer_bits)

    @property
    def max_value(self) -> float:
        return (2.0 ** self.integer_bits) - self.step

    def quantize(self, value: np.ndarray | list[float] | float) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(value, dtype=float)
        clipped = np.clip(array, self.min_value, self.max_value)
        quantized = np.round(clipped / self.step) * self.step
        saturated = np.logical_or(array < self.min_value, array > self.max_value)
        return quantized.astype(float), saturated


def _signed_to_u32(value: int) -> int:
    return int(value & 0xFFFFFFFF)


def _u32_to_signed(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return value - (1 << 32)
    return value


@dataclass(frozen=True)
class AxiRegisterMap:
    """Register and fixed-point helpers shared by driver and backend."""

    fixed_point_spec: str = "Q4.20"
    ctrl_addr: int = 0x00
    status_addr: int = 0x04
    hist_meta_addr: int = 0x08
    overflow_count_addr: int = 0x0C
    k11_addr: int = 0x10
    k12_addr: int = 0x14
    k21_addr: int = 0x18
    k22_addr: int = 0x1C
    b1_addr: int = 0x20
    b2_addr: int = 0x24
    active_bank_addr: int = 0x30
    epoch_id_addr: int = 0x34
    commit_epoch_addr: int = 0x38
    hist_seq_addr: int = 0x3C

    ctrl_start_mask: int = 1 << 0
    ctrl_reset_hist_mask: int = 1 << 1
    ctrl_commit_bank_mask: int = 1 << 2

    status_ready_mask: int = 1 << 0
    status_hist_ready_mask: int = 1 << 1
    status_commit_ack_mask: int = 1 << 2
    status_overflow_alert_mask: int = 1 << 3

    @property
    def fixed_point(self) -> FixedPointFormat:
        return FixedPointFormat.from_spec(self.fixed_point_spec)

    @property
    def param_addrs(self) -> Dict[str, int]:
        return {
            "K11": self.k11_addr,
            "K12": self.k12_addr,
            "K21": self.k21_addr,
            "K22": self.k22_addr,
            "b1": self.b1_addr,
            "b2": self.b2_addr,
        }

    def pack_scalar(self, value: float) -> int:
        quantized, _ = self.fixed_point.quantize(float(value))
        scaled = int(np.round(float(quantized) / self.fixed_point.step))
        return _signed_to_u32(scaled)

    def unpack_scalar(self, raw_value: int) -> float:
        signed = _u32_to_signed(int(raw_value))
        return float(signed * self.fixed_point.step)

    def pack_params(self, params: DecoderRuntimeParams) -> Dict[int, int]:
        return {
            self.k11_addr: self.pack_scalar(params.K[0, 0]),
            self.k12_addr: self.pack_scalar(params.K[0, 1]),
            self.k21_addr: self.pack_scalar(params.K[1, 0]),
            self.k22_addr: self.pack_scalar(params.K[1, 1]),
            self.b1_addr: self.pack_scalar(params.b[0]),
            self.b2_addr: self.pack_scalar(params.b[1]),
        }

    def unpack_params(self, registers: Dict[int, int]) -> DecoderRuntimeParams:
        k = np.array(
            [
                [
                    self.unpack_scalar(registers.get(self.k11_addr, 0)),
                    self.unpack_scalar(registers.get(self.k12_addr, 0)),
                ],
                [
                    self.unpack_scalar(registers.get(self.k21_addr, 0)),
                    self.unpack_scalar(registers.get(self.k22_addr, 0)),
                ],
            ],
            dtype=float,
        )
        b = np.array(
            [
                self.unpack_scalar(registers.get(self.b1_addr, 0)),
                self.unpack_scalar(registers.get(self.b2_addr, 0)),
            ],
            dtype=float,
        )
        return DecoderRuntimeParams(K=k, b=b)

    def build_ctrl_word(
        self,
        *,
        start: bool = False,
        reset_hist: bool = False,
        commit_bank: bool = False,
    ) -> int:
        word = 0
        if start:
            word |= self.ctrl_start_mask
        if reset_hist:
            word |= self.ctrl_reset_hist_mask
        if commit_bank:
            word |= self.ctrl_commit_bank_mask
        return word

    def decode_ctrl_word(self, word: int) -> Dict[str, bool]:
        return {
            "start": bool(word & self.ctrl_start_mask),
            "reset_hist": bool(word & self.ctrl_reset_hist_mask),
            "commit_bank": bool(word & self.ctrl_commit_bank_mask),
        }

    def build_status_word(self, *, ready: bool, hist_ready: bool, commit_ack: bool, overflow_alert: bool = False) -> int:
        word = 0
        if ready:
            word |= self.status_ready_mask
        if hist_ready:
            word |= self.status_hist_ready_mask
        if commit_ack:
            word |= self.status_commit_ack_mask
        if overflow_alert:
            word |= self.status_overflow_alert_mask
        return word

    def decode_status_word(self, word: int) -> Dict[str, bool]:
        return {
            "ready": bool(word & self.status_ready_mask),
            "hist_ready": bool(word & self.status_hist_ready_mask),
            "commit_ack": bool(word & self.status_commit_ack_mask),
            "overflow_alert": bool(word & self.status_overflow_alert_mask),
        }

    def build_hist_meta_word(self, *, buffer_id: int, overflow_alert: bool = False) -> int:
        if buffer_id < 0 or buffer_id > 0xFFFF:
            raise ValueError(f"buffer_id out of range: {buffer_id}")
        word = int(buffer_id & 0xFFFF)
        if overflow_alert:
            word |= 1 << 16
        return word

    def decode_hist_meta_word(self, word: int) -> Dict[str, int | bool]:
        return {
            "buffer_id": int(word & 0xFFFF),
            "overflow_alert": bool(word & (1 << 16)),
        }

    def encode_bank(self, bank_name: str) -> int:
        normalized = str(bank_name).upper()
        if normalized == "A":
            return 0
        if normalized == "B":
            return 1
        raise ValueError(f"Unsupported bank name: {bank_name}")

    def decode_bank(self, raw_value: int) -> str:
        return "B" if int(raw_value) else "A"


AXI_REGISTER_MAP = AxiRegisterMap()
