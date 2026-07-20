"""Quantize the selected T4.4.1 GRU checkpoint into unified hardware memories."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "docs/t4_4_1_bounded_residual_rnn_teacher_checkpoints.pt"
VALIDATION = ROOT / "docs/t4_4_1_bounded_residual_rnn_teacher_validation.json"
OUTPUT = ROOT / "cnn_fpga/rtl/generated"
WEIGHT_NAMES = (
    "gru.weight_ih", "gru.weight_hh", "dense1.weight",
    "dense2.weight", "output.weight",
)
BIAS_NAMES = (
    "gru.bias_ih", "gru.bias_hh", "dense1.bias", "dense2.bias", "output.bias",
)
BIAS_BITS = 18
BIAS_FRACTIONAL_BITS = 14


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_torch() -> Any:
    import torch
    return torch


def load_selected_state() -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _require_torch()
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    selected_index = int(payload["selected_restart_index"])
    selected = payload["restarts"][selected_index]
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if selected["checkpoint_sha256"] != validation["checkpoint"]["selected_state_sha256"]:
        raise ValueError("selected GRU state hash mismatch")
    return selected["state_dict"], {
        "selected_restart_index": selected_index,
        "training_seed": int(selected["training_seed"]),
        "selected_state_sha256": selected["checkpoint_sha256"],
    }


def weight_fractional_bits(maximum_absolute: float) -> int:
    if not math.isfinite(maximum_absolute) or maximum_absolute <= 0.0:
        raise ValueError("weight tensor must have finite nonzero range")
    return max(0, min(15, int(math.floor(math.log2(127.0 / maximum_absolute)))))


def _quantize_signed(values: Any, bits: int, fractional_bits: int) -> list[int]:
    torch = _require_torch()
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    flattened = values.detach().cpu().to(dtype=torch.float64).reshape(-1).tolist()
    return [min(max(int(round(float(value) * (1 << fractional_bits))), minimum), maximum) for value in flattened]


def _write_mem(path: Path, codes: list[int], bits: int) -> dict[str, Any]:
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    path.write_text("".join(f"{code & mask:0{digits}x}\n" for code in codes), encoding="ascii")
    return {"path": path.relative_to(ROOT).as_posix(), "entries": len(codes), "bits": bits, "sha256": _sha256(path)}


def generate() -> dict[str, Any]:
    state, provenance = load_selected_state()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    weight_codes: list[int] = []
    weight_tensors = []
    for name in WEIGHT_NAMES:
        tensor = state[name]
        maximum = float(tensor.detach().abs().max().item())
        fractional_bits = weight_fractional_bits(maximum)
        codes = _quantize_signed(tensor, 8, fractional_bits)
        offset = len(weight_codes)
        weight_codes.extend(codes)
        restored = [code / float(1 << fractional_bits) for code in codes]
        original = tensor.detach().cpu().to(dtype=_require_torch().float64).reshape(-1).tolist()
        weight_tensors.append({
            "name": name,
            "shape": list(tensor.shape),
            "offset": offset,
            "entries": len(codes),
            "fractional_bits": fractional_bits,
            "maximum_absolute_value": maximum,
            "maximum_absolute_quantization_error": max(abs(float(a) - b) for a, b in zip(original, restored)),
        })
    bias_codes: list[int] = []
    bias_tensors = []
    for name in BIAS_NAMES:
        tensor = state[name]
        codes = _quantize_signed(tensor, BIAS_BITS, BIAS_FRACTIONAL_BITS)
        offset = len(bias_codes)
        bias_codes.extend(codes)
        restored = [code / float(1 << BIAS_FRACTIONAL_BITS) for code in codes]
        original = tensor.detach().cpu().to(dtype=_require_torch().float64).reshape(-1).tolist()
        bias_tensors.append({
            "name": name,
            "shape": list(tensor.shape),
            "offset": offset,
            "entries": len(codes),
            "fractional_bits": BIAS_FRACTIONAL_BITS,
            "maximum_absolute_quantization_error": max(abs(float(a) - b) for a, b in zip(original, restored)),
        })
    if len(weight_codes) != 72_266 or len(bias_codes) != 587:
        raise ValueError("GRU parameter accounting mismatch")
    weight_file = _write_mem(OUTPUT / "t5_5_4_quantized_gru_weights.mem", weight_codes, 8)
    bias_file = _write_mem(OUTPUT / "t5_5_4_quantized_gru_biases.mem", bias_codes, BIAS_BITS)
    manifest = {
        "schema_version": "t5.5.4-selected-gru-power2-int8-v1",
        "checkpoint": {"path": CHECKPOINT.relative_to(ROOT).as_posix(), "sha256": _sha256(CHECKPOINT)},
        "validation": {"path": VALIDATION.relative_to(ROOT).as_posix(), "sha256": _sha256(VALIDATION)},
        **provenance,
        "architecture": "GRU10-DENSE256-DENSE256-OUT15",
        "weight_quantization": "per-tensor signed int8 symmetric power-of-two scale ties-to-even saturation",
        "bias_quantization": "signed Q3.14 ties-to-even saturation",
        "weight_tensors": weight_tensors,
        "bias_tensors": bias_tensors,
        "weight_file": weight_file,
        "bias_file": bias_file,
        "counts": {
            "weight_macs": len(weight_codes),
            "bias_scalars": len(bias_codes),
            "total_parameters": len(weight_codes) + len(bias_codes),
            "weight_storage_bits": len(weight_codes) * 8,
            "bias_storage_bits": len(bias_codes) * BIAS_BITS,
            "total_quantized_parameter_bits": len(weight_codes) * 8 + len(bias_codes) * BIAS_BITS,
        },
    }
    manifest_path = OUTPUT / "t5_5_4_quantized_gru_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(generate(), indent=2))


if __name__ == "__main__":
    main()
