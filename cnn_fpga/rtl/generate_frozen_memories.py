"""Generate exact signed-22 RTL memories from the frozen T5.5.1 images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cnn_fpga.benchmark.bit_accurate_hardware_reference import load_frozen_images


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "cnn_fpga" / "rtl" / "generated"


def _hex22(value: int) -> str:
    if not -(1 << 21) <= value < (1 << 21):
        raise ValueError("table value is outside signed 22-bit range")
    return f"{value & ((1 << 22) - 1):06x}"


def generate(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    images = load_frozen_images()
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for bank in (0, 1):
        for phase, label in enumerate(("x", "z")):
            path = output_dir / f"t5_5_1_bank{bank}_{label}.mem"
            payload = "\n".join(_hex22(code) for code in images[bank].table_codes[phase]) + "\n"
            path.write_text(payload, encoding="ascii", newline="\n")
            files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "entries": 257,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "image_version": bank,
                    "phase": phase,
                }
            )
    manifest = {
        "schema_version": "t-risk-20260716-01-frozen-rtl-memory-v1",
        "source": "T5.5.1 frozen images v0/v1",
        "word_bits": 22,
        "encoding": "six lowercase hex digits, two's complement, one word per line",
        "files": files,
    }
    manifest_path = output_dir / "t5_5_1_memory_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = generate(args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
