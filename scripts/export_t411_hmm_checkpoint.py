"""Export the selected T4.1.1 Gaussian-HMM from the torch checkpoint to JSON.

The main Route-A runner deliberately has no PyTorch dependency.  This helper is
the provenance-preserving conversion boundary: it must be run in an
environment that can read the original ``.pt`` file, and the emitted JSON keeps
the source checkpoint SHA-256 plus the exact selected model payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "t6.6.1-exported-t4.1.1-gaussian-hmm-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility.
        checkpoint = torch.load(source, map_location="cpu")
    if checkpoint.get("schema_version") != "t4.1.1-slow-loop-model-selection-checkpoints-v1":
        raise ValueError("unexpected T4.1.1 checkpoint schema")
    if checkpoint.get("selected_family_from_validation") != "gaussian_hmm":
        raise ValueError("T4.1.1 validation did not select gaussian_hmm")
    models = checkpoint.get("models")
    if not isinstance(models, dict) or "gaussian_hmm" not in models:
        raise ValueError("T4.1.1 checkpoint has no gaussian_hmm payload")
    selected = models["gaussian_hmm"]
    if selected.get("family") != "gaussian_hmm":
        raise ValueError("selected payload family mismatch")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": source.as_posix(),
        "source_checkpoint_sha256": _sha256(source),
        "source_schema_version": checkpoint["schema_version"],
        "source_implementation_sha256": checkpoint["implementation_sha256"],
        "selected_family_from_validation": checkpoint["selected_family_from_validation"],
        "temperature": float(selected["temperature"]),
        "model": selected["model"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("docs/t4_1_1_slow_loop_model_selection_checkpoints.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/route_a/t4_1_1_gaussian_hmm.json"),
    )
    args = parser.parse_args()
    payload = export_checkpoint(args.source, args.output)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
                "selected_family": payload["selected_family_from_validation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
