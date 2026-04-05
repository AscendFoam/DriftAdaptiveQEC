"""Subprocess inference worker for slow-loop model serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from cnn_fpga.runtime.inference_service import ArtifactHistogramPredictor, InferenceServiceError, TFLiteHistogramPredictor


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run histogram inference worker.")
    parser.add_argument("--backend", required=True, type=str, choices=["artifact_npz", "tflite"])
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--label-names", default="sigma,mu_q,mu_p,theta_deg", type=str)
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    label_names = [item.strip() for item in args.label_names.split(",") if item.strip()]
    try:
        model_path = Path(args.model_path).expanduser().resolve()
        if args.backend == "artifact_npz":
            predictor = ArtifactHistogramPredictor(model_path)
        else:
            predictor = TFLiteHistogramPredictor(model_path, label_names)
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": f"worker_init_failed:{exc}"}), flush=True)
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            histogram = np.asarray(payload["histogram"], dtype=np.float32)
            prediction = predictor.predict(histogram)
            print(json.dumps(prediction.to_dict(), ensure_ascii=False), flush=True)
        except InferenceServiceError as exc:
            print(json.dumps({"error": exc.reason}, ensure_ascii=False), flush=True)
        except Exception as exc:  # pragma: no cover
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
