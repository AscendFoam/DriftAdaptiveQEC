"""Deterministic Torch policy checkpoint helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


def state_dict_cpu(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()
