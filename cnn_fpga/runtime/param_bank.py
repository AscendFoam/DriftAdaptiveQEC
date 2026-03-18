"""Double-buffered parameter bank for glitch-free decoder updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


def _coerce_array(value: np.ndarray | list[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass
class DecoderRuntimeParams:
    """Runtime decoder parameters stored inside the fast-path bank."""

    K: np.ndarray
    b: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.K = _coerce_array(self.K, (2, 2), "K")
        self.b = _coerce_array(self.b, (2,), "b")
        self.metadata = dict(self.metadata)

    @classmethod
    def identity(cls) -> "DecoderRuntimeParams":
        return cls(K=np.eye(2, dtype=float), b=np.zeros(2, dtype=float))

    def copy(self) -> "DecoderRuntimeParams":
        return DecoderRuntimeParams(K=self.K.copy(), b=self.b.copy(), metadata=dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "K": self.K.tolist(),
            "b": self.b.tolist(),
            "metadata": dict(self.metadata),
        }


@dataclass
class PendingCommit:
    """A staged update waiting for the next cycle-boundary commit."""

    target_bank: str
    commit_epoch: int
    staged_epoch: int
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommitResult:
    """Result of a completed bank switch."""

    activated_bank: str
    commit_epoch: int
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParamBank:
    """Two-bank parameter storage with explicit stage and commit phases."""

    def __init__(
        self,
        initial_params: Optional[DecoderRuntimeParams] = None,
        initial_epoch: int = 0,
    ) -> None:
        if initial_epoch < 0:
            raise ValueError("initial_epoch must be non-negative")

        params = (initial_params or DecoderRuntimeParams.identity()).copy()
        self._banks: Dict[str, DecoderRuntimeParams] = {
            "A": params.copy(),
            "B": params.copy(),
        }
        self._versions: Dict[str, int] = {"A": 0, "B": 0}
        self._active_bank = "A"
        self._staging_bank = "B"
        self._pending_commit: Optional[PendingCommit] = None
        self._epoch_id = initial_epoch
        self._last_commit_epoch = initial_epoch
        self._next_version = 1

    @property
    def epoch_id(self) -> int:
        return self._epoch_id

    @property
    def active_bank_name(self) -> str:
        return self._active_bank

    @property
    def staging_bank_name(self) -> str:
        return self._staging_bank

    @property
    def has_pending_commit(self) -> bool:
        return self._pending_commit is not None

    @property
    def last_commit_epoch(self) -> int:
        return self._last_commit_epoch

    @property
    def active_version(self) -> int:
        return self._versions[self._active_bank]

    def read_active(self) -> DecoderRuntimeParams:
        return self._banks[self._active_bank].copy()

    def read_staging(self) -> DecoderRuntimeParams:
        return self._banks[self._staging_bank].copy()

    def stage_update(
        self,
        params: DecoderRuntimeParams,
        *,
        commit_epoch: int,
        staged_epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PendingCommit:
        if commit_epoch < self._epoch_id:
            raise ValueError(
                f"commit_epoch must be >= current epoch ({self._epoch_id}), got {commit_epoch}"
            )

        staged_params = params.copy()
        staged_metadata = dict(metadata or {})
        staged_bank = self._staging_bank
        version = self._next_version
        self._next_version += 1

        self._banks[staged_bank] = staged_params
        self._versions[staged_bank] = version
        self._pending_commit = PendingCommit(
            target_bank=staged_bank,
            commit_epoch=commit_epoch,
            staged_epoch=self._epoch_id if staged_epoch is None else staged_epoch,
            version=version,
            metadata=staged_metadata,
        )
        return self._pending_commit

    def commit_if_ready(self, epoch_id: int) -> Optional[CommitResult]:
        if epoch_id < self._epoch_id:
            raise ValueError(f"epoch_id must be monotonic, got {epoch_id} after {self._epoch_id}")

        self._epoch_id = epoch_id
        if self._pending_commit is None:
            return None
        if epoch_id < self._pending_commit.commit_epoch:
            return None

        target_bank = self._pending_commit.target_bank
        self._active_bank = target_bank
        self._staging_bank = "A" if target_bank == "B" else "B"
        self._last_commit_epoch = epoch_id

        result = CommitResult(
            activated_bank=target_bank,
            commit_epoch=epoch_id,
            version=self._pending_commit.version,
            metadata=dict(self._pending_commit.metadata),
        )
        self._pending_commit = None
        return result

    def discard_pending(self) -> None:
        self._pending_commit = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "epoch_id": self._epoch_id,
            "active_bank": self._active_bank,
            "staging_bank": self._staging_bank,
            "active_version": self._versions[self._active_bank],
            "pending_commit": None
            if self._pending_commit is None
            else {
                "target_bank": self._pending_commit.target_bank,
                "commit_epoch": self._pending_commit.commit_epoch,
                "staged_epoch": self._pending_commit.staged_epoch,
                "version": self._pending_commit.version,
                "metadata": dict(self._pending_commit.metadata),
            },
            "banks": {
                name: {
                    "version": self._versions[name],
                    "params": self._banks[name].to_dict(),
                }
                for name in ("A", "B")
            },
        }
