"""DMA client abstractions for HIL histogram exchange."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import mmap
import os
from typing import Any, Dict

import numpy as np

from cnn_fpga.runtime.scheduler import WindowFrame


@dataclass
class DMAReadout:
    """One histogram window fetched from the FPGA DMA buffer."""

    buffer_id: int
    byte_count: int
    window: WindowFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_window(
        cls,
        *,
        buffer_id: int,
        window: WindowFrame,
        metadata: Dict[str, Any] | None = None,
    ) -> "DMAReadout":
        histogram = np.asarray(window.payload.get("histogram", []), dtype=np.float32)
        return cls(
            buffer_id=buffer_id,
            byte_count=int(histogram.nbytes),
            window=window,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buffer_id": self.buffer_id,
            "byte_count": self.byte_count,
            "window": self.window.to_dict(),
            "metadata": dict(self.metadata),
        }


class DMAClient(ABC):
    """Abstract DMA readout interface."""

    @abstractmethod
    def histogram_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_histogram(self) -> DMAReadout:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class BackendDMAClient(DMAClient):
    """DMA client that delegates to an in-process backend object."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def histogram_available(self) -> bool:
        return bool(self.backend.histogram_available())

    def read_histogram(self) -> DMAReadout:
        readout = self.backend.pop_histogram_buffer()
        if not isinstance(readout, DMAReadout):
            raise TypeError("backend.pop_histogram_buffer() must return DMAReadout")
        return readout

    def reset(self) -> None:
        self.backend.reset_histogram()


@dataclass(frozen=True)
class MemoryMappedDMAConfig:
    """Memory-mapped DMA buffer configuration."""

    path: str
    buffer_bytes: int
    buffer_count: int = 2


class MemoryMappedDMAClient(DMAClient):
    """DMA client for mmap-backed histogram buffers."""

    def __init__(self, config: MemoryMappedDMAConfig, *, backend: Any | None = None) -> None:
        self.config = config
        self.backend = backend
        self.fd = os.open(config.path, os.O_RDWR | os.O_SYNC)
        self.region = mmap.mmap(self.fd, config.buffer_bytes * max(1, config.buffer_count), access=mmap.ACCESS_READ)

    def histogram_available(self) -> bool:
        if self.backend is None:
            return True
        return bool(self.backend.histogram_available())

    def read_histogram(self) -> DMAReadout:
        if self.backend is None:
            raise RuntimeError("MemoryMappedDMAClient requires backend metadata to construct DMAReadout")
        return self.backend.pop_histogram_buffer()

    def reset(self) -> None:
        if self.backend is not None:
            self.backend.reset_histogram()

    def close(self) -> None:
        self.region.close()
        os.close(self.fd)
