"""Placeholder real-board backend using memory-mapped AXI/DMA interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cnn_fpga.hwio.axi_map import AXI_REGISTER_MAP, AxiRegisterMap
from cnn_fpga.hwio.dma_client import DMAReadout
from cnn_fpga.runtime import SchedulerConfig, WindowFrame
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


class BoardBackendError(RuntimeError):
    """Structured board-backend error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BoardBackendUnavailableError(BoardBackendError):
    """Raised when requested board resources are not available on the host."""


@dataclass(frozen=True)
class MMIOConfig:
    """Memory-mapped AXI-Lite region configuration."""

    path: str
    size_bytes: int = 4096
    base_offset: int = 0


@dataclass(frozen=True)
class DMABufferConfig:
    """Memory-mapped DMA region configuration."""

    path: str
    histogram_shape: tuple[int, int] = (32, 32)
    dtype: str = "float32"
    buffer_count: int = 2
    size_bytes: int = 4096


@dataclass(frozen=True)
class BoardFPGAConfig:
    """Configuration of the real-board backend placeholder."""

    scheduler: SchedulerConfig
    mmio: MMIOConfig
    dma: DMABufferConfig
    allow_missing_device: bool = True

    @classmethod
    def from_config(cls, config: Dict) -> "BoardFPGAConfig":
        hil_cfg = config.get("hil", {})
        board_cfg = hil_cfg.get("board_io", {})
        dma_cfg = config.get("dma", {})
        hist_bins = int(config.get("fast_loop", {}).get("histogram_bins", 32))
        return cls(
            scheduler=SchedulerConfig.from_config(config),
            mmio=MMIOConfig(
                path=str(board_cfg.get("axi_uio_path", "/dev/uio0")),
                size_bytes=int(board_cfg.get("axi_map_size_bytes", 4096)),
                base_offset=int(board_cfg.get("axi_base_offset", 0)),
            ),
            dma=DMABufferConfig(
                path=str(board_cfg.get("dma_buffer_path", "/dev/uio1")),
                histogram_shape=(hist_bins, hist_bins),
                dtype=str(board_cfg.get("dma_dtype", "float32")),
                buffer_count=int(dma_cfg.get("buffer_count", 2)),
                size_bytes=int(dma_cfg.get("histogram_buffer_bytes", 4096)),
            ),
            allow_missing_device=bool(board_cfg.get("allow_missing_device", True)),
        )


class MemoryMappedRegisterIO:
    """Minimal memory-mapped AXI-Lite register accessor."""

    def __init__(self, config: MMIOConfig) -> None:
        self.config = config
        self.fd = os.open(config.path, os.O_RDWR | os.O_SYNC)
        self.region = mmap.mmap(self.fd, config.size_bytes, access=mmap.ACCESS_WRITE, offset=config.base_offset)

    def read_u32(self, addr: int) -> int:
        self.region.seek(addr)
        return int.from_bytes(self.region.read(4), byteorder="little", signed=False)

    def write_u32(self, addr: int, value: int) -> None:
        self.region.seek(addr)
        self.region.write(int(value & 0xFFFFFFFF).to_bytes(4, byteorder="little", signed=False))
        self.region.flush()

    def close(self) -> None:
        self.region.close()
        os.close(self.fd)


class MemoryMappedDMARegion:
    """Minimal DMA buffer view backed by mmap."""

    def __init__(self, config: DMABufferConfig) -> None:
        self.config = config
        self.fd = os.open(config.path, os.O_RDWR | os.O_SYNC)
        total_size = config.size_bytes * max(1, config.buffer_count)
        self.region = mmap.mmap(self.fd, total_size, access=mmap.ACCESS_READ)

    def read_histogram(self, buffer_id: int) -> np.ndarray:
        if buffer_id < 0 or buffer_id >= self.config.buffer_count:
            raise BoardBackendError(f"invalid_dma_buffer_id:{buffer_id}")
        offset = buffer_id * self.config.size_bytes
        self.region.seek(offset)
        payload = self.region.read(self.config.size_bytes)
        dtype = np.dtype(self.config.dtype)
        count = int(np.prod(self.config.histogram_shape))
        return np.frombuffer(payload, dtype=dtype, count=count).reshape(self.config.histogram_shape).astype(np.float32)

    def close(self) -> None:
        self.region.close()
        os.close(self.fd)


class BoardFPGA:
    """Placeholder real-board backend exposing the same API as MockFPGA."""

    def __init__(
        self,
        config: BoardFPGAConfig,
        *,
        axi_map: Optional[AxiRegisterMap] = None,
        register_io: Optional[MemoryMappedRegisterIO] = None,
        dma_region: Optional[MemoryMappedDMARegion] = None,
    ) -> None:
        self.config = config
        self.axi_map = axi_map or AXI_REGISTER_MAP
        self.register_io = register_io
        self.dma_region = dma_region
        self.time_us = 0.0
        self.epoch_id = 0
        self._shadow_active_params = DecoderRuntimeParams.identity()
        self._shadow_staged_params = DecoderRuntimeParams.identity()
        self._pending_commit = False
        self._last_hist_sequence = 0
        self._last_buffer_id = 0
        self._last_overflow_alert = False
        self._last_overflow_count = 0
        self._started = False

    @classmethod
    def from_config(cls, config: Dict) -> "BoardFPGA":
        board_cfg = BoardFPGAConfig.from_config(config)
        mmio_path = Path(board_cfg.mmio.path).expanduser()
        dma_path = Path(board_cfg.dma.path).expanduser()
        if not mmio_path.exists() or not dma_path.exists():
            if board_cfg.allow_missing_device:
                missing = []
                if not mmio_path.exists():
                    missing.append(str(mmio_path))
                if not dma_path.exists():
                    missing.append(str(dma_path))
                raise BoardBackendUnavailableError(f"board_device_missing:{','.join(missing)}")
            raise FileNotFoundError(f"Board MMIO/DMA path missing: {mmio_path}, {dma_path}")

        return cls(
            board_cfg,
            axi_map=AxiRegisterMap(
                fixed_point_spec=str(config.get("hardware_defaults", {}).get("fixed_point", "Q4.20"))
            ),
            register_io=MemoryMappedRegisterIO(board_cfg.mmio),
            dma_region=MemoryMappedDMARegion(board_cfg.dma),
        )

    @property
    def scheduler_config(self) -> SchedulerConfig:
        return self.config.scheduler

    def start(self) -> None:
        self._started = True
        self.write_register(self.axi_map.ctrl_addr, self.axi_map.build_ctrl_word(start=True))

    def reset_histogram(self) -> None:
        self.write_register(self.axi_map.ctrl_addr, self.axi_map.build_ctrl_word(start=self._started, reset_hist=True))

    def read_active_params(self) -> DecoderRuntimeParams:
        return self._shadow_active_params.copy()

    def has_pending_commit(self) -> bool:
        status = self.read_status_fields()
        if status.get("commit_ack", False):
            self._pending_commit = False
        return self._pending_commit

    def histogram_available(self) -> bool:
        return bool(self.read_status_fields().get("hist_ready", False))

    def read_status_fields(self) -> Dict[str, object]:
        status = self.axi_map.decode_status_word(self.read_register(self.axi_map.status_addr))
        epoch_id = int(self.read_register(self.axi_map.epoch_id_addr))
        hist_meta = self.axi_map.decode_hist_meta_word(self.read_register(self.axi_map.hist_meta_addr))
        commit_epoch = int(self.read_register(self.axi_map.commit_epoch_addr))
        hist_seq = int(self.read_register(self.axi_map.hist_seq_addr))
        overflow_count = int(self.read_register(self.axi_map.overflow_count_addr))
        self.epoch_id = epoch_id
        self.time_us = epoch_id * self.config.scheduler.t_fast_us
        self._last_hist_sequence = hist_seq
        self._last_buffer_id = int(hist_meta["buffer_id"])
        self._last_overflow_alert = bool(hist_meta["overflow_alert"])
        self._last_overflow_count = overflow_count
        status.update(
            {
                "epoch_id": epoch_id,
                "time_us": self.time_us,
                "active_bank": self.axi_map.decode_bank(self.read_register(self.axi_map.active_bank_addr)),
                "pending_commit": self._pending_commit,
                "buffer_id": self._last_buffer_id,
                "hist_sequence": hist_seq,
                "overflow_count": overflow_count,
                "commit_epoch": commit_epoch,
            }
        )
        return status

    def read_register(self, addr: int) -> int:
        if self.register_io is None:
            raise BoardBackendUnavailableError("board_mmio_uninitialized")
        return self.register_io.read_u32(addr)

    def write_register(self, addr: int, value: int) -> None:
        if self.register_io is None:
            raise BoardBackendUnavailableError("board_mmio_uninitialized")
        self.register_io.write_u32(addr, value)

    def stage_params(self, params: DecoderRuntimeParams) -> None:
        self._shadow_staged_params = params.copy()
        for reg_addr, word in self.axi_map.pack_params(params).items():
            self.write_register(reg_addr, word)

    def schedule_commit(
        self,
        *,
        commit_epoch: int,
        ack_delay_us: float | None = None,
        metadata: Dict | None = None,
    ) -> Dict[str, object]:
        del ack_delay_us, metadata
        self._pending_commit = True
        self.write_register(self.axi_map.ctrl_addr, self.axi_map.build_ctrl_word(start=self._started, commit_bank=True))
        self.write_register(self.axi_map.commit_epoch_addr, int(commit_epoch))
        self._shadow_active_params = self._shadow_staged_params.copy()
        return {"target_bank": None, "commit_epoch": int(commit_epoch), "version": None, "ack_delay_us": None}

    def pop_histogram_buffer(self) -> DMAReadout:
        if self.dma_region is None:
            raise BoardBackendUnavailableError("board_dma_uninitialized")
        status = self.read_status_fields()
        epoch_id = int(status["epoch_id"])
        buffer_id = int(status.get("buffer_id", 0))
        histogram = self.dma_region.read_histogram(buffer_id)
        window = WindowFrame(
            window_id=epoch_id,
            start_epoch=max(1, epoch_id - self.config.scheduler.window_size + 1),
            end_epoch=epoch_id,
            ready_time_us=epoch_id * self.config.scheduler.t_fast_us,
            payload={
                "histogram": histogram,
                "diagnostics": {"valid_window": True, "board_backend": True},
            },
        )
        return DMAReadout.from_window(
            buffer_id=buffer_id,
            window=window,
            metadata={"backend": "board", "epoch_id": epoch_id},
        )

    def step(self, cycles: int = 1) -> List[Dict]:
        del cycles
        self.read_status_fields()
        return []

    def snapshot(self) -> Dict[str, object]:
        return {
            "backend": "board",
            "epoch_id": self.epoch_id,
            "time_us": self.time_us,
            "started": self._started,
            "status": self.read_status_fields(),
            "hist_meta": {
                "buffer_id": self._last_buffer_id,
                "hist_sequence": self._last_hist_sequence,
                "overflow_alert": self._last_overflow_alert,
                "overflow_count": self._last_overflow_count,
            },
            "shadow_active_params": self._shadow_active_params.to_dict(),
        }

    def close(self) -> None:
        if self.dma_region is not None:
            self.dma_region.close()
        if self.register_io is not None:
            self.register_io.close()
