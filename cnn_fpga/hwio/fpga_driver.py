"""Unified FPGA driver facade for mock and future real HIL backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cnn_fpga.hwio.axi_map import AXI_REGISTER_MAP, AxiRegisterMap
from cnn_fpga.hwio.dma_client import BackendDMAClient, DMAReadout
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.scheduler import SchedulerConfig


class FPGADriverError(RuntimeError):
    """Structured driver-side error."""


@dataclass(frozen=True)
class FPGADriverConfig:
    """Polling and timeout knobs for the HIL driver."""

    poll_interval_cycles: int = 1
    commit_timeout_cycles: int = 8192

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FPGADriverConfig":
        axi_cfg = config.get("axi", {})
        return cls(
            poll_interval_cycles=int(axi_cfg.get("poll_interval_cycles", 1)),
            commit_timeout_cycles=int(axi_cfg.get("commit_timeout_cycles", 8192)),
        )


class FPGADriver:
    """Driver facade exposing board-like operations to HIL scripts."""

    def __init__(
        self,
        backend: Any,
        *,
        axi_map: Optional[AxiRegisterMap] = None,
        config: Optional[FPGADriverConfig] = None,
    ) -> None:
        self.backend = backend
        self.axi_map = axi_map or AXI_REGISTER_MAP
        self.config = config or FPGADriverConfig()
        self.dma_client = BackendDMAClient(backend)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        noise_provider: Any = None,
        seed: int | None = None,
    ) -> "FPGADriver":
        backend_name = str(config.get("hil", {}).get("backend", "mock")).lower()
        if backend_name == "mock":
            from .mock_fpga import MockFPGA

            backend = MockFPGA.from_config(config, noise_provider=noise_provider, seed=seed)
            return cls(
                backend,
                axi_map=backend.axi_map,
                config=FPGADriverConfig.from_config(config),
            )
        if backend_name in {"board", "real"}:
            from .board_backend import BoardFPGA

            backend = BoardFPGA.from_config(config)
            return cls(
                backend,
                axi_map=backend.axi_map,
                config=FPGADriverConfig.from_config(config),
            )
        raise NotImplementedError(
            f"HIL backend '{backend_name}' is reserved for future real-board integration and is not implemented yet."
        )

    @property
    def scheduler_config(self) -> SchedulerConfig:
        return self.backend.scheduler_config

    def start(self) -> None:
        self.backend.start()

    def advance_cycles(self, cycles: int = 1) -> List[Any]:
        return list(self.backend.step(cycles))

    def reset_histogram(self) -> None:
        ctrl_word = self.axi_map.build_ctrl_word(start=True, reset_hist=True)
        self.backend.write_register(self.axi_map.ctrl_addr, ctrl_word)

    def read_status(self) -> Dict[str, Any]:
        if hasattr(self.backend, "read_status_fields"):
            return dict(self.backend.read_status_fields())
        return self.axi_map.decode_status_word(self.backend.read_register(self.axi_map.status_addr))

    def read_epoch(self) -> int:
        return int(self.backend.read_register(self.axi_map.epoch_id_addr))

    def read_time_us(self) -> float:
        return float(getattr(self.backend, "time_us", self.read_epoch() * self.scheduler_config.t_fast_us))

    def histogram_available(self) -> bool:
        return bool(self.dma_client.histogram_available())

    def read_histogram(self) -> DMAReadout:
        return self.dma_client.read_histogram()

    def read_active_params(self) -> DecoderRuntimeParams:
        return self.backend.read_active_params()

    def has_pending_commit(self) -> bool:
        return bool(self.backend.has_pending_commit())

    def stage_params(self, params: DecoderRuntimeParams, metadata: Dict[str, Any] | None = None) -> None:
        del metadata
        for addr, value in self.axi_map.pack_params(params).items():
            self.backend.write_register(addr, value)
        if hasattr(self.backend, "stage_params"):
            self.backend.stage_params(params)

    def commit_bank(
        self,
        *,
        commit_epoch: Optional[int] = None,
        ack_delay_us: Optional[float] = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        target_epoch = self.read_epoch() + self.scheduler_config.commit_delay_cycles if commit_epoch is None else commit_epoch
        if hasattr(self.backend, "schedule_commit"):
            return dict(
                self.backend.schedule_commit(
                    commit_epoch=target_epoch,
                    ack_delay_us=ack_delay_us,
                    metadata=metadata,
                )
            )
        ctrl_word = self.axi_map.build_ctrl_word(start=True, commit_bank=True)
        self.backend.write_register(self.axi_map.ctrl_addr, ctrl_word)
        return {"commit_epoch": target_epoch}

    def wait_commit_ack(self, max_cycles: int | None = None) -> Dict[str, Any]:
        timeout_cycles = self.config.commit_timeout_cycles if max_cycles is None else max_cycles
        for _ in range(timeout_cycles):
            status = self.read_status()
            if bool(status.get("commit_ack", False)):
                return status
            self.advance_cycles(self.config.poll_interval_cycles)
        raise FPGADriverError("commit_ack_timeout")

    def snapshot(self) -> Dict[str, Any]:
        if hasattr(self.backend, "snapshot"):
            return dict(self.backend.snapshot())
        return {
            "backend": "unknown",
            "epoch_id": self.read_epoch(),
            "time_us": self.read_time_us(),
            "status": self.read_status(),
        }

    def close(self) -> None:
        if hasattr(self.backend, "close"):
            self.backend.close()
