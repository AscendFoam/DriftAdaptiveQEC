"""Mock FPGA backend for P3 HIL event-driven validation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from cnn_fpga.hwio.axi_map import AXI_REGISTER_MAP, AxiRegisterMap
from cnn_fpga.hwio.dma_client import DMAReadout
from cnn_fpga.runtime import FastLoopEmulator, LatencyInjector, ParamBank, SchedulerConfig, WindowFrame
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


NoiseProvider = Callable[[int, float], Dict[str, Any]]


@dataclass(frozen=True)
class MockFPGAConfig:
    """Configuration of the mock FPGA board-side behavior."""

    scheduler: SchedulerConfig
    dma_buffer_count: int = 2
    histogram_buffer_bytes: int = 4096

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MockFPGAConfig":
        dma_cfg = config.get("dma", {})
        return cls(
            scheduler=SchedulerConfig.from_config(config),
            dma_buffer_count=int(dma_cfg.get("buffer_count", 2)),
            histogram_buffer_bytes=int(dma_cfg.get("histogram_buffer_bytes", 4096)),
        )


@dataclass
class MockFPGAEvent:
    """Structured backend event for HIL logging."""

    kind: str
    epoch_id: int
    time_us: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "epoch_id": self.epoch_id,
            "time_us": self.time_us,
            "details": dict(self.details),
        }


class MockFPGA:
    """Mock board backend that emulates AXI/DMA-visible FPGA behavior."""

    def __init__(
        self,
        config: MockFPGAConfig,
        *,
        param_bank: Optional[ParamBank] = None,
        fast_loop: Optional[FastLoopEmulator] = None,
        latency_injector: Optional[LatencyInjector] = None,
        axi_map: Optional[AxiRegisterMap] = None,
    ) -> None:
        self.config = config
        self.param_bank = param_bank or ParamBank()
        self.latency_injector = latency_injector or LatencyInjector()
        self.axi_map = axi_map or AXI_REGISTER_MAP
        self.fast_loop = fast_loop or FastLoopEmulator.from_config({}, param_bank=self.param_bank)

        self.epoch_id = 0
        self.time_us = 0.0
        self.window_counter = 0
        self.next_window_emit_epoch = self.config.scheduler.window_size
        self.fast_cycle_budget_violations = 0
        self.dropped_dma_buffers = 0
        self._hist_sequence = 0
        self._last_buffer_id = 0
        self._last_overflow_alert = False
        self._last_overflow_count = 0

        self._started = False
        self._commit_ack = False
        self._pending_commit_ack_time_us: Optional[float] = None
        self._pending_commit_ack_delay_us: Optional[float] = None
        self._staged_params: DecoderRuntimeParams = self.param_bank.read_staging()
        self._registers: Dict[int, int] = {}
        self._dma_buffers: Deque[DMAReadout] = deque(maxlen=self.config.dma_buffer_count)
        self._next_buffer_id = 0

        identity_words = self.axi_map.pack_params(self.param_bank.read_staging())
        self._registers.update(identity_words)
        self._registers[self.axi_map.ctrl_addr] = 0
        self._registers[self.axi_map.active_bank_addr] = self.axi_map.encode_bank(self.param_bank.active_bank_name)
        self._registers[self.axi_map.epoch_id_addr] = 0
        self._registers[self.axi_map.hist_meta_addr] = 0
        self._registers[self.axi_map.overflow_count_addr] = 0
        self._registers[self.axi_map.commit_epoch_addr] = 0
        self._registers[self.axi_map.hist_seq_addr] = 0

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        noise_provider: Optional[NoiseProvider] = None,
        seed: int | None = None,
    ) -> "MockFPGA":
        latency = LatencyInjector.from_config(config, seed=seed)
        param_bank = ParamBank()
        fast_loop = FastLoopEmulator.from_config(
            config,
            param_bank=param_bank,
            noise_provider=noise_provider,
            seed=None if seed is None else seed + 1,
        )
        return cls(
            MockFPGAConfig.from_config(config),
            param_bank=param_bank,
            fast_loop=fast_loop,
            latency_injector=latency,
            axi_map=AxiRegisterMap(
                fixed_point_spec=str(config.get("hardware_defaults", {}).get("fixed_point", "Q4.20"))
            ),
        )

    @property
    def scheduler_config(self) -> SchedulerConfig:
        return self.config.scheduler

    def start(self) -> None:
        self._started = True
        self._registers[self.axi_map.ctrl_addr] = self.axi_map.build_ctrl_word(start=True)

    def reset_histogram(self) -> None:
        self._dma_buffers.clear()
        self._commit_ack = False
        self._pending_commit_ack_time_us = None
        self._pending_commit_ack_delay_us = None
        self._registers[self.axi_map.hist_meta_addr] = 0
        self._registers[self.axi_map.overflow_count_addr] = 0

    def read_active_params(self) -> DecoderRuntimeParams:
        return self.param_bank.read_active()

    def has_pending_commit(self) -> bool:
        return self.param_bank.has_pending_commit

    def histogram_available(self) -> bool:
        return bool(self._dma_buffers)

    def read_status_fields(self) -> Dict[str, Any]:
        return {
            **self.axi_map.decode_status_word(self.read_register(self.axi_map.status_addr)),
            "epoch_id": self.epoch_id,
            "time_us": self.time_us,
            "active_bank": self.param_bank.active_bank_name,
            "active_version": self.param_bank.active_version,
            "pending_commit": self.param_bank.has_pending_commit,
            "pending_dma_buffers": len(self._dma_buffers),
        }

    def read_register(self, addr: int) -> int:
        if addr == self.axi_map.status_addr:
            return self.axi_map.build_status_word(
                ready=self._started,
                hist_ready=bool(self._dma_buffers),
                commit_ack=self._commit_ack,
                overflow_alert=self._last_overflow_alert,
            )
        if addr == self.axi_map.active_bank_addr:
            return self.axi_map.encode_bank(self.param_bank.active_bank_name)
        if addr == self.axi_map.epoch_id_addr:
            return int(self.epoch_id)
        return int(self._registers.get(addr, 0))

    def write_register(self, addr: int, value: int) -> None:
        self._registers[addr] = int(value) & 0xFFFFFFFF
        if addr != self.axi_map.ctrl_addr:
            return

        ctrl = self.axi_map.decode_ctrl_word(value)
        if ctrl["start"]:
            self._started = True
        if ctrl["reset_hist"]:
            self.reset_histogram()

    def stage_params(self, params: DecoderRuntimeParams) -> None:
        self._staged_params = params.copy()
        self._registers.update(self.axi_map.pack_params(params))

    def schedule_commit(
        self,
        *,
        commit_epoch: int,
        ack_delay_us: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        params = self._staged_params.copy()
        pending = self.param_bank.stage_update(
            params,
            commit_epoch=commit_epoch,
            staged_epoch=self.epoch_id,
            metadata=dict(metadata or {}),
        )
        self._commit_ack = False
        self._pending_commit_ack_time_us = None
        self._pending_commit_ack_delay_us = float(
            self.latency_injector.commit_ack.sample(self.latency_injector._rng)
            if ack_delay_us is None
            else ack_delay_us
        )
        self._registers[self.axi_map.ctrl_addr] = self.axi_map.build_ctrl_word(start=self._started, commit_bank=True)
        self._registers[self.axi_map.commit_epoch_addr] = int(commit_epoch)
        return {
            "target_bank": pending.target_bank,
            "commit_epoch": pending.commit_epoch,
            "version": pending.version,
            "ack_delay_us": self._pending_commit_ack_delay_us,
        }

    def pop_histogram_buffer(self) -> DMAReadout:
        if not self._dma_buffers:
            raise RuntimeError("histogram buffer empty")
        return self._dma_buffers.popleft()

    def _push_histogram(self, window: WindowFrame, events: List[MockFPGAEvent]) -> None:
        buffer_id = self._next_buffer_id
        self._next_buffer_id = (self._next_buffer_id + 1) % max(1, self.config.dma_buffer_count)
        readout = DMAReadout.from_window(
            buffer_id=buffer_id,
            window=window,
            metadata={"backend": "mock_fpga"},
        )

        if len(self._dma_buffers) >= self._dma_buffers.maxlen:
            dropped = self._dma_buffers.popleft()
            self.dropped_dma_buffers += 1
            events.append(
                MockFPGAEvent(
                    kind="dma_buffer_dropped",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={"dropped_buffer_id": dropped.buffer_id, "dropped_window_id": dropped.window.window_id},
                )
            )
        self._dma_buffers.append(readout)
        diagnostics = dict(window.payload.get("diagnostics", {}))
        self._hist_sequence += 1
        self._last_buffer_id = buffer_id
        self._last_overflow_alert = bool(diagnostics.get("overflow_alert", False))
        overflow_ratio = float(diagnostics.get("overflow_ratio", 0.0))
        self._last_overflow_count = int(round(overflow_ratio * self.config.scheduler.window_size))
        self._registers[self.axi_map.hist_meta_addr] = self.axi_map.build_hist_meta_word(
            buffer_id=buffer_id,
            overflow_alert=self._last_overflow_alert,
        )
        self._registers[self.axi_map.overflow_count_addr] = int(self._last_overflow_count)
        self._registers[self.axi_map.hist_seq_addr] = int(self._hist_sequence)

    def step(self, cycles: int = 1) -> List[MockFPGAEvent]:
        if cycles <= 0:
            raise ValueError("cycles must be positive")

        events: List[MockFPGAEvent] = []
        for _ in range(cycles):
            if not self._started:
                continue

            self.epoch_id += 1
            self.time_us = self.epoch_id * self.config.scheduler.t_fast_us
            self._registers[self.axi_map.epoch_id_addr] = int(self.epoch_id)

            fast_latency = self.latency_injector.sample_fast_cycle()
            if fast_latency > self.config.scheduler.fast_path_budget_us:
                self.fast_cycle_budget_violations += 1
                events.append(
                    MockFPGAEvent(
                        kind="fast_budget_violation",
                        epoch_id=self.epoch_id,
                        time_us=self.time_us,
                        details={
                            "latency_us": fast_latency,
                            "budget_us": self.config.scheduler.fast_path_budget_us,
                        },
                    )
                )

            commit_result = self.param_bank.commit_if_ready(self.epoch_id)
            if commit_result is not None:
                ack_delay_us = float(
                    self.latency_injector.commit_ack.sample(self.latency_injector._rng)
                    if self._pending_commit_ack_delay_us is None
                    else self._pending_commit_ack_delay_us
                )
                self._pending_commit_ack_time_us = self.time_us + max(0.0, ack_delay_us)
                self._pending_commit_ack_delay_us = None
                self._registers[self.axi_map.active_bank_addr] = self.axi_map.encode_bank(commit_result.activated_bank)
                events.append(
                    MockFPGAEvent(
                        kind="commit_applied",
                        epoch_id=self.epoch_id,
                        time_us=self.time_us,
                        details={
                            "activated_bank": commit_result.activated_bank,
                            "version": commit_result.version,
                        },
                    )
                )

            if self._pending_commit_ack_time_us is not None and self.time_us >= self._pending_commit_ack_time_us:
                self._commit_ack = True
                self._pending_commit_ack_time_us = None
                self._registers[self.axi_map.commit_epoch_addr] = int(self.epoch_id)
                events.append(
                    MockFPGAEvent(
                        kind="commit_ack_asserted",
                        epoch_id=self.epoch_id,
                        time_us=self.time_us,
                        details={"active_bank": self.param_bank.active_bank_name},
                    )
                )

            emit_window = self.epoch_id >= self.next_window_emit_epoch
            payload = self.fast_loop.step(self.epoch_id, self.time_us, emit_window=emit_window)
            if emit_window:
                self.window_counter += 1
                window = WindowFrame(
                    window_id=self.window_counter,
                    start_epoch=self.epoch_id - self.config.scheduler.window_size + 1,
                    end_epoch=self.epoch_id,
                    ready_time_us=self.time_us,
                    payload=dict(payload or {}),
                )
                self._push_histogram(window, events)
                events.append(
                    MockFPGAEvent(
                        kind="window_ready",
                        epoch_id=self.epoch_id,
                        time_us=self.time_us,
                        details={
                            "window_id": window.window_id,
                            "buffer_id": self._dma_buffers[-1].buffer_id,
                            "epoch_start": window.start_epoch,
                            "epoch_end": window.end_epoch,
                        },
                    )
                )
                self.next_window_emit_epoch += self.config.scheduler.resolved_window_stride

        return events

    def snapshot(self) -> Dict[str, Any]:
        return {
            "backend": "mock",
            "epoch_id": self.epoch_id,
            "time_us": self.time_us,
            "started": self._started,
            "fast_cycle_budget_violations": self.fast_cycle_budget_violations,
            "dropped_dma_buffers": self.dropped_dma_buffers,
            "pending_commit_ack_time_us": self._pending_commit_ack_time_us,
            "status": self.read_status_fields(),
            "dma": {
                "pending_buffers": len(self._dma_buffers),
                "buffer_ids": [item.buffer_id for item in self._dma_buffers],
            },
            "param_bank": self.param_bank.snapshot(),
            "hist_meta": {
                "buffer_id": self._last_buffer_id,
                "hist_sequence": self._hist_sequence,
                "overflow_alert": self._last_overflow_alert,
                "overflow_count": self._last_overflow_count,
            },
            "fast_loop": self.fast_loop.summary(),
        }
