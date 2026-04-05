"""Hardware-I/O layer for mock/real CNN-FPGA HIL backends."""

# 中文说明：
# - hwio 子模块把 HIL 实验入口和具体板级访问方式隔离开。
# - 当前首版先提供 mock FPGA 后端与统一 driver API，后续可平滑替换为真实板卡实现。

from .axi_map import AXI_REGISTER_MAP, AxiRegisterMap
from .board_backend import BoardBackendError, BoardBackendUnavailableError, BoardFPGA, BoardFPGAConfig
from .dma_client import BackendDMAClient, DMAClient, DMAReadout, MemoryMappedDMAClient, MemoryMappedDMAConfig
from .fpga_driver import FPGADriver, FPGADriverConfig, FPGADriverError
from .mock_fpga import MockFPGA, MockFPGAConfig, MockFPGAEvent

__all__ = [
    "AXI_REGISTER_MAP",
    "AxiRegisterMap",
    "BackendDMAClient",
    "BoardBackendError",
    "BoardBackendUnavailableError",
    "BoardFPGA",
    "BoardFPGAConfig",
    "DMAClient",
    "DMAReadout",
    "FPGADriver",
    "FPGADriverConfig",
    "FPGADriverError",
    "MemoryMappedDMAClient",
    "MemoryMappedDMAConfig",
    "MockFPGA",
    "MockFPGAConfig",
    "MockFPGAEvent",
]
