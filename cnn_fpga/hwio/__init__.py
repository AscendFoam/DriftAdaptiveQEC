"""Hardware-I/O layer for mock/real CNN-FPGA HIL backends."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AXI_REGISTER_MAP": "cnn_fpga.hwio.axi_map",
    "AxiRegisterMap": "cnn_fpga.hwio.axi_map",
    "BoardBackendError": "cnn_fpga.hwio.board_backend",
    "BoardBackendUnavailableError": "cnn_fpga.hwio.board_backend",
    "BoardFPGA": "cnn_fpga.hwio.board_backend",
    "BoardFPGAConfig": "cnn_fpga.hwio.board_backend",
    "BackendDMAClient": "cnn_fpga.hwio.dma_client",
    "DMAClient": "cnn_fpga.hwio.dma_client",
    "DMAReadout": "cnn_fpga.hwio.dma_client",
    "MemoryMappedDMAClient": "cnn_fpga.hwio.dma_client",
    "MemoryMappedDMAConfig": "cnn_fpga.hwio.dma_client",
    "FPGADriver": "cnn_fpga.hwio.fpga_driver",
    "FPGADriverConfig": "cnn_fpga.hwio.fpga_driver",
    "FPGADriverError": "cnn_fpga.hwio.fpga_driver",
    "MockFPGA": "cnn_fpga.hwio.mock_fpga",
    "MockFPGAConfig": "cnn_fpga.hwio.mock_fpga",
    "MockFPGAEvent": "cnn_fpga.hwio.mock_fpga",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
