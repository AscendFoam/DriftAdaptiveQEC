"""physics 包的对外统一入口（惰性加载层）。

设计目的：
- physics 子模块（gkp_state、noise_channels、syndrome_measurement、
  error_correction、logical_tracking）之间存在依赖，且部分子模块会尝试导入
  较重的第三方库（如 Strawberry Fields）。为了避免 ``import physics`` 时一次性
  把所有子模块及其依赖全部加载，本文件通过模块级 ``__getattr__`` 实现"按需惰性加载"：
  只有在真正访问某个公开符号时，才导入它所在的子模块。

约定：
- ``_EXPORTS`` 字典维护"公开符号名 -> 所在子模块"的映射；
- ``__all__`` 由 ``_EXPORTS`` 的键自动生成，用于 ``from physics import *``；
- 首次访问某符号后，结果会缓存到模块 globals()，后续访问不再重复导入。
"""

from __future__ import annotations

from importlib import import_module


# 公开符号 -> 所在子模块的映射。新增导出符号时只需在此登记。
_EXPORTS = {
    "ApproximateGKPState": "physics.gkp_state",
    "GKPStateFactory": "physics.gkp_state",
    "QuantumNoiseChannel": "physics.noise_channels",
    "PhotonLossChannel": "physics.noise_channels",
    "ThermalNoiseChannel": "physics.noise_channels",
    "SyndromeMeasurement": "physics.syndrome_measurement",
    "RealisticSyndromeMeasurement": "physics.syndrome_measurement",
    "GKPErrorCorrector": "physics.error_correction",
    "LinearDecoder": "physics.error_correction",
    "LogicalErrorTracker": "physics.logical_tracking",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """模块级惰性属性访问钩子（PEP 562）。

    功能:
        把 ``physics.X`` 或 ``from physics import X`` 解析为对子模块的按需导入——
        只有在真正访问 X 时，才触发对应子模块的 import，并把结果缓存到模块
        ``globals()``，从而避免一次性加载所有子模块及其重依赖。

    输入:
        name: 被访问的属性名，即某个公开符号（如 'ApproximateGKPState'）。

    输出:
        返回该符号在对应子模块中的实际对象；首次解析后缓存，后续访问直接命中缓存。

    异常:
        若 ``name`` 不在 ``_EXPORTS`` 中，抛出 ``AttributeError``，
        提示当前模块没有该属性。
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
