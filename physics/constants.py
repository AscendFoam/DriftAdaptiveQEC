"""轻量级共享物理常量模块。

本模块集中定义整个 physics 包共享的基础物理常量，避免各子模块各自重复定义、
也避免出现多份取值不一致的常量。当前仅导出 GKP 晶格常量 ``LATTICE_CONST``。

LATTICE_CONST = √(2π) ≈ 2.5066
- 物理含义：GKP 码在相空间 (q, p) 中所采用的方晶格的晶格常数，
  即相邻两个 GKP 梳齿之间的间距。
- 使用范围：综合征取模测量、逻辑错误判定边界 (±LATTICE_CONST/2)、
  残差 wrap 等几乎所有物理子模块都依赖该常量。
"""

from __future__ import annotations

from math import pi, sqrt


# GKP 晶格常数 √(2π)。全库共享，请勿在各子模块中重复定义不同取值。
LATTICE_CONST = sqrt(2.0 * pi)

__all__ = ["LATTICE_CONST"]
