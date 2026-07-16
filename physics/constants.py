"""轻量级共享物理常量模块。

本模块保留旧接口 ``LATTICE_CONST``，其唯一语义是 classical
decoder-standardized logical-cell spacing。完整的 canonical、displacement-amplitude
和 symplectic chart 约定在 :mod:`physics.quadrature_conventions` 中冻结。

LATTICE_CONST = √(2π) ≈ 2.5066
- 仓库约定：它是两个独立 classical decoder axis 的 standard-binning correction
  cell 间距，也是 centered modular syndrome 的周期；相邻 cell 属于相反逻辑陪集。
- 使用范围：综合征取模测量、逻辑错误判定边界 (±LATTICE_CONST/2)、残差 wrap。
- 禁止用法：不能把两个 decoder axis 一起解释成 canonical quantum operators；
  各轴同乘 √2 会把 commutator 从 i 改成 2i，不是 symplectic map。
- 数值巧合：Sivak 等论文 displacement-amplitude convention 的 stabilizer constant
  ``l_S`` 也等于 √(2π)，但它不是相邻 logical-coset correction-cell spacing。
"""

from __future__ import annotations

from .quadrature_conventions import DECODER_LOGICAL_CELL_SPACING


# 兼容旧接口：仅表示 decoder-standardized logical-cell spacing。
LATTICE_CONST = DECODER_LOGICAL_CELL_SPACING

__all__ = ["LATTICE_CONST"]
