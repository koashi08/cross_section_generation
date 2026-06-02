"""中間表現（RawSample）。CSV 実態に合わせた版。

生データ(CSV) --[parser.py]--> RawSample --[conversion.py]--> PyG Data
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RawNode:
    """1ノードの生情報（CSV のノード行に対応）。"""
    node_type: int                       # node_type_code
    r: float
    theta: float
    x: float
    y: float
    sin_theta: float
    cos_theta: float


@dataclass
class RawEdge:
    """1エッジの生情報（CSV のエッジ行に対応、無向の論理エッジ1本）。"""
    src: int                             # src_node_id
    dst: int                             # dst_node_id
    edge_type: int                       # edge_type_code
    edge_length: float                   # 補間点から計算済み
    kappa_abs_mean: float                # 補間点から3点法で計算済み（絶対平均曲率）


@dataclass
class RawSample:
    """1サンプル（1断面形状）。"""
    graph_id: int
    nodes: list[RawNode]
    edges: list[RawEdge]
    extremum_count: int
    section_type: int                    # closed_flag -> SECTION_TYPE
    has_extension: bool                  # ノードタイプから導出

    def __post_init__(self):
        n = len(self.nodes)
        for e in self.edges:
            if not (0 <= e.src < n and 0 <= e.dst < n):
                raise ValueError(
                    f"graph_id={self.graph_id}: edge index out of range "
                    f"({e.src},{e.dst}), n_nodes={n}"
                )
