"""RawSample -> PyG Data への変換（CSV 実態版）。

- 座標は CSV の x, y, r, theta, sin, cos をそのまま使う
- エッジ長・曲率は CSV の precomputed 値を使う
- ターゲット・anchor は Cartesian（ユーザー方針 (a)）
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from . import constants as C
from .raw_sample import RawSample


def _reference_length(xy: np.ndarray) -> float:
    """代表長 = bounding box 対角長（Cartesian 座標から）。"""
    if xy.shape[0] == 0:
        return 1.0
    diag = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    return max(diag, 1e-6)


def _build_edges(sample: RawSample, l_ref: float):
    """無向の論理エッジを両方向化。kappa_abs_mean は対称なので向きでスワップ不要。"""
    src_list, dst_list, attr_list, type_list = [], [], [], []

    for e in sample.edges:
        length_norm = e.edge_length / l_ref
        kappa_norm = e.kappa_abs_mean * l_ref          # 1/length を無次元化
        etype_onehot = np.zeros(C.NUM_EDGE_TYPES, dtype=np.float32)
        etype_onehot[e.edge_type] = 1.0
        attr = np.concatenate([[length_norm, kappa_norm], etype_onehot]).astype(np.float32)

        # 両方向（対称なので同じ attr）
        for s, d in ((e.src, e.dst), (e.dst, e.src)):
            src_list.append(s)
            dst_list.append(d)
            attr_list.append(attr.copy())
            type_list.append(e.edge_type)

    if not src_list:
        return (np.zeros((2, 0), np.int64),
                np.zeros((0, C.EDGE_FEATURE_DIM), np.float32),
                np.zeros((0,), np.int64))

    edge_index = np.stack([np.array(src_list), np.array(dst_list)], 0).astype(np.int64)
    edge_attr = np.stack(attr_list, 0).astype(np.float32)
    edge_type = np.array(type_list, dtype=np.int64)
    return edge_index, edge_attr, edge_type


def _build_node_features(sample, xy, l_ref, edge_index, edge_lengths):
    """ノード特徴量 x を構築。"""
    n = len(sample.nodes)
    x = np.zeros((n, C.NODE_FEATURE_DIM), dtype=np.float32)

    # 隣接エッジ統計
    inc_len = np.zeros(n, np.float64)
    inc_cnt = np.zeros(n, np.float64)
    if edge_index.shape[1] > 0:
        src = edge_index[0]
        np.add.at(inc_len, src, edge_lengths)
        np.add.at(inc_cnt, src, 1.0)
    inc_len_mean = np.where(inc_cnt > 0, inc_len / np.maximum(inc_cnt, 1.0), 0.0) / l_ref

    for i, node in enumerate(sample.nodes):
        c = 0
        x[i, c + node.node_type] = 1.0;  c += C.NUM_NODE_TYPES   # type one-hot
        x[i, c] = node.r / l_ref;        c += 1                  # r_norm
        x[i, c] = node.sin_theta;        c += 1                  # sin
        x[i, c] = node.cos_theta;        c += 1                  # cos
        x[i, c] = node.x / l_ref;        c += 1                  # x_norm
        x[i, c] = node.y / l_ref;        c += 1                  # y_norm
        x[i, c] = inc_len_mean[i];       c += 1                  # 隣接エッジ長平均
        x[i, c] = inc_cnt[i] / 8.0;      c += 1                  # 隣接エッジ数（緩く正規化）
    return x


def _compute_anchor_cartesian(xy, node_types, edge_index):
    """可変ノードの anchor を、隣接 context の Cartesian 平均で計算。
    context 隣接なし -> 可変隣接平均 -> 全体重心 でフォールバック。"""
    n = xy.shape[0]
    anchor = np.zeros((n, 2), np.float64)
    is_var = node_types == C.VAR_ID
    is_ctx = np.isin(node_types, list(C.CONTEXT_TYPE_IDS))

    if edge_index.shape[1] == 0:
        anchor[is_var] = xy[is_var]
        return anchor

    src, dst = edge_index[0], edge_index[1]

    # 1段: context -> var
    acc = np.zeros((n, 2)); cnt = np.zeros(n)
    m = is_var[dst] & is_ctx[src]
    np.add.at(acc, dst[m], xy[src[m]]); np.add.at(cnt, dst[m], 1.0)
    has = cnt > 0
    sel = is_var & has
    anchor[sel] = acc[sel] / cnt[sel, None]

    # 2段: var -> var（context 隣接なしの可変ノード用）
    rest = is_var & ~has
    if rest.any():
        acc2 = np.zeros((n, 2)); cnt2 = np.zeros(n)
        m2 = rest[dst] & is_var[src]
        np.add.at(acc2, dst[m2], xy[src[m2]]); np.add.at(cnt2, dst[m2], 1.0)
        has2 = cnt2 > 0
        sel2 = rest & has2
        anchor[sel2] = acc2[sel2] / cnt2[sel2, None]
        # 3段: 全体重心
        rest2 = rest & ~has2
        if rest2.any():
            anchor[rest2] = xy.mean(0)
    return anchor


def _compute_anchor(xy, node_types, edge_index, anchor_mode):
    """anchor をモードに応じて計算する。

    anchor_mode:
      "zero"         : anchor を全ゼロにする（=直接予測。pred_pos = 0 + offset = 絶対座標）
      "context_mean" : 隣接 context 平均（可変-可変接続が多い場合は質が落ちる点に注意）
      （将来）"design_space" : 設計空間中心。設計空間定義が判明したら追加する。
    """
    n = xy.shape[0]
    if anchor_mode == "zero":
        return np.zeros((n, 2), np.float64)
    elif anchor_mode == "context_mean":
        return _compute_anchor_cartesian(xy, node_types, edge_index)
    else:
        raise ValueError(f"未知の anchor_mode: {anchor_mode}")


def build_data(sample: RawSample, anchor_mode: str = "zero") -> Data:
    """RawSample から PyG Data を構築する。

    Args:
        sample: 入力サンプル
        anchor_mode: anchor の計算方式。デフォルト "zero"（直接予測）。
                     設計空間定義が判明したら "design_space" を追加して切替える。
    """
    n = len(sample.nodes)

    polar = np.array([[nd.r, nd.theta] for nd in sample.nodes], dtype=np.float64)
    xy = np.array([[nd.x, nd.y] for nd in sample.nodes], dtype=np.float64)
    node_types = np.array([nd.node_type for nd in sample.nodes], dtype=np.int64)
    l_ref = _reference_length(xy)

    edge_index, edge_attr, edge_type = _build_edges(sample, l_ref)
    edge_lengths = (edge_attr[:, 0].astype(np.float64) * l_ref
                    if edge_attr.shape[0] > 0 else np.zeros((0,), np.float64))

    x = _build_node_features(sample, xy, l_ref, edge_index, edge_lengths)
    variable_mask = (node_types == C.VAR_ID)

    # slot_id: 可変ノードに一筆書き順（node_id 順）の通し番号を振る。
    # RawSample.nodes は parser で node_id 順にソート済みなので、
    # variable_mask の昇順インデックスがそのまま一筆書きの可変順序になる。
    # context ノードは -1（未使用を表す）。
    slot_id = np.full(n, -1, dtype=np.int64)
    var_idx = np.where(variable_mask)[0]        # 昇順 = 一筆書き順
    slot_id[var_idx] = np.arange(len(var_idx))  # 0,1,2,... と採番

    # anchor / target は Cartesian（ユーザー方針 (a)）
    # 当面は anchor_mode="zero"（直接予測）。設計空間判明後に切替予定。
    anchor = _compute_anchor(xy, node_types, edge_index, anchor_mode)
    y = np.zeros((n, 2), np.float64)
    y[variable_mask] = xy[variable_mask]    # 可変ノード位置自体がターゲット（VAE 再構成）

    data = Data(
        x=torch.from_numpy(x),
        pos=torch.from_numpy(polar.astype(np.float32)),       # 極座標（参照用）
        pos_cart=torch.from_numpy(xy.astype(np.float32)),     # Cartesian（target/anchor 基準）
        node_type=torch.from_numpy(node_types),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        edge_type=torch.from_numpy(edge_type),
        variable_mask=torch.from_numpy(variable_mask),
        anchor=torch.from_numpy(anchor.astype(np.float32)),   # Cartesian（zero モードでは全ゼロ）
        y=torch.from_numpy(y.astype(np.float32)),             # Cartesian
        slot_id=torch.from_numpy(slot_id),                    # 一筆書き順の可変通し番号（context は -1）
    )
    data.graph_id = torch.tensor([sample.graph_id], dtype=torch.long)
    data.extremum_count = torch.tensor([sample.extremum_count], dtype=torch.long)
    data.section_type = torch.tensor([sample.section_type], dtype=torch.long)
    data.has_extension = torch.tensor([sample.has_extension], dtype=torch.bool)
    data.l_ref = torch.tensor([l_ref], dtype=torch.float32)
    return data
