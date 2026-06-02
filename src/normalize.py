"""正規化。

2層構成:
  - 第1層: per-sample 正規化（l_ref、build_data 内で実施済み、リークなし）
  - 第2層: global 標準化（訓練セットのみで fit、one-hot/sin/cos は除外）

加えて、Cartesian オフセット (y - anchor) の標準化統計も訓練から計算する。
これらの統計は保存し、推論時にも同じものを適用する。
"""

from __future__ import annotations

import torch

from . import constants as C


class FeatureNormalizer:
    """ノード/エッジ特徴量の global 標準化。訓練データのみで fit する。"""

    def __init__(self, node_norm_cols=None, edge_norm_cols=None):
        self.node_norm_cols = node_norm_cols if node_norm_cols is not None else C.NODE_NORM_COLS
        self.edge_norm_cols = edge_norm_cols if edge_norm_cols is not None else C.EDGE_NORM_COLS
        self.node_mean = self.node_std = None
        self.edge_mean = self.edge_std = None
        self.offset_mean = self.offset_std = None   # Cartesian オフセット (y - anchor)

    def fit(self, train_data_list):
        # ノード特徴量
        xs = torch.cat([d.x[:, self.node_norm_cols] for d in train_data_list], dim=0)
        self.node_mean = xs.mean(0)
        self.node_std = xs.std(0).clamp(min=1e-6)

        # エッジ特徴量（エッジを持つサンプルのみ）
        eas = [d.edge_attr[:, self.edge_norm_cols] for d in train_data_list
               if d.edge_attr.shape[0] > 0]
        if eas:
            eas = torch.cat(eas, dim=0)
            self.edge_mean = eas.mean(0)
            self.edge_std = eas.std(0).clamp(min=1e-6)
        else:
            k = len(self.edge_norm_cols)
            self.edge_mean = torch.zeros(k)
            self.edge_std = torch.ones(k)

        # オフセット統計（可変ノードの y - anchor、Cartesian）
        offsets = []
        for d in train_data_list:
            vm = d.variable_mask
            off = d.y[vm] - d.anchor[vm]      # [N_var, 2]
            offsets.append(off)
        offsets = torch.cat(offsets, dim=0)
        self.offset_mean = offsets.mean(0)
        self.offset_std = offsets.std(0).clamp(min=1e-6)
        return self

    def transform(self, data):
        """data を in-place で標準化（特徴量のみ。anchor/y は変えない）。"""
        data.x[:, self.node_norm_cols] = (
            (data.x[:, self.node_norm_cols] - self.node_mean) / self.node_std
        )
        if data.edge_attr.shape[0] > 0:
            data.edge_attr[:, self.edge_norm_cols] = (
                (data.edge_attr[:, self.edge_norm_cols] - self.edge_mean) / self.edge_std
            )
        return data

    def normalize_offset(self, offset):
        """損失計算用: Cartesian オフセットを標準化。"""
        return (offset - self.offset_mean) / self.offset_std

    def denormalize_offset(self, offset_norm):
        """推論用: 標準化オフセットを元スケールに戻す。"""
        return offset_norm * self.offset_std + self.offset_mean

    def save(self, path):
        torch.save({
            "node_norm_cols": self.node_norm_cols,
            "edge_norm_cols": self.edge_norm_cols,
            "node_mean": self.node_mean, "node_std": self.node_std,
            "edge_mean": self.edge_mean, "edge_std": self.edge_std,
            "offset_mean": self.offset_mean, "offset_std": self.offset_std,
        }, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        obj = cls(d["node_norm_cols"], d["edge_norm_cols"])
        obj.node_mean, obj.node_std = d["node_mean"], d["node_std"]
        obj.edge_mean, obj.edge_std = d["edge_mean"], d["edge_std"]
        obj.offset_mean, obj.offset_std = d["offset_mean"], d["offset_std"]
        return obj
