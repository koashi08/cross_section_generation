"""デコーダ本体（Data 版）。

ContextEmbed -> ContextGNN -> SlotInit -> Slot Self-Att
-> Slot Cross-Att -> DecoderGNN -> OutputHead

OutputHead は use_anchor フラグで両対応:
  use_anchor=False（当面）: 直接座標予測 pred_pos = mlp(h)
  use_anchor=True         : オフセット予測 pred_pos = anchor + mlp(h)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph, to_dense_batch

from . import constants as C
from .encoder_modules import GATv2Stack, _AttnBlock


class OutputHead(nn.Module):
    """可変ノード埋め込み -> 2D 座標（or オフセット）。"""

    def __init__(self, hidden_dim, use_anchor=False, dropout=0.1):
        super().__init__()
        self.use_anchor = use_anchor
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h_var, anchor):
        out = self.mlp(h_var)
        if self.use_anchor:
            return anchor + out, out         # pred_pos, offset
        return out, out                      # pred_pos == offset（直接予測）


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int = C.NODE_FEATURE_DIM,
        hidden_dim: int = 64,
        z_global_dim: int = 8,
        z_var_dim: int = 16,
        z_fixed_dim: int = 8,
        edge_dim: int = C.EDGE_FEATURE_DIM,
        context_gnn_layers: int = 2,
        slot_self_layers: int = 2,
        slot_cross_layers: int = 2,
        decoder_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_var_nodes: int = 10,
        max_context_nodes: int = 50,
        use_anchor: bool = False,
        context_type_ids=C.CONTEXT_TYPE_IDS,
    ):
        super().__init__()
        self.context_type_ids = context_type_ids
        self.max_var_nodes = max_var_nodes
        self.max_context_nodes = max_context_nodes

        # (a) Context 埋め込み（x + z_global + z_fixed）
        self.context_embed = nn.Sequential(
            nn.Linear(in_dim + z_global_dim + z_fixed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # (b) Context GNN
        self.context_gnn = GATv2Stack(hidden_dim, context_gnn_layers, num_heads, edge_dim, dropout)

        # (c) Slot 初期化（in_dim(=可変ノードの x) + z_global + z_var）
        #     ※ x にノード位置が含まれるが、可変ノードの位置は予測対象なので
        #        ここでは x のうち位置以外の情報（タイプ等）を活かす設計。
        #        ただし直接予測では位置情報を使わないため、x をそのまま渡してもよい。
        self.slot_init = nn.Sequential(
            nn.Linear(in_dim + z_global_dim + z_var_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # (d)(e) Slot Self/Cross-Att
        self.slot_self = _AttnBlock(hidden_dim, slot_self_layers, num_heads, dropout)
        self.slot_cross = _AttnBlock(hidden_dim, slot_cross_layers, num_heads, dropout)

        # (f) DecoderGNN（全グラフ）
        self.decoder_gnn = GATv2Stack(hidden_dim, decoder_gnn_layers, num_heads, edge_dim, dropout)

        # (g) OutputHead
        self.out_head = OutputHead(hidden_dim, use_anchor=use_anchor, dropout=dropout)

    def forward(self, data, z_global, z_var, z_fixed, context_info=None, batch_size=None):
        device = data.x.device
        N = data.x.size(0)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(N, dtype=torch.long, device=device)
        if batch_size is None:
            batch_size = int(batch.max()) + 1

        var_mask = data.variable_mask
        ctx_mask = ~var_mask
        ctx_nodes = ctx_mask.nonzero(as_tuple=False).squeeze(-1)
        var_nodes = var_mask.nonzero(as_tuple=False).squeeze(-1)

        # (1) Context 埋め込み（z 注入）
        x_ctx = data.x[ctx_nodes]
        b_ctx = batch[ctx_nodes]
        h_ctx = self.context_embed(
            torch.cat([x_ctx, z_global[b_ctx], z_fixed[b_ctx]], dim=-1))

        # (2) Context GNN（context 部分グラフ）
        ctx_ei, ctx_ea = subgraph(
            subset=ctx_mask, edge_index=data.edge_index, edge_attr=data.edge_attr,
            relabel_nodes=True, num_nodes=N)
        h_ctx = self.context_gnn(h_ctx, ctx_ei, ctx_ea)

        # (3) Slot 初期化（z 注入）
        x_var = data.x[var_nodes]
        b_var = batch[var_nodes]
        slot_h = self.slot_init(
            torch.cat([x_var, z_global[b_var], z_var[b_var]], dim=-1))

        # (4) Slot Self-Att
        sd, sm = to_dense_batch(slot_h, b_var, max_num_nodes=self.max_var_nodes, batch_size=batch_size)
        x = self.slot_self(sd, kv=None, q_kpm=~sm)

        # (5) Slot Cross-Att（K/V = ContextGNN 出力）
        cd, cm = to_dense_batch(h_ctx, b_ctx, max_num_nodes=self.max_context_nodes, batch_size=batch_size)
        x = self.slot_cross(x, kv=cd, kv_kpm=~cm)
        slot_h = x[sm]                          # dense -> sparse

        # (6) 全グラフ DecoderGNN
        h_full = torch.zeros(N, slot_h.size(-1), device=device)
        h_full[ctx_nodes] = h_ctx
        h_full[var_nodes] = slot_h
        h_full = self.decoder_gnn(h_full, data.edge_index, data.edge_attr)

        # (7) OutputHead
        anchor_var = data.anchor[var_nodes]
        pred_pos, offset = self.out_head(h_full[var_nodes], anchor_var)

        return {
            "pred_pos": pred_pos,       # [N_var, 2]
            "offset": offset,           # [N_var, 2]
            "anchor": anchor_var,       # [N_var, 2]
            "var_nodes": var_nodes,     # 元グラフでの index（損失計算用）
        }
