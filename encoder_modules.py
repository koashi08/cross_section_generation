"""エンコーダの構成要素（Data 版）。

- SharedNodeEmbedding: per-node MLP（メッセージパッシングなし）
- GlobalBranch: GATv2Conv -> pool -> MLP -> (mu, logvar)
- ContextBranch: subgraph GATv2Conv -> 型別pool -> MLP -> z_fixed + ノード埋め込み
- VariableBranch: Self-Att -> Cross-Att -> pool -> MLP -> (mu, logvar)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import subgraph, to_dense_batch

from . import constants as C


# --------------------------------------------------------------------------
# SharedNodeEmbedding
# --------------------------------------------------------------------------
class SharedNodeEmbedding(nn.Module):
    """ノード特徴量を hidden_dim に射影する per-node MLP。MP なし。
    将来の軽量 GNN 化に備え、forward は edge 引数を受け取れる。"""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.embed(x)


# --------------------------------------------------------------------------
# 共通: GATv2Conv スタック
# --------------------------------------------------------------------------
class GATv2Stack(nn.Module):
    """残差接続 + LayerNorm 付きの GATv2Conv スタック。"""

    def __init__(self, hidden_dim, num_layers, num_heads, edge_dim, dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim は num_heads で割り切れる必要あり"
        head_dim = hidden_dim // num_heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=True,   # 孤立ノード対策
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout

    def forward(self, h, edge_index, edge_attr):
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h = F.dropout(F.relu(norm(h_new + h)), p=self.dropout, training=self.training)
        return h


# --------------------------------------------------------------------------
# GlobalBranch
# --------------------------------------------------------------------------
class GlobalBranch(nn.Module):
    """全ノード GNN -> pool -> MLP -> (mu, logvar)。"""

    def __init__(self, hidden_dim=64, z_dim=8, num_layers=2,
                 num_heads=4, edge_dim=C.EDGE_FEATURE_DIM, dropout=0.1):
        super().__init__()
        self.gnn = GATv2Stack(hidden_dim, num_layers, num_heads, edge_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

    def forward(self, h, edge_index, edge_attr, batch):
        h = self.gnn(h, edge_index, edge_attr)
        pooled = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch),
        ], dim=-1)
        mu, logvar = self.mlp(pooled).chunk(2, dim=-1)
        return mu, logvar


# --------------------------------------------------------------------------
# ContextBranch
# --------------------------------------------------------------------------
class ContextBranch(nn.Module):
    """context 部分グラフ GNN -> 型別 pool -> z_fixed。
    ノード埋め込みも返す（VariableBranch の Cross-Att 用）。"""

    def __init__(self, hidden_dim=64, z_fixed_dim=8, num_layers=2,
                 num_heads=4, edge_dim=C.EDGE_FEATURE_DIM, dropout=0.1,
                 context_type_ids=C.CONTEXT_TYPE_IDS):
        super().__init__()
        self.context_type_ids = context_type_ids
        self.hidden_dim = hidden_dim
        self.gnn = GATv2Stack(hidden_dim, num_layers, num_heads, edge_dim, dropout)
        in_dim_mlp = 2 * len(context_type_ids) * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim_mlp, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_fixed_dim),
        )

    def forward(self, h, edge_index, edge_attr, batch, node_type, batch_size):
        ctx_mask = node_type != C.VAR_ID
        ctx_nodes = ctx_mask.nonzero(as_tuple=False).squeeze(-1)

        # context 部分グラフ抽出（relabel）
        ctx_ei, ctx_ea = subgraph(
            subset=ctx_mask, edge_index=edge_index, edge_attr=edge_attr,
            relabel_nodes=True, num_nodes=h.size(0),
        )
        h_ctx = h[ctx_nodes]
        batch_ctx = batch[ctx_nodes]
        node_type_ctx = node_type[ctx_nodes]

        h_ctx = self.gnn(h_ctx, ctx_ei, ctx_ea)

        # 型別 pool して連結
        pool_list = []
        device = h_ctx.device
        for t_id in self.context_type_ids:
            t_mask = node_type_ctx == t_id
            if t_mask.any():
                ht = h_ctx[t_mask]
                bt = batch_ctx[t_mask]
                mean_p = global_mean_pool(ht, bt, size=batch_size)
                max_p = global_max_pool(ht, bt, size=batch_size)
            else:
                mean_p = torch.zeros(batch_size, self.hidden_dim, device=device)
                max_p = torch.zeros(batch_size, self.hidden_dim, device=device)
            pool_list.extend([mean_p, max_p])
        z_fixed = self.mlp(torch.cat(pool_list, dim=-1))

        context_info = {
            "h_context": h_ctx,            # [N_ctx, H]
            "batch_context": batch_ctx,    # [N_ctx]
        }
        return context_info, z_fixed


# --------------------------------------------------------------------------
# 共通: Transformer ブロック（Self/Cross 共用）
# --------------------------------------------------------------------------
class _AttnBlock(nn.Module):
    """MultiheadAttention + FFN + 残差LN を num_layers 段。
    self-attention（kv=None）と cross-attention（kv指定）の両対応。"""

    def __init__(self, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, q, kv=None, q_kpm=None, kv_kpm=None):
        x = q
        for attn, ffn, n1, n2 in zip(self.attns, self.ffns, self.norms1, self.norms2):
            key = value = (x if kv is None else kv)
            kpm = (q_kpm if kv is None else kv_kpm)
            a, _ = attn(x, key, value, key_padding_mask=kpm)
            x = n1(x + a)
            x = n2(x + ffn(x))
        return x


# --------------------------------------------------------------------------
# VariableBranch
# --------------------------------------------------------------------------
class VariableBranch(nn.Module):
    """可変ノード Self-Att -> Cross-Att(対 context) -> pool -> (mu, logvar)。GNN なし。"""

    def __init__(self, hidden_dim=64, z_dim=16,
                 num_self_layers=2, num_cross_layers=2,
                 num_heads=4, dropout=0.1,
                 max_var_nodes=10, max_context_nodes=50):
        super().__init__()
        self.max_var_nodes = max_var_nodes
        self.max_context_nodes = max_context_nodes
        self.self_block = _AttnBlock(hidden_dim, num_self_layers, num_heads, dropout)
        self.cross_block = _AttnBlock(hidden_dim, num_cross_layers, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

    def forward(self, h, batch, node_type, context_info, batch_size):
        var_mask = node_type == C.VAR_ID
        h_var = h[var_mask]
        batch_var = batch[var_mask]

        var_dense, var_m = to_dense_batch(
            h_var, batch_var, max_num_nodes=self.max_var_nodes, batch_size=batch_size)
        ctx_dense, ctx_m = to_dense_batch(
            context_info["h_context"], context_info["batch_context"],
            max_num_nodes=self.max_context_nodes, batch_size=batch_size)

        # Self-Att（可変同士）
        x = self.self_block(var_dense, kv=None, q_kpm=~var_m)
        # Cross-Att（Q=可変, K/V=context）
        x = self.cross_block(x, kv=ctx_dense, kv_kpm=~ctx_m)

        # mask 付き pool
        mask_f = var_m.unsqueeze(-1).float()
        mean_p = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        max_p = x.masked_fill(~var_m.unsqueeze(-1), float("-inf")).amax(1)
        mu, logvar = self.mlp(torch.cat([mean_p, max_p], dim=-1)).chunk(2, dim=-1)
        return mu, logvar
