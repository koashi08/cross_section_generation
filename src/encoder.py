"""エンコーダ本体。SharedNodeEmbedding -> 3ブランチ -> reparameterize。"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import constants as C
from .encoder_modules import (
    SharedNodeEmbedding, GlobalBranch, ContextBranch, VariableBranch,
)


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int = C.NODE_FEATURE_DIM,
        hidden_dim: int = 64,
        z_global_dim: int = 8,
        z_var_dim: int = 16,
        z_fixed_dim: int = 8,
        edge_dim: int = C.EDGE_FEATURE_DIM,
        global_num_layers: int = 2,
        context_num_layers: int = 2,
        var_self_layers: int = 2,
        var_cross_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_var_nodes: int = 10,
        max_context_nodes: int = 50,
    ):
        super().__init__()
        self.shared_embed = SharedNodeEmbedding(in_dim, hidden_dim)
        self.global_branch = GlobalBranch(
            hidden_dim, z_global_dim, global_num_layers, num_heads, edge_dim, dropout)
        self.context_branch = ContextBranch(
            hidden_dim, z_fixed_dim, context_num_layers, num_heads, edge_dim, dropout)
        self.variable_branch = VariableBranch(
            hidden_dim, z_var_dim, var_self_layers, var_cross_layers,
            num_heads, dropout, max_var_nodes, max_context_nodes)

    def forward(self, data):
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        batch_size = int(batch.max()) + 1

        # 1. SharedNodeEmbedding
        h = self.shared_embed(data.x, data.edge_index, data.edge_attr)

        # 2. GlobalBranch（h を破壊しないようコピーを渡す）
        mu_g, lv_g = self.global_branch(
            h.clone(), data.edge_index, data.edge_attr, batch)

        # 3. ContextBranch（中間ノード埋め込み + z_fixed）
        context_info, z_fixed = self.context_branch(
            h, data.edge_index, data.edge_attr, batch, data.node_type, batch_size)

        # 4. VariableBranch
        mu_v, lv_v = self.variable_branch(
            h, batch, data.node_type, context_info, batch_size)

        # 5. Reparameterize（z_fixed は決定論なので対象外）
        z_global = self._reparam(mu_g, lv_g)
        z_var = self._reparam(mu_v, lv_v)

        return {
            "z_global": z_global, "mu_global": mu_g, "logvar_global": lv_g,
            "z_var": z_var, "mu_var": mu_v, "logvar_var": lv_v,
            "z_fixed": z_fixed,
            "context_info": context_info,
            "batch_size": batch_size,
        }

    @staticmethod
    def _reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
