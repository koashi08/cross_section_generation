"""CrossSectionVAE 本体と損失関数。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C
from .encoder import Encoder
from .decoder import Decoder


class CrossSectionVAE(nn.Module):
    def __init__(
        self,
        in_dim: int = C.NODE_FEATURE_DIM,
        hidden_dim: int = 64,
        z_global_dim: int = 8,
        z_var_dim: int = 16,
        z_fixed_dim: int = 8,
        edge_dim: int = C.EDGE_FEATURE_DIM,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_var_nodes: int = 10,
        max_context_nodes: int = 50,
        use_anchor: bool = False,
        # 層数（必要に応じて調整）
        global_num_layers: int = 2,
        context_num_layers: int = 2,
        var_self_layers: int = 2,
        var_cross_layers: int = 2,
        decoder_context_gnn_layers: int = 2,
        slot_self_layers: int = 2,
        slot_cross_layers: int = 2,
        decoder_gnn_layers: int = 2,
    ):
        super().__init__()
        common = dict(
            in_dim=in_dim, hidden_dim=hidden_dim,
            z_global_dim=z_global_dim, z_var_dim=z_var_dim, z_fixed_dim=z_fixed_dim,
            edge_dim=edge_dim, num_heads=num_heads, dropout=dropout,
            max_var_nodes=max_var_nodes, max_context_nodes=max_context_nodes,
        )
        self.encoder = Encoder(
            global_num_layers=global_num_layers,
            context_num_layers=context_num_layers,
            var_self_layers=var_self_layers,
            var_cross_layers=var_cross_layers,
            **common,
        )
        self.decoder = Decoder(
            context_gnn_layers=decoder_context_gnn_layers,
            slot_self_layers=slot_self_layers,
            slot_cross_layers=slot_cross_layers,
            decoder_gnn_layers=decoder_gnn_layers,
            use_anchor=use_anchor,
            **common,
        )

    def forward(self, data):
        enc = self.encoder(data)
        dec = self.decoder(
            data,
            z_global=enc["z_global"],
            z_var=enc["z_var"],
            z_fixed=enc["z_fixed"],
            context_info=enc["context_info"],
            batch_size=enc["batch_size"],
        )
        return {**enc, **dec}

    @torch.no_grad()
    def generate(self, data, z_global=None, z_var=None):
        """推論: z をサンプリング（or 指定）して可変ノードを生成する。
        z_fixed は context から決定論的に計算する。"""
        self.eval()
        enc = self.encoder(data)          # z_fixed と context_info を得るため
        bs = enc["batch_size"]
        device = data.x.device
        if z_global is None:
            z_global = torch.randn(bs, enc["mu_global"].size(-1), device=device)
        if z_var is None:
            z_var = torch.randn(bs, enc["mu_var"].size(-1), device=device)
        dec = self.decoder(
            data, z_global=z_global, z_var=z_var, z_fixed=enc["z_fixed"],
            context_info=enc["context_info"], batch_size=bs)
        return dec


def vae_loss(
    outputs,
    data,
    normalizer=None,
    beta_global: float = 1.0,
    beta_var: float = 1.0,
):
    """VAE 損失。

    再構成損失は Cartesian。normalizer があれば標準化空間で MSE を取る
    （出力ヘッドの学習安定化）。z_fixed は決定論なので KL 対象外。
    """
    var_nodes = outputs["var_nodes"]
    pred_pos = outputs["pred_pos"]                 # [N_var, 2]
    target = data.y[var_nodes]                     # [N_var, 2]（Cartesian）
    anchor = outputs["anchor"]                     # [N_var, 2]

    if normalizer is not None:
        # 標準化空間で MSE（anchor=0 のときは y そのものの標準化）
        pred_off_norm = normalizer.normalize_offset(pred_pos - anchor)
        tgt_off_norm = normalizer.normalize_offset(target - anchor)
        recon = F.mse_loss(pred_off_norm, tgt_off_norm)
    else:
        recon = F.mse_loss(pred_pos, target)

    mu_g, lv_g = outputs["mu_global"], outputs["logvar_global"]
    mu_v, lv_v = outputs["mu_var"], outputs["logvar_var"]
    kl_g = -0.5 * torch.mean(1 + lv_g - mu_g.pow(2) - lv_g.exp())
    kl_v = -0.5 * torch.mean(1 + lv_v - mu_v.pow(2) - lv_v.exp())

    total = recon + beta_global * kl_g + beta_var * kl_v
    return total, {
        "recon": float(recon.detach()),
        "kl_global": float(kl_g.detach()),
        "kl_var": float(kl_v.detach()),
        "total": float(total.detach()),
    }
