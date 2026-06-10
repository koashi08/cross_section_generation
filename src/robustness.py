"""z ロバスト性チェック。

潜在変数 z_global / z_var に任意の摂動を与えて形状を生成し、
破綻（非有限・異常配置）がないかを可視化で確認するためのツール群。

4つの摂動戦略:
  A. prior サンプリング    : z ~ N(0, σ²)。σ を振って分布の端での挙動を見る
  B. posterior 周り摂動    : z = mu + N(0, σ²)。実在形状の近傍応答（最適化に直結）
  C. 次元トラバーサル      : 1次元だけ動かし、他は mu に固定。次元の役割を見る
  D. 補間                  : 2サンプルの mu を線形補間。経路上の滑らかさを見る

妥当性チェックは有限性（NaN/Inf なし）のみ（ユーザー要望）。
context（固定構造）は既存サンプルのものを固定し、z の影響だけを見る。
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from . import constants as C


# ==========================================================================
# 基盤: 複製バッチ・エンコード・生成
# ==========================================================================
def replicate_sample(data, n):
    """1サンプルを n 個複製したバッチを作る（同じ context で n 通りの z を試す）。"""
    return Batch.from_data_list([data] * n)


@torch.no_grad()
def _encode(model, batch):
    """エンコーダを呼び、mu と z_fixed を取得（決定論的な基準として mu を使う）。"""
    enc = model.encoder(batch)
    return enc["mu_global"], enc["mu_var"], enc["z_fixed"], enc["batch_size"]


@torch.no_grad()
def _decode(model, batch, z_g, z_v, z_fixed, bs):
    """指定した z でデコードし、生成結果を返す。"""
    dec = model.decoder(batch, z_global=z_g, z_var=z_v, z_fixed=z_fixed,
                        batch_size=bs)
    return dec


# ==========================================================================
# 妥当性チェック（有限性のみ）
# ==========================================================================
def validity_check(pred_pos):
    """各可変ノードの予測が有限（NaN/Inf なし）かを返す。

    Returns:
        valid: [N_var] bool（ノード単位）
        all_valid: bool（全ノード有限か）
    """
    valid = torch.isfinite(pred_pos).all(dim=-1)
    return valid, bool(valid.all())


# ==========================================================================
# 戦略A / B: σ サンプリング（prior / posterior 中心）
# ==========================================================================
@torch.no_grad()
def generate_variations(model, data, n=8, sigma_g=1.0, sigma_v=1.0,
                        center="prior", device="cpu", seed=None):
    """z をサンプリングして n 通りの形状を生成する。

    Args:
        center: "prior"     -> z ~ N(0, σ²)（戦略A）
                "posterior" -> z = mu + N(0, σ²)（戦略B、mu はこのサンプルのエンコード値）
        sigma_g, sigma_v: z_global / z_var の摂動スケール
        seed: 再現用シード（None なら現在の乱数状態）
    Returns:
        dict(pred_pos[N_var_total,2], var_nodes, batch, z_global, z_var, valid_rate)
    """
    model.eval()
    model.to(device)
    if seed is not None:
        torch.manual_seed(seed)

    batch = replicate_sample(data, n).to(device)
    mu_g, mu_v, z_fixed, bs = _encode(model, batch)   # 複製なので mu は全行同値

    noise_g = sigma_g * torch.randn_like(mu_g)
    noise_v = sigma_v * torch.randn_like(mu_v)
    if center == "prior":
        z_g, z_v = noise_g, noise_v
    elif center == "posterior":
        z_g, z_v = mu_g + noise_g, mu_v + noise_v
    else:
        raise ValueError(f"未知の center: {center}（prior / posterior）")

    dec = _decode(model, batch, z_g, z_v, z_fixed, bs)
    valid, _ = validity_check(dec["pred_pos"])
    return {
        "pred_pos": dec["pred_pos"], "var_nodes": dec["var_nodes"],
        "batch": batch, "z_global": z_g, "z_var": z_v,
        "valid": valid, "valid_rate": float(valid.float().mean()),
    }


# ==========================================================================
# 戦略C: 次元トラバーサル
# ==========================================================================
@torch.no_grad()
def latent_traversal(model, data, target="z_var", dim=0,
                     values=(-2.0, -1.0, 0.0, 1.0, 2.0), device="cpu"):
    """1次元だけを values の各値に置き換えて生成（他次元は mu に固定）。

    Args:
        target: "z_global" or "z_var"
        dim: 動かす次元
        values: その次元にセットする値の列
    Returns:
        generate_variations と同形式の dict（行 = values の各値）
    """
    model.eval()
    model.to(device)
    n = len(values)
    batch = replicate_sample(data, n).to(device)
    mu_g, mu_v, z_fixed, bs = _encode(model, batch)

    z_g, z_v = mu_g.clone(), mu_v.clone()
    vals = torch.tensor(values, dtype=torch.float, device=device)
    if target == "z_global":
        z_g[:, dim] = vals
    elif target == "z_var":
        z_v[:, dim] = vals
    else:
        raise ValueError(f"未知の target: {target}（z_global / z_var）")

    dec = _decode(model, batch, z_g, z_v, z_fixed, bs)
    valid, _ = validity_check(dec["pred_pos"])
    return {
        "pred_pos": dec["pred_pos"], "var_nodes": dec["var_nodes"],
        "batch": batch, "z_global": z_g, "z_var": z_v,
        "valid": valid, "valid_rate": float(valid.float().mean()),
    }


# ==========================================================================
# 戦略D: 補間
# ==========================================================================
@torch.no_grad()
def interpolate_z(model, data_a, data_b, steps=5, device="cpu"):
    """2サンプルの mu を線形補間して生成する。

    context（固定構造・anchor）は data_a のものに固定し、z だけを
    mu_A -> mu_B へ動かす（条件を固定して z の経路だけを見る）。
    Returns:
        generate_variations と同形式の dict（行 = t=0..1 の各ステップ）
    """
    model.eval()
    model.to(device)

    # 端点の mu を取得
    ba = Batch.from_data_list([data_a]).to(device)
    bb = Batch.from_data_list([data_b]).to(device)
    mu_g_a, mu_v_a, _, _ = _encode(model, ba)
    mu_g_b, mu_v_b, _, _ = _encode(model, bb)

    # context は data_a 固定で steps 個複製
    batch = replicate_sample(data_a, steps).to(device)
    _, _, z_fixed, bs = _encode(model, batch)

    t = torch.linspace(0, 1, steps, device=device).unsqueeze(-1)   # [steps,1]
    z_g = (1 - t) * mu_g_a + t * mu_g_b   # [steps, zg_dim]（ブロードキャスト）
    z_v = (1 - t) * mu_v_a + t * mu_v_b

    dec = _decode(model, batch, z_g, z_v, z_fixed, bs)
    valid, _ = validity_check(dec["pred_pos"])
    return {
        "pred_pos": dec["pred_pos"], "var_nodes": dec["var_nodes"],
        "batch": batch, "z_global": z_g, "z_var": z_v,
        "valid": valid, "valid_rate": float(valid.float().mean()),
        "t": t.view(-1).tolist(),
    }


# ==========================================================================
# 可視化
# ==========================================================================
def plot_variation_grid(result, labels, savepath, suptitle="", ncols=4,
                        show_reference=True):
    """生成結果をグリッド描画する。

    各セル: context（type 別色分け）+ 参照（元の可変ノード位置, 薄赤）
            + 生成された可変ノード（黒×）。非有限ノードはタイトルに警告。

    Args:
        result: generate_variations / latent_traversal / interpolate_z の返り値
        labels: 各セルのタイトル（生成数と同じ長さ）
        show_reference: 元サンプルの可変ノード正解位置（data.y）を薄く表示するか
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch = result["batch"]
    pred = result["pred_pos"].detach().cpu().numpy()
    var_nodes = result["var_nodes"].cpu().numpy()
    valid = result["valid"].cpu().numpy()
    b = batch.batch.cpu().numpy()
    cart = batch.pos_cart.cpu().numpy()
    ntype = batch.node_type.cpu().numpy()
    y = batch.y.cpu().numpy()
    vmask = batch.variable_mask.cpu().numpy()

    n = len(labels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    type_colors = [(C.NODE_TYPE["fixed"], "#1f77b4", "fixed"),
                   (C.NODE_TYPE["extension"], "#2ca02c", "ext"),
                   (C.NODE_TYPE["dependent"], "#ff7f0e", "dep")]

    for si in range(n):
        ax = axes[si]
        idx = np.where(b == si)[0]
        # context（可変以外）
        for t, color, label in type_colors:
            sel = idx[(ntype[idx] == t)]
            if sel.size:
                ax.scatter(cart[sel, 0], cart[sel, 1], c=color, s=30,
                           zorder=3, edgecolors="white", linewidths=0.5,
                           label=label)
        # 参照: 元の可変ノード位置（薄赤）
        if show_reference:
            ref_sel = idx[vmask[idx]]
            if ref_sel.size:
                ax.scatter(y[ref_sel, 0], y[ref_sel, 1], c="#d62728", s=25,
                           alpha=0.30, zorder=2, label="ref")
        # 生成された可変ノード（黒×、非有限は描けないので除外）
        gen_idx = [j for j, vn in enumerate(var_nodes) if b[vn] == si]
        n_invalid = 0
        if gen_idx:
            gp = pred[gen_idx]
            gv = valid[gen_idx]
            n_invalid = int((~gv).sum())
            gp_ok = gp[gv]
            if gp_ok.size:
                ax.scatter(gp_ok[:, 0], gp_ok[:, 1], marker="x", c="black",
                           s=55, zorder=4, label="gen")
        title = labels[si]
        if n_invalid > 0:
            title += f"  ⚠非有限{n_invalid}点"
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        if si == 0:
            ax.legend(fontsize=6, loc="upper right")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if suptitle else None)
    fig.savefig(savepath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return savepath


def plot_traversal_rows(model, data, target, dims, values, savepath,
                        device="cpu", suptitle=""):
    """複数次元のトラバーサルを「行=次元、列=値」で1枚に描画する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = len(dims), len(values)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.6 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes)

    type_colors = [(C.NODE_TYPE["fixed"], "#1f77b4"),
                   (C.NODE_TYPE["extension"], "#2ca02c"),
                   (C.NODE_TYPE["dependent"], "#ff7f0e")]

    total_invalid = 0
    for r, dim in enumerate(dims):
        res = latent_traversal(model, data, target=target, dim=dim,
                               values=values, device=device)
        pred = res["pred_pos"].detach().cpu().numpy()
        var_nodes = res["var_nodes"].cpu().numpy()
        valid = res["valid"].cpu().numpy()
        batch = res["batch"]
        b = batch.batch.cpu().numpy()
        cart = batch.pos_cart.cpu().numpy()
        ntype = batch.node_type.cpu().numpy()

        for c in range(ncols):
            ax = axes[r, c]
            idx = np.where(b == c)[0]
            for t, color in type_colors:
                sel = idx[(ntype[idx] == t)]
                if sel.size:
                    ax.scatter(cart[sel, 0], cart[sel, 1], c=color, s=18, zorder=3)
            gen_idx = [j for j, vn in enumerate(var_nodes) if b[vn] == c]
            if gen_idx:
                gp = pred[gen_idx]
                gv = valid[gen_idx]
                total_invalid += int((~gv).sum())
                gp_ok = gp[gv]
                if gp_ok.size:
                    ax.scatter(gp_ok[:, 0], gp_ok[:, 1], marker="x",
                               c="black", s=35, zorder=4)
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{values[c]:+.1f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"dim {dim}", fontsize=9)

    head = suptitle or f"{target} traversal"
    if total_invalid > 0:
        head += f"  ⚠非有限{total_invalid}点"
    fig.suptitle(head, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(savepath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return savepath
