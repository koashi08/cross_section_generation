"""断面形状 VAE の学習エントリーポイント（YAML 設定対応）。

使い方:
    python run_train.py --config configs/default.yaml
    python run_train.py --config configs/exp_001.yaml --epochs 5   # 一部上書き

処理の流れ:
    1. 設定読み込み（YAML）
    2. 実データスキーマの事前チェック（切り捨て・型コード）
    3. データ準備（分割 -> 正規化 fit -> 変換 -> 保存）
    4. モデル構築
    5. 学習（Trainer）
    6. best モデルでテスト評価
    7. 生成サンプルの可視化（任意）
    8. 使用した設定を出力先に保存（再現性）
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from cross_section import constants as C
from cross_section.config import (
    load_config, to_train_config, model_kwargs_from, save_config,
)
from cross_section.dataset import prepare_datasets
from cross_section.vae import CrossSectionVAE
from cross_section.train import Trainer
from cross_section.metrics import reconstruction_metrics


# ==========================================================================
# Step 2: 実データスキーマの事前チェック
# ==========================================================================
def precheck_schema(raw_dir, max_var_nodes, max_context_nodes):
    """切り捨て・型コードの事前チェック。問題があれば早期に気づけるようにする。"""
    print("=" * 64)
    print("【事前チェック】実データスキーマ")
    print("=" * 64)

    nodes_path = os.path.join(raw_dir, "nodes.csv")
    edges_path = os.path.join(raw_dir, "edges.csv")
    graphs_path = os.path.join(raw_dir, "graphs.csv")
    for p in (nodes_path, edges_path, graphs_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV が見つかりません: {p}")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    graphs_df = pd.read_csv(graphs_path)

    # --- 型コードの値域を確認（マッピング漏れ検出）---
    node_codes = sorted(nodes_df["node_type_code"].unique().tolist())
    edge_codes = sorted(edges_df["edge_type_code"].unique().tolist())
    print(f"node_type_code の値域: {node_codes}")
    print(f"edge_type_code の値域: {edge_codes}")

    from cross_section.parser import NODE_TYPE_CODE_MAP, EDGE_TYPE_CODE_MAP
    unmapped_n = [c for c in node_codes if c not in NODE_TYPE_CODE_MAP]
    unmapped_e = [c for c in edge_codes if c not in EDGE_TYPE_CODE_MAP]
    if unmapped_n:
        raise ValueError(
            f"未対応の node_type_code: {unmapped_n}。"
            f" parser.py の NODE_TYPE_CODE_MAP を実コードに合わせてください。")
    if unmapped_e:
        raise ValueError(
            f"未対応の edge_type_code: {unmapped_e}。"
            f" parser.py の EDGE_TYPE_CODE_MAP を実コードに合わせてください。")
    print("✓ 型コードのマッピングに漏れなし")

    # --- ノード数上限の確認（切り捨て検出）---
    counts = (nodes_df.groupby(["graph_id", "node_type_code"]).size()
              .reset_index(name="n"))
    var_code = None
    for code, internal_id in NODE_TYPE_CODE_MAP.items():
        if internal_id == C.VAR_ID:
            var_code = code
            break
    var_max = counts[counts.node_type_code == var_code].n.max()
    ctx_max = (counts[counts.node_type_code != var_code]
               .groupby("graph_id").n.sum().max())
    print(f"可変ノード最大: {var_max} (上限 {max_var_nodes})")
    print(f"context ノード最大: {ctx_max} (上限 {max_context_nodes})")

    if var_max > max_var_nodes:
        raise ValueError(
            f"可変ノード最大 {var_max} が max_var_nodes={max_var_nodes} を超過。"
            f" to_dense_batch で切り捨てが起きます。model.max_var_nodes を上げてください。")
    if ctx_max > max_context_nodes:
        raise ValueError(
            f"context ノード最大 {ctx_max} が max_context_nodes={max_context_nodes} を超過。"
            f" model.max_context_nodes を上げてください。")
    print("✓ ノード数上限内（切り捨てなし）")

    # --- kappa スケールの目安 ---
    kappa = edges_df["kappa_abs_mean"]
    print(f"kappa_abs_mean: min={kappa.min():.4g}, "
          f"max={kappa.max():.4g}, mean={kappa.mean():.4g}")
    print(f"\nサンプル数: {graphs_df['graph_id'].nunique()}")
    return True


# ==========================================================================
# Step 6: テスト評価
# ==========================================================================
@torch.no_grad()
def evaluate_test(model, test_ds, normalizer, device, batch_size):
    print("\n" + "=" * 64)
    print("【テスト評価】best モデル")
    print("=" * 64)
    model.eval()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    agg = {"euclid": 0.0, "mse": 0.0, "n": 0}
    per_ext = {}
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        m = reconstruction_metrics(out, batch, normalizer=normalizer)
        bs = int(batch.batch.max()) + 1
        agg["euclid"] += m["mean_euclidean"] * bs
        agg["mse"] += m["mse_real"] * bs
        agg["n"] += bs
        for ext in batch.extremum_count.view(-1).tolist():
            per_ext[ext] = per_ext.get(ext, 0) + 1

    n = max(agg["n"], 1)
    print(f"テスト 実スケール平均ユークリッド誤差: {agg['euclid'] / n:.4f}")
    print(f"テスト 実スケール MSE: {agg['mse'] / n:.4f}")
    print(f"テストサンプル数（極値数別）: {dict(sorted(per_ext.items()))}")
    return agg["euclid"] / n


# ==========================================================================
# Step 7: 生成サンプルの可視化
# ==========================================================================
@torch.no_grad()
def visualize_reconstructions(model, test_ds, normalizer, device, n_samples, savepath):
    import matplotlib.pyplot as plt

    model.eval()
    samples = [test_ds[i] for i in range(min(n_samples, len(test_ds)))]
    batch = next(iter(DataLoader(samples, batch_size=len(samples), shuffle=False))).to(device)

    out = model(batch)
    pred = out["pred_pos"].cpu().numpy()
    var_nodes = out["var_nodes"].cpu().numpy()
    b = batch.batch.cpu().numpy()
    cart = batch.pos_cart.cpu().numpy()
    ntype = batch.node_type.cpu().numpy()

    ncols = 4
    nrows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for si in range(len(samples)):
        ax = axes[si]
        idx = np.where(b == si)[0]
        for t, color in [(C.NODE_TYPE["fixed"], "#1f77b4"),
                         (C.NODE_TYPE["variable"], "#d62728"),
                         (C.NODE_TYPE["extension"], "#2ca02c"),
                         (C.NODE_TYPE["dependent"], "#ff7f0e")]:
            sel = idx[ntype[idx] == t]
            if sel.size:
                ax.scatter(cart[sel, 0], cart[sel, 1], c=color, s=40,
                           label=C.NODE_TYPE_INV[t], zorder=3,
                           edgecolors="white", linewidths=0.5)
        var_in_sample = [j for j, vn in enumerate(var_nodes) if b[vn] == si]
        if var_in_sample:
            pv = pred[var_in_sample]
            ax.scatter(pv[:, 0], pv[:, 1], marker="x", c="black", s=60,
                       label="pred", zorder=4)
        ax.set_aspect("equal")
        ax.set_title(f"sample {si} (ext={int(batch.extremum_count[si])})", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.2)
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"✓ 再構成比較を {savepath} に保存")


# ==========================================================================
# main
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="断面形状 VAE の学習")
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="YAML 設定ファイルのパス")
    # よく上書きする項目だけ CLI でも受ける（任意）
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--raw-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def build_overrides(args) -> dict:
    """CLI 引数を YAML 上書き用の dict に変換（指定されたものだけ）。"""
    ov = {}
    if args.epochs is not None:
        ov.setdefault("training", {})["epochs"] = args.epochs
    if args.device is not None:
        ov.setdefault("training", {})["device"] = args.device
    if args.raw_dir is not None:
        ov.setdefault("paths", {})["raw_dir"] = args.raw_dir
    return ov


def main():
    args = parse_args()

    # Step 1: 設定読み込み
    cfg = load_config(args.config, overrides=build_overrides(args))
    seed = cfg["data"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_cfg = to_train_config(cfg)
    model_kwargs = model_kwargs_from(cfg)
    paths = cfg["paths"]

    print(f"設定ファイル: {args.config}")
    print(f"device: {train_cfg.device}")

    # Step 2: 事前チェック
    precheck_schema(paths["raw_dir"],
                    model_kwargs["max_var_nodes"],
                    model_kwargs["max_context_nodes"])

    # Step 3: データ準備
    print("\n" + "=" * 64)
    print("【データ準備】")
    print("=" * 64)
    train_ds, val_ds, test_ds, normalizer = prepare_datasets(
        raw_dir=paths["raw_dir"],
        processed_root=paths["processed_root"],
        ratios=tuple(cfg["data"]["ratios"]),
        seed=seed,
        anchor_mode=cfg["data"]["anchor_mode"],
    )

    # Step 4: モデル構築
    print("\n" + "=" * 64)
    print("【モデル構築】")
    print("=" * 64)
    model = CrossSectionVAE(**model_kwargs)
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # Step 5: 学習
    print("\n" + "=" * 64)
    print("【学習】")
    print("=" * 64)
    trainer = Trainer(model, train_ds, val_ds, normalizer, train_cfg)
    trainer.train()

    # Step 6: テスト評価
    trainer.load_best()
    evaluate_test(model, test_ds, normalizer, train_cfg.device, train_cfg.batch_size)

    # Step 7: 可視化
    if cfg.get("misc", {}).get("visualize_samples", False):
        print("\n" + "=" * 64)
        print("【再構成の可視化】")
        print("=" * 64)
        visualize_reconstructions(
            model, test_ds, normalizer, train_cfg.device,
            cfg["misc"].get("n_vis_samples", 8),
            os.path.join(paths["checkpoint_dir"], "reconstructions.png"))

    # Step 8: 使用した設定を保存（再現性）
    save_config(cfg, os.path.join(paths["checkpoint_dir"], "config_used.yaml"))
    print("\n" + "=" * 64)
    print("完了 ✓")
    print(f"  チェックポイント: {os.path.join(paths['checkpoint_dir'], 'best.pt')}")
    print(f"  学習履歴: {os.path.join(paths['checkpoint_dir'], 'history.csv')}")
    print(f"  使用設定: {os.path.join(paths['checkpoint_dir'], 'config_used.yaml')}")
    print("=" * 64)


if __name__ == "__main__":
    main()
