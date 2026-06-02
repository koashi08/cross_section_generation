"""断面形状 VAE の学習エントリーポイント（組み立てスクリプト）。

使い方:
    python run_train.py

設定は下記の CONFIG セクションを編集する。実データ投入時は RAW_DIR を
実データの CSV ディレクトリに向け、必要なら parser.py の
NODE_TYPE_CODE_MAP / EDGE_TYPE_CODE_MAP を実コードに合わせること。

処理の流れ:
    1. 実データスキーマの事前チェック（切り捨て・型コード）
    2. データ準備（分割 -> 正規化 fit -> 変換 -> 保存）
    3. モデル構築
    4. 学習（Trainer）
    5. best モデルでテスト評価
    6. 生成サンプルの可視化（任意）
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from cross_section import constants as C
from cross_section.dataset import prepare_datasets
from cross_section.vae import CrossSectionVAE
from cross_section.train import Trainer, TrainConfig
from cross_section.metrics import reconstruction_metrics


# ==========================================================================
# CONFIG（ここを編集）
# ==========================================================================
# --- パス ---
RAW_DIR = "./data_raw"               # nodes.csv, edges.csv, graphs.csv がある場所
PROCESSED_ROOT = "./data_processed"  # InMemoryDataset / 統計 / 分割の保存先
CHECKPOINT_DIR = "./checkpoints"     # モデル・履歴の保存先

# --- データ ---
SPLIT_RATIOS = (0.95, 0.04, 0.01)
SEED = 0
ANCHOR_MODE = "zero"                 # "zero"=直接予測（当面）

# --- ノード数上限（実データ最大 8/14 に余裕 +2）---
MAX_VAR_NODES = 10
MAX_CONTEXT_NODES = 16

# --- モデル ---
HIDDEN_DIM = 64
Z_GLOBAL_DIM = 8
Z_VAR_DIM = 16
Z_FIXED_DIM = 8
NUM_HEADS = 4
DROPOUT = 0.1
USE_ANCHOR = False                   # 直接予測（当面）

# --- 学習 ---
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 64
WARMUP_EPOCHS = 10
BETA_GLOBAL_MAX = 1.0
BETA_VAR_MAX = 1.0
FREE_BITS = 0.5                      # posterior collapse 対策（必要に応じて 0 に）
GRAD_CLIP = 1.0
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 可視化 ---
VISUALIZE_SAMPLES = True
N_VIS_SAMPLES = 8


# ==========================================================================
# Step 1: 実データスキーマの事前チェック
# ==========================================================================
def precheck_schema(raw_dir):
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
    print(f"可変ノード最大: {var_max} (上限 {MAX_VAR_NODES})")
    print(f"context ノード最大: {ctx_max} (上限 {MAX_CONTEXT_NODES})")

    if var_max > MAX_VAR_NODES:
        raise ValueError(
            f"可変ノード最大 {var_max} が MAX_VAR_NODES={MAX_VAR_NODES} を超過。"
            f" to_dense_batch で切り捨てが起きます。MAX_VAR_NODES を上げてください。")
    if ctx_max > MAX_CONTEXT_NODES:
        raise ValueError(
            f"context ノード最大 {ctx_max} が MAX_CONTEXT_NODES={MAX_CONTEXT_NODES} を超過。"
            f" MAX_CONTEXT_NODES を上げてください。")
    print("✓ ノード数上限内（切り捨てなし）")

    # --- kappa スケールの目安を表示（無次元化の確認用）---
    kappa = edges_df["kappa_abs_mean"]
    print(f"kappa_abs_mean: min={kappa.min():.4g}, "
          f"max={kappa.max():.4g}, mean={kappa.mean():.4g}")
    print("  ※ 生の曲率(1/length)想定で conversion.py が × l_ref している。")
    print("    値が極端な場合はスケール調整を検討。")

    print(f"\nサンプル数: {graphs_df['graph_id'].nunique()}")
    return True


# ==========================================================================
# Step 5: テスト評価
# ==========================================================================
@torch.no_grad()
def evaluate_test(model, test_ds, normalizer, device):
    print("\n" + "=" * 64)
    print("【テスト評価】best モデル")
    print("=" * 64)
    model.eval()
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    agg = {"euclid": 0.0, "mse": 0.0, "n": 0}
    # 極値数別の誤差も集計
    per_ext = {}
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        m = reconstruction_metrics(out, batch, normalizer=normalizer)
        bs = int(batch.batch.max()) + 1
        agg["euclid"] += m["mean_euclidean"] * bs
        agg["mse"] += m["mse_real"] * bs
        agg["n"] += bs
        # 極値数別（バッチ内で集計）
        for ext in batch.extremum_count.view(-1).tolist():
            per_ext.setdefault(ext, 0)
            per_ext[ext] += 1

    n = max(agg["n"], 1)
    print(f"テスト 実スケール平均ユークリッド誤差: {agg['euclid'] / n:.4f}")
    print(f"テスト 実スケール MSE: {agg['mse'] / n:.4f}")
    print(f"テストサンプル数（極値数別）: {dict(sorted(per_ext.items()))}")
    return agg["euclid"] / n


# ==========================================================================
# Step 6: 生成サンプルの可視化
# ==========================================================================
@torch.no_grad()
def visualize_reconstructions(model, test_ds, normalizer, device, n_samples, savepath):
    """再構成（μ を使った決定論的予測）と元形状を比較描画。"""
    import matplotlib.pyplot as plt
    from cross_section import geometry as G

    model.eval()
    samples = [test_ds[i] for i in range(min(n_samples, len(test_ds)))]
    loader = DataLoader(samples, batch_size=len(samples), shuffle=False)
    batch = next(iter(loader)).to(device)

    out = model(batch)
    pred = out["pred_pos"].cpu().numpy()      # [N_var, 2]（正規化座標）
    var_nodes = out["var_nodes"].cpu().numpy()
    b = batch.batch.cpu().numpy()
    l_ref = batch.l_ref.view(-1).cpu().numpy()

    ncols = 4
    nrows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for si in range(len(samples)):
        ax = axes[si]
        node_mask = (b == si)
        # 元形状（全ノード、正規化座標 x/l_ref, y/l_ref を pos_cart に持っている）
        cart = batch.pos_cart.cpu().numpy()
        ntype = batch.node_type.cpu().numpy()
        idx = np.where(node_mask)[0]
        # context と可変を色分け
        for t, color in [(C.NODE_TYPE["fixed"], "#1f77b4"),
                         (C.NODE_TYPE["variable"], "#d62728"),
                         (C.NODE_TYPE["extension"], "#2ca02c"),
                         (C.NODE_TYPE["dependent"], "#ff7f0e")]:
            sel = idx[ntype[idx] == t]
            if sel.size:
                ax.scatter(cart[sel, 0], cart[sel, 1], c=color, s=40,
                           label=C.NODE_TYPE_INV[t], zorder=3, edgecolors="white", linewidths=0.5)
        # 予測（可変ノードのみ）
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
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Step 1: 事前チェック
    precheck_schema(RAW_DIR)

    # Step 2: データ準備
    print("\n" + "=" * 64)
    print("【データ準備】")
    print("=" * 64)
    train_ds, val_ds, test_ds, normalizer = prepare_datasets(
        raw_dir=RAW_DIR,
        processed_root=PROCESSED_ROOT,
        ratios=SPLIT_RATIOS,
        seed=SEED,
        anchor_mode=ANCHOR_MODE,
    )

    # Step 3: モデル構築
    print("\n" + "=" * 64)
    print("【モデル構築】")
    print("=" * 64)
    model = CrossSectionVAE(
        in_dim=C.NODE_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        z_global_dim=Z_GLOBAL_DIM,
        z_var_dim=Z_VAR_DIM,
        z_fixed_dim=Z_FIXED_DIM,
        edge_dim=C.EDGE_FEATURE_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_var_nodes=MAX_VAR_NODES,
        max_context_nodes=MAX_CONTEXT_NODES,
        use_anchor=USE_ANCHOR,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {n_params:,}")
    print(f"device: {DEVICE}")

    # Step 4: 学習
    print("\n" + "=" * 64)
    print("【学習】")
    print("=" * 64)
    config = TrainConfig(
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
        warmup_epochs=WARMUP_EPOCHS,
        beta_global_max=BETA_GLOBAL_MAX, beta_var_max=BETA_VAR_MAX,
        free_bits=FREE_BITS, grad_clip=GRAD_CLIP, patience=PATIENCE,
        checkpoint_dir=CHECKPOINT_DIR, device=DEVICE, log_every=1,
    )
    trainer = Trainer(model, train_ds, val_ds, normalizer, config)
    trainer.train()

    # Step 5: best モデルでテスト評価
    trainer.load_best()
    evaluate_test(model, test_ds, normalizer, DEVICE)

    # Step 6: 生成サンプルの可視化
    if VISUALIZE_SAMPLES:
        print("\n" + "=" * 64)
        print("【再構成の可視化】")
        print("=" * 64)
        visualize_reconstructions(
            model, test_ds, normalizer, DEVICE, N_VIS_SAMPLES,
            os.path.join(CHECKPOINT_DIR, "reconstructions.png"))

    print("\n" + "=" * 64)
    print("完了 ✓")
    print(f"  チェックポイント: {os.path.join(CHECKPOINT_DIR, 'best.pt')}")
    print(f"  学習履歴: {os.path.join(CHECKPOINT_DIR, 'history.csv')}")
    print("=" * 64)


if __name__ == "__main__":
    main()
