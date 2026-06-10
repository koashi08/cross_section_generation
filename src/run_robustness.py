"""学習済みモデルの z ロバスト性チェック（可視化中心）。

使い方:
    python run_robustness.py --config configs/default.yaml
    python run_robustness.py --config configs/default.yaml --n-per-sigma 8 --device cuda

設定 YAML の paths（checkpoint_dir / processed_root）から best.pt とテスト
データを読み込み、4戦略の摂動生成を行って PNG を checkpoint_dir/robustness/
以下に保存する。妥当性チェックは有限性のみ。

出力:
    robustness/prior_ext{k}.png       戦略A: z ~ N(0,σ²)（σ 別グリッド）
    robustness/posterior_ext{k}.png   戦略B: mu 周り摂動（σ 別グリッド）
    robustness/traversal_zg_ext{k}.png  戦略C: z_global 次元トラバーサル
    robustness/traversal_zv_ext{k}.png  戦略C: z_var 次元トラバーサル
    robustness/interpolation.png      戦略D: 2サンプル間の補間
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from cross_section.config import load_config, model_kwargs_from
from cross_section.dataset import CrossSectionDataset
from cross_section.normalize import FeatureNormalizer
from cross_section.vae import CrossSectionVAE
from cross_section.robustness import (
    generate_variations, interpolate_z,
    plot_variation_grid, plot_traversal_rows,
)


def parse_args():
    p = argparse.ArgumentParser(description="z ロバスト性チェック")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-per-sigma", type=int, default=8,
                   help="σ ごとの生成数（戦略A/B）")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(cfg, device):
    """config と best.pt からモデルを復元する。"""
    model = CrossSectionVAE(**model_kwargs_from(cfg))
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"✓ モデル復元: {ckpt_path} (epoch={ckpt.get('epoch', '?')})")
    return model


def pick_samples_per_extremum(test_ds):
    """テストデータから極値数ごとに1サンプルずつ選ぶ。"""
    picked = {}
    for d in test_ds:
        k = int(d.extremum_count)
        if k not in picked:
            picked[k] = d
    return dict(sorted(picked.items()))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg["paths"]["checkpoint_dir"], "robustness")
    os.makedirs(out_dir, exist_ok=True)

    # データ・モデル
    test_ds = CrossSectionDataset(cfg["paths"]["processed_root"], "test")
    normalizer = FeatureNormalizer.load(
        os.path.join(cfg["paths"]["processed_root"], "normalizer.pt"))
    model = load_model(cfg, device)

    samples = pick_samples_per_extremum(test_ds)
    print(f"対象サンプル（極値数別）: {list(samples.keys())}")

    summary = []   # (戦略, 対象, valid_rate)

    # ----- 戦略A: prior サンプリング（σ を振る）-----
    print("\n[戦略A] prior サンプリング z ~ N(0, σ²)")
    sigmas_prior = [0.5, 1.0, 2.0]
    for ext, data in samples.items():
        results, labels = [], []
        for sg in sigmas_prior:
            r = generate_variations(model, data, n=args.n_per_sigma,
                                    sigma_g=sg, sigma_v=sg,
                                    center="prior", device=device,
                                    seed=args.seed)
            results.append(r)
            labels += [f"σ={sg} #{i}" for i in range(args.n_per_sigma)]
            summary.append((f"prior σ={sg}", f"ext{ext}", r["valid_rate"]))
        merged = _merge_results(results)
        path = os.path.join(out_dir, f"prior_ext{ext}.png")
        plot_variation_grid(merged, labels, path,
                            suptitle=f"prior sampling (ext={ext})",
                            ncols=args.n_per_sigma)
        print(f"  ext={ext}: {path}")

    # ----- 戦略B: posterior 周り摂動 -----
    print("\n[戦略B] posterior 周り摂動 z = mu + N(0, σ²)")
    sigmas_post = [0.1, 0.3, 0.5]
    for ext, data in samples.items():
        results, labels = [], []
        for sg in sigmas_post:
            r = generate_variations(model, data, n=args.n_per_sigma,
                                    sigma_g=sg, sigma_v=sg,
                                    center="posterior", device=device,
                                    seed=args.seed)
            results.append(r)
            labels += [f"σ={sg} #{i}" for i in range(args.n_per_sigma)]
            summary.append((f"posterior σ={sg}", f"ext{ext}", r["valid_rate"]))
        merged = _merge_results(results)
        path = os.path.join(out_dir, f"posterior_ext{ext}.png")
        plot_variation_grid(merged, labels, path,
                            suptitle=f"posterior perturbation (ext={ext})",
                            ncols=args.n_per_sigma)
        print(f"  ext={ext}: {path}")

    # ----- 戦略C: 次元トラバーサル -----
    print("\n[戦略C] 次元トラバーサル（基準=mu、1次元のみ置換）")
    trav_values = (-2.0, -1.0, 0.0, 1.0, 2.0)
    zg_dim = model_kwargs_from(cfg)["z_global_dim"]
    zv_dim = model_kwargs_from(cfg)["z_var_dim"]
    for ext, data in samples.items():
        p1 = os.path.join(out_dir, f"traversal_zg_ext{ext}.png")
        plot_traversal_rows(model, data, "z_global", list(range(zg_dim)),
                            trav_values, p1, device=device,
                            suptitle=f"z_global traversal (ext={ext})")
        p2 = os.path.join(out_dir, f"traversal_zv_ext{ext}.png")
        plot_traversal_rows(model, data, "z_var", list(range(zv_dim)),
                            trav_values, p2, device=device,
                            suptitle=f"z_var traversal (ext={ext})")
        print(f"  ext={ext}: {p1}, {p2}")

    # ----- 戦略D: 補間 -----
    print("\n[戦略D] 2サンプル間の z 補間")
    if len(test_ds) >= 2:
        ds_list = list(samples.values())
        da = ds_list[0]
        db = ds_list[-1] if len(ds_list) > 1 else test_ds[1]
        r = interpolate_z(model, da, db, steps=6, device=device)
        labels = [f"t={t:.2f}" for t in r["t"]]
        path = os.path.join(out_dir, "interpolation.png")
        plot_variation_grid(r, labels, path,
                            suptitle="z interpolation (context=sample A 固定)",
                            ncols=6, show_reference=False)
        summary.append(("interpolation", "A->B", r["valid_rate"]))
        print(f"  {path}")

    # ----- 有限チェックのサマリ -----
    print("\n" + "=" * 56)
    print("有限チェック サマリ（valid_rate = 有限ノードの割合）")
    print("=" * 56)
    all_ok = True
    for strat, target, rate in summary:
        mark = "✓" if rate == 1.0 else "⚠"
        if rate < 1.0:
            all_ok = False
        print(f"  {mark} {strat:<18} {target:<8} valid={rate*100:.1f}%")
    print("-" * 56)
    print("✓ 全生成で非有限なし" if all_ok else "⚠ 非有限の生成あり（該当 PNG を確認）")
    print(f"\n出力先: {out_dir}/")


def _merge_results(results):
    """複数の generate_variations 結果を1つに結合する（グリッド描画用）。"""
    from torch_geometric.data import Batch
    datas = []
    for r in results:
        datas += r["batch"].to_data_list()
    merged_batch = Batch.from_data_list(datas)
    pred = torch.cat([r["pred_pos"] for r in results], dim=0)
    valid = torch.cat([r["valid"] for r in results], dim=0)
    # var_nodes を結合後のノード番号にオフセット
    var_nodes_list, node_off, sample_off = [], 0, 0
    for r in results:
        var_nodes_list.append(r["var_nodes"] + node_off)
        node_off += r["batch"].num_nodes
    var_nodes = torch.cat(var_nodes_list, dim=0)
    return {"pred_pos": pred, "var_nodes": var_nodes,
            "batch": merged_batch, "valid": valid}


if __name__ == "__main__":
    main()
