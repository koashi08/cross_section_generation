"""CSV から Dataset 構築までの全パイプライン検証。"""

import os
import shutil

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from cross_section import constants as C
from cross_section.csv_export import generate_csv_dataset
from cross_section.parser import load_raw_samples
from cross_section.dataset import prepare_datasets
from cross_section.normalize import FeatureNormalizer

RAW = "/home/claude/_data_raw"
ROOT = "/home/claude/_data_proc"


def main():
    # クリーンスタート
    for d in (RAW, ROOT):
        if os.path.exists(d):
            shutil.rmtree(d)

    print("=" * 64)
    print("【1】合成 CSV 生成（新スキーマ）")
    print("=" * 64)
    n = generate_csv_dataset(RAW, n_per_extremum=500, seed=0)
    print(f"生成サンプル数: {n}")
    # CSV ヘッダー確認
    import pandas as pd
    for f in ("nodes.csv", "edges.csv", "graphs.csv"):
        df = pd.read_csv(os.path.join(RAW, f))
        print(f"  {f}: {list(df.columns)}  ({len(df)} 行)")

    print("\n" + "=" * 64)
    print("【2】CSV -> RawSample パース確認")
    print("=" * 64)
    samples = load_raw_samples(
        os.path.join(RAW, "nodes.csv"),
        os.path.join(RAW, "edges.csv"),
        os.path.join(RAW, "graphs.csv"),
    )
    print(f"RawSample 数: {len(samples)}")
    s0 = samples[0]
    print(f"graph_id=0: ノード{len(s0.nodes)}, エッジ{len(s0.edges)}, "
          f"極値{s0.extremum_count}, 閉={s0.section_type}, 伸張={s0.has_extension}")

    print("\n" + "=" * 64)
    print("【3】データセット準備（分割→正規化fit→変換→保存）")
    print("=" * 64)
    train_ds, val_ds, test_ds, normalizer = prepare_datasets(
        RAW, ROOT, ratios=(0.95, 0.04, 0.01), seed=0
    )

    print("\n" + "=" * 64)
    print("【4】正規化のリーク防止確認")
    print("=" * 64)
    # 訓練の標準化後特徴量は概ね平均0・分散1のはず
    train_x = torch.cat([d.x[:, C.NODE_NORM_COLS] for d in train_ds], dim=0)
    print(f"訓練 x (標準化列) mean: {train_x.mean(0).round(decimals=3).tolist()}")
    print(f"訓練 x (標準化列) std : {train_x.std(0).round(decimals=3).tolist()}")
    print("→ 平均≈0, 分散≈1 なら訓練統計で正しく標準化されている")
    print(f"\nオフセット統計（訓練から計算）:")
    print(f"  offset_mean: {normalizer.offset_mean.round(decimals=4).tolist()}")
    print(f"  offset_std : {normalizer.offset_std.round(decimals=4).tolist()}")

    print("\n" + "=" * 64)
    print("【5】DataLoader バッチ化確認")
    print("=" * 64)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    print(batch)
    bs = int(batch.batch.max()) + 1
    print(f"\nbatch_size: {bs}")
    print(f"x: {tuple(batch.x.shape)} (期待: [N, {C.NODE_FEATURE_DIM}])")
    print(f"edge_attr: {tuple(batch.edge_attr.shape)} (期待: [E, {C.EDGE_FEATURE_DIM}])")
    print(f"anchor: {tuple(batch.anchor.shape)} (Cartesian)")
    print(f"y: {tuple(batch.y.shape)} (Cartesian)")
    print(f"可変ノード総数: {int(batch.variable_mask.sum())}")

    # 形状アサート
    assert batch.x.shape[1] == C.NODE_FEATURE_DIM
    assert batch.edge_attr.shape[1] == C.EDGE_FEATURE_DIM

    print("\n" + "=" * 64)
    print("【6】オフセット標準化の動作確認")
    print("=" * 64)
    vm = batch.variable_mask
    offset = batch.y[vm] - batch.anchor[vm]              # Cartesian オフセット
    offset_norm = normalizer.normalize_offset(offset)
    recovered = normalizer.denormalize_offset(offset_norm)
    print(f"オフセット mean (生): {offset.mean(0).round(decimals=4).tolist()}")
    print(f"オフセット std  (生): {offset.std(0).round(decimals=4).tolist()}")
    print(f"標準化後 mean: {offset_norm.mean(0).round(decimals=4).tolist()}")
    print(f"標準化後 std : {offset_norm.std(0).round(decimals=4).tolist()}")
    assert torch.allclose(offset, recovered, atol=1e-4), "正規化の往復が一致しない"
    print("✓ オフセット正規化の往復 OK")

    print("\n" + "=" * 64)
    print("【7】NaN/範囲チェック")
    print("=" * 64)
    for ds_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for d in ds:
            for name in ["x", "edge_attr", "anchor", "y"]:
                t = getattr(d, name)
                assert torch.isfinite(t).all(), f"{ds_name} の {name} に NaN/Inf"
    print("✓ 全分割で NaN/Inf なし")

    print("\n" + "=" * 64)
    print("全パイプライン検証完了 ✓")
    print("=" * 64)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
