"""直接予測モード（anchor_mode="zero"）の検証。

確認項目:
  - anchor が全ゼロ
  - offset 統計 == y そのものの統計（可変ノードのみマスク）
  - pred_pos = anchor(0) + offset = offset が絶対座標予測になる
  - y の標準化が可変ノードのみで計算されている
"""

import os
import shutil

import numpy as np
import torch

from cross_section import constants as C
from cross_section.csv_export import generate_csv_dataset
from cross_section.dataset import prepare_datasets

RAW = "/home/claude/_data_raw_zero"
ROOT = "/home/claude/_data_proc_zero"


def main():
    for d in (RAW, ROOT):
        if os.path.exists(d):
            shutil.rmtree(d)

    print("=" * 64)
    print("【1】合成 CSV 生成 + 直接予測モードでデータセット準備")
    print("=" * 64)
    generate_csv_dataset(RAW, n_per_extremum=500, seed=0)
    train_ds, val_ds, test_ds, normalizer = prepare_datasets(
        RAW, ROOT, ratios=(0.95, 0.04, 0.01), seed=0, anchor_mode="zero"
    )

    print("\n" + "=" * 64)
    print("【2】anchor が全ゼロか確認")
    print("=" * 64)
    sample = train_ds[0]
    print(f"anchor shape: {tuple(sample.anchor.shape)}")
    print(f"anchor の絶対値合計: {sample.anchor.abs().sum().item():.6f}")
    all_zero = all((d.anchor.abs().sum().item() == 0) for d in train_ds)
    assert all_zero, "anchor がゼロでないサンプルがある"
    print("✓ 全サンプルで anchor が完全にゼロ")

    print("\n" + "=" * 64)
    print("【3】offset 統計 == y の統計（可変ノードのみ）か確認")
    print("=" * 64)
    # normalizer の offset 統計
    print(f"normalizer.offset_mean: {normalizer.offset_mean.tolist()}")
    print(f"normalizer.offset_std : {normalizer.offset_std.tolist()}")

    # 独立に「可変ノードの y」の統計を計算して照合
    ys = []
    for d in train_ds:
        vm = d.variable_mask
        # 注意: 正規化適用後の data なので、生の y を再計算する必要はなく
        #       d.y は標準化されていない（normalizer.transform は x, edge_attr のみ変更）
        ys.append(d.y[vm])
    ys = torch.cat(ys, dim=0)
    y_mean = ys.mean(0)
    y_std = ys.std(0).clamp(min=1e-6)
    print(f"\n独立計算した y[variable] mean: {y_mean.tolist()}")
    print(f"独立計算した y[variable] std : {y_std.tolist()}")

    # anchor=0 なので offset = y - 0 = y のはず → 統計が一致
    assert torch.allclose(normalizer.offset_mean, y_mean, atol=1e-4), \
        "offset_mean が y の統計と一致しない"
    assert torch.allclose(normalizer.offset_std, y_std, atol=1e-4), \
        "offset_std が y の統計と一致しない"
    print("✓ anchor=0 なので offset 統計 == 可変ノード y の統計（一致確認）")

    print("\n" + "=" * 64)
    print("【4】y 標準化が可変ノードのみで行われているか確認")
    print("=" * 64)
    # 全ノードの y で統計を取ると、context のゼロが混入して std が小さくなるはず
    ys_allnodes = torch.cat([d.y for d in train_ds], dim=0)   # 全ノード（ゼロ含む）
    y_std_allnodes = ys_allnodes.std(0)
    print(f"可変ノードのみ y std : {y_std.tolist()}")
    print(f"全ノード y std (誤): {y_std_allnodes.tolist()}")
    print("→ 全ノードだとゼロ混入で std が小さく歪む。可変のみが正しい")
    assert (y_std > y_std_allnodes).all(), "マスクの効果が確認できない"
    print("✓ 可変ノードマスクが正しく効いている（全ノードより std が大きい）")

    print("\n" + "=" * 64)
    print("【5】直接予測のシミュレーション（pred_pos = anchor + offset）")
    print("=" * 64)
    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    batch = next(iter(loader))
    vm = batch.variable_mask

    # デコーダ出力を模擬（ランダムな offset）
    n_var = int(vm.sum())
    fake_offset = torch.randn(n_var, 2)

    anchor_var = batch.anchor[vm]          # 全ゼロのはず
    pred_pos = anchor_var + fake_offset    # = fake_offset（anchor=0）
    assert torch.allclose(pred_pos, fake_offset), "anchor=0 で pred_pos == offset にならない"
    print(f"anchor[variable] の絶対値合計: {anchor_var.abs().sum().item():.6f}")
    print("✓ anchor=0 のとき pred_pos = offset（=直接座標予測）として機能")

    print("\n" + "=" * 64)
    print("【6】損失計算の流れ確認（標準化込み）")
    print("=" * 64)
    target = batch.y[vm]                                    # 可変ノードのターゲット
    target_offset = target - anchor_var                     # = target（anchor=0）
    target_offset_norm = normalizer.normalize_offset(target_offset)
    pred_offset_norm = normalizer.normalize_offset(pred_pos - anchor_var)
    loss = torch.nn.functional.mse_loss(pred_offset_norm, target_offset_norm)
    print(f"標準化ターゲット mean (このバッチのみ): {target_offset_norm.mean(0).tolist()}")
    print(f"標準化ターゲット std  (このバッチのみ): {target_offset_norm.std(0).tolist()}")
    print(f"ダミー損失（標準化空間）: {loss.item():.4f}")
    print("注: 上記はバッチ16サンプル単体の統計。訓練全体では mean≈0, std≈1 になる")
    print("   （標準化は訓練全体の統計で行うため、バッチ単体ではズレて見えるのが正常）")
    print("✓ 直接予測でも y 標準化が機能")

    print("\n" + "=" * 64)
    print("直接予測モード検証完了 ✓")
    print("=" * 64)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
