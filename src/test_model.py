"""VAE モデルの動作確認。

- forward が通るか（形状確認）
- 損失計算
- backward + optimizer.step（学習1ステップ）
- パラメータ数の実測
- generate（推論）の動作
- 極値数1-4 すべてで動くか
"""

import os
import shutil

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from cross_section import constants as C
from cross_section.csv_export import generate_csv_dataset
from cross_section.dataset import prepare_datasets, CrossSectionDataset
from cross_section.normalize import FeatureNormalizer
from cross_section.vae import CrossSectionVAE, vae_loss

RAW = "/home/claude/_data_raw_model"
ROOT = "/home/claude/_data_proc_model"


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    for d in (RAW, ROOT):
        if os.path.exists(d):
            shutil.rmtree(d)

    print("=" * 64)
    print("【1】データ準備（直接予測モード）")
    print("=" * 64)
    generate_csv_dataset(RAW, n_per_extremum=200, seed=0)
    train_ds, val_ds, test_ds, normalizer = prepare_datasets(
        RAW, ROOT, ratios=(0.95, 0.04, 0.01), seed=0, anchor_mode="zero")

    print("\n" + "=" * 64)
    print("【2】モデル構築 + パラメータ数")
    print("=" * 64)
    model = CrossSectionVAE(
        hidden_dim=64, num_heads=4, use_anchor=False,
        max_var_nodes=10, max_context_nodes=50,
    )
    print(f"総パラメータ数: {count_params(model):,}")
    print(f"  encoder: {count_params(model.encoder):,}")
    print(f"    shared_embed:   {count_params(model.encoder.shared_embed):,}")
    print(f"    global_branch:  {count_params(model.encoder.global_branch):,}")
    print(f"    context_branch: {count_params(model.encoder.context_branch):,}")
    print(f"    variable_branch:{count_params(model.encoder.variable_branch):,}")
    print(f"  decoder: {count_params(model.decoder):,}")

    print("\n" + "=" * 64)
    print("【3】forward 動作確認（形状）")
    print("=" * 64)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    model.train()
    out = model(batch)

    n_var = int(batch.variable_mask.sum())
    bs = int(batch.batch.max()) + 1
    print(f"batch_size: {bs}, 可変ノード総数: {n_var}")
    print(f"z_global:  {tuple(out['z_global'].shape)} (期待 [{bs}, 8])")
    print(f"z_var:     {tuple(out['z_var'].shape)} (期待 [{bs}, 16])")
    print(f"z_fixed:   {tuple(out['z_fixed'].shape)} (期待 [{bs}, 8])")
    print(f"pred_pos:  {tuple(out['pred_pos'].shape)} (期待 [{n_var}, 2])")
    print(f"offset:    {tuple(out['offset'].shape)}")

    assert out["z_global"].shape == (bs, 8)
    assert out["z_var"].shape == (bs, 16)
    assert out["z_fixed"].shape == (bs, 8)
    assert out["pred_pos"].shape == (n_var, 2)
    # 直接予測なので pred_pos == offset
    assert torch.allclose(out["pred_pos"], out["offset"]), "use_anchor=False で pred_pos==offset のはず"
    print("✓ 全テンソル形状 OK / 直接予測（pred_pos==offset）確認")

    print("\n" + "=" * 64)
    print("【4】損失計算")
    print("=" * 64)
    loss, logs = vae_loss(out, batch, normalizer=normalizer, beta_global=1.0, beta_var=1.0)
    print(f"loss 内訳: {logs}")
    assert torch.isfinite(loss), "損失が NaN/Inf"
    print("✓ 損失が有限値")

    print("\n" + "=" * 64)
    print("【5】backward + optimizer.step（学習1ステップ）")
    print("=" * 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    # 勾配が流れているか確認
    n_with_grad = sum(1 for p in model.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    print(f"勾配が流れたパラメータ: {n_with_grad}/{n_total}")
    optimizer.step()
    print("✓ backward + step 成功")

    print("\n" + "=" * 64)
    print("【6】数ステップ学習して損失が下がるか")
    print("=" * 64)
    model.train()
    losses = []
    for step in range(20):
        b = next(iter(DataLoader(train_ds, batch_size=32, shuffle=True)))
        out = model(b)
        loss, logs = vae_loss(out, b, normalizer=normalizer, beta_global=0.1, beta_var=0.1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(logs["recon"])
        if step % 5 == 0:
            print(f"  step {step:2d}: recon={logs['recon']:.4f}, "
                  f"kl_g={logs['kl_global']:.4f}, kl_v={logs['kl_var']:.4f}")
    print(f"\n再構成損失: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "再構成損失が下がっていない"
    print("✓ 再構成損失が減少（学習が機能）")

    print("\n" + "=" * 64)
    print("【7】generate（推論）動作確認")
    print("=" * 64)
    model.eval()
    b = next(iter(DataLoader(test_ds, batch_size=8, shuffle=False)))
    gen = model.generate(b)   # z をランダムサンプリング
    n_var_test = int(b.variable_mask.sum())
    print(f"生成 pred_pos: {tuple(gen['pred_pos'].shape)} (期待 [{n_var_test}, 2])")
    assert gen["pred_pos"].shape == (n_var_test, 2)
    assert torch.isfinite(gen["pred_pos"]).all()
    print("✓ generate 動作 OK")

    print("\n" + "=" * 64)
    print("【8】極値数1-4 すべてで forward が通るか")
    print("=" * 64)
    model.train()
    for ext in (1, 2, 3, 4):
        # 該当する極値数のサンプルだけ抽出
        sub = [d for d in train_ds if int(d.extremum_count) == ext][:4]
        b = next(iter(DataLoader(sub, batch_size=4, shuffle=False)))
        out = model(b)
        loss, _ = vae_loss(out, b, normalizer=normalizer)
        assert torch.isfinite(loss)
        print(f"  極値{ext}: 可変ノード{int(b.variable_mask.sum())}, loss={float(loss):.4f}  OK")
    print("✓ 全極値数で forward OK")

    print("\n" + "=" * 64)
    print("【9】use_anchor=True でも forward が通るか（将来の切替確認）")
    print("=" * 64)
    model_anchor = CrossSectionVAE(hidden_dim=64, use_anchor=True)
    b = next(iter(DataLoader(train_ds, batch_size=8, shuffle=False)))
    out_a = model_anchor(b)
    # use_anchor=True なら pred_pos = anchor + offset（anchor=0 なので結果は同じだが経路が違う）
    assert out_a["pred_pos"].shape == out_a["offset"].shape
    print(f"use_anchor=True: pred_pos {tuple(out_a['pred_pos'].shape)}")
    print("✓ use_anchor=True でも forward OK（フラグ切替が機能）")

    print("\n" + "=" * 64)
    print("モデル動作確認 完了 ✓")
    print("=" * 64)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
