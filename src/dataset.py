"""InMemoryDataset と、データ準備のオーケストレーション。

リーク防止の順序:
  分割 -> 訓練のみで正規化 fit -> 全データ変換+正規化適用 -> 保存
"""

from __future__ import annotations

import os

import torch
from torch_geometric.data import InMemoryDataset

from .parser import load_raw_samples
from .conversion import build_data
from .split import stratified_split, save_split, load_split, report_split_balance
from .normalize import FeatureNormalizer


class CrossSectionDataset(InMemoryDataset):
    """1分割（train/val/test）の InMemoryDataset。"""

    def __init__(self, root, split, data_list=None, transform=None):
        self._split = split
        self._incoming = data_list          # process 時に使う（None なら raw から作らない）
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self._split}.pt"]

    def process(self):
        if self._incoming is None:
            raise RuntimeError(
                f"'{self._split}' の processed が無く data_list も渡されていません。"
                " prepare_datasets() を経由してください。"
            )
        self.save(self._incoming, self.processed_paths[0])


def prepare_datasets(
    raw_dir: str,
    processed_root: str,
    ratios=(0.95, 0.04, 0.01),
    seed: int = 0,
    anchor_mode: str = "zero",
    nodes_csv="nodes.csv",
    edges_csv="edges.csv",
    graphs_csv="graphs.csv",
):
    """CSV からデータセット一式を準備する。

    Args:
        anchor_mode: anchor の計算方式。"zero"（直接予測、デフォルト）/
                     "context_mean"（隣接 context 平均）。
                     設計空間判明後に "design_space" を追加予定。

    Returns:
        (train_ds, val_ds, test_ds, normalizer)
    """
    os.makedirs(processed_root, exist_ok=True)

    # 1. CSV -> RawSample
    samples = load_raw_samples(
        os.path.join(raw_dir, nodes_csv),
        os.path.join(raw_dir, edges_csv),
        os.path.join(raw_dir, graphs_csv),
    )
    print(f"[1] RawSample: {len(samples)} サンプル")

    # 2. 層化分割（先に分割：リーク防止の肝）
    train_ids, val_ids, test_ids = stratified_split(samples, ratios, seed)
    save_split(train_ids, val_ids, test_ids, os.path.join(processed_root, "split.json"))
    print(f"[2] 分割: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    report_split_balance(samples, train_ids, val_ids, test_ids)

    # 3. build_data（分割ごと、anchor_mode を伝播）
    train_data = [build_data(samples[g], anchor_mode=anchor_mode) for g in train_ids]
    val_data = [build_data(samples[g], anchor_mode=anchor_mode) for g in val_ids]
    test_data = [build_data(samples[g], anchor_mode=anchor_mode) for g in test_ids]
    print(f"[3] build_data 完了 (anchor_mode='{anchor_mode}')")

    # 4. 正規化 fit（★訓練のみ★）
    normalizer = FeatureNormalizer().fit(train_data)
    normalizer.save(os.path.join(processed_root, "normalizer.pt"))
    print(f"[4] 正規化統計を訓練データから計算（リークなし）")

    # 5. 全分割に正規化適用
    train_data = [normalizer.transform(d) for d in train_data]
    val_data = [normalizer.transform(d) for d in val_data]
    test_data = [normalizer.transform(d) for d in test_data]
    print(f"[5] 正規化適用 完了")

    # 6. InMemoryDataset 保存
    #    processed をクリアしてから作り直す
    for split in ("train", "val", "test"):
        p = os.path.join(processed_root, "processed", f"{split}.pt")
        if os.path.exists(p):
            os.remove(p)

    train_ds = CrossSectionDataset(processed_root, "train", train_data)
    val_ds = CrossSectionDataset(processed_root, "val", val_data)
    test_ds = CrossSectionDataset(processed_root, "test", test_data)
    print(f"[6] InMemoryDataset 保存 完了")

    return train_ds, val_ds, test_ds, normalizer
