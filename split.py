"""訓練/検証/テスト分割（極値数で層化）。

- 層化軸: extremum_count のみ（ユーザー方針）
- デフォルト比率: 95:4:1
- 分割結果（graph_id リスト）は JSON で保存し再現性を担保
"""

from __future__ import annotations

import json

from sklearn.model_selection import train_test_split

from .raw_sample import RawSample


def stratified_split(
    samples: dict[int, RawSample],
    ratios: tuple[float, float, float] = (0.95, 0.04, 0.01),
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """extremum_count で層化して train/val/test に分割。"""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios の合計が 1 でない"

    gids = list(samples.keys())
    strata = [samples[g].extremum_count for g in gids]
    n_classes = len(set(strata))

    # train vs (val+test)
    test_val_frac = ratios[1] + ratios[2]
    # 層化が可能か（各分割が最低 n_classes 以上のサンプルを持てるか）を判定
    n_temp = int(round(len(gids) * test_val_frac))
    use_stratify_1 = n_temp >= n_classes

    train_ids, temp_ids = train_test_split(
        gids, test_size=test_val_frac,
        stratify=strata if use_stratify_1 else None,
        random_state=seed,
    )

    # val vs test
    temp_strata = [samples[g].extremum_count for g in temp_ids]
    val_frac_in_temp = ratios[1] / test_val_frac
    n_val = int(round(len(temp_ids) * val_frac_in_temp))
    n_test = len(temp_ids) - n_val
    n_temp_classes = len(set(temp_strata))
    use_stratify_2 = (n_val >= n_temp_classes and n_test >= n_temp_classes)

    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_frac_in_temp,
        stratify=temp_strata if use_stratify_2 else None,
        random_state=seed,
    )

    if not (use_stratify_1 and use_stratify_2):
        import warnings
        warnings.warn(
            "サンプル数が少なく層化が一部スキップされました（ランダム分割で代替）。"
            " 実データ規模では層化が有効になります。"
        )
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def save_split(train_ids, val_ids, test_ids, path: str):
    json.dump(
        {"train": list(map(int, train_ids)),
         "val": list(map(int, val_ids)),
         "test": list(map(int, test_ids))},
        open(path, "w"), indent=2,
    )


def load_split(path: str):
    d = json.load(open(path))
    return d["train"], d["val"], d["test"]


def report_split_balance(samples, train_ids, val_ids, test_ids):
    """各分割の極値数分布を出力（層化の検証用）。"""
    from collections import Counter
    print(f"{'split':<8}{'size':>8}   極値数分布")
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        cnt = Counter(samples[g].extremum_count for g in ids)
        total = len(ids)
        dist = {k: round(cnt[k] / total, 3) for k in sorted(cnt)}
        print(f"{name:<8}{total:>8}   {dist}")
