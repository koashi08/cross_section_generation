"""YAML 設定の読み込みと検証。

設定を data / model / training / paths / misc のセクションに分けて管理する。
読み込んだ dict から、各コンポーネントが必要とする引数を取り出すヘルパーを提供する。

使い方:
    cfg = load_config("configs/default.yaml")
    train_cfg = to_train_config(cfg)            # TrainConfig オブジェクト
    model_kwargs = model_kwargs_from(cfg)       # CrossSectionVAE の引数 dict
"""

from __future__ import annotations

import copy
import os

import yaml

from .train import TrainConfig


# 設定に必須のトップレベルセクション
_REQUIRED_SECTIONS = ["paths", "data", "model", "training"]


def load_config(path: str, overrides: dict | None = None) -> dict:
    """YAML を読み込み、検証して dict を返す。

    Args:
        path: YAML ファイルパス
        overrides: ネストした dict で一部の値を上書き（任意）。
                   例: {"training": {"epochs": 5}}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = _deep_update(cfg, overrides)

    _validate(cfg)
    return cfg


def _deep_update(base: dict, update: dict) -> dict:
    """ネストした dict を再帰的に上書き（base は破壊しない）。"""
    result = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def _validate(cfg: dict):
    """必須セクション・キーの存在を検証する。"""
    for sec in _REQUIRED_SECTIONS:
        if sec not in cfg:
            raise ValueError(f"設定に必須セクション '{sec}' がありません")

    # paths
    for key in ["raw_dir", "processed_root", "checkpoint_dir"]:
        if key not in cfg["paths"]:
            raise ValueError(f"paths に '{key}' がありません")

    # data
    ratios = cfg["data"].get("ratios")
    if ratios is None or len(ratios) != 3:
        raise ValueError("data.ratios は3要素のリストである必要があります")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"data.ratios の合計が 1 ではありません: {sum(ratios)}")

    # model: 整合チェック（hidden_dim が num_heads で割り切れるか）
    hd = cfg["model"].get("hidden_dim")
    nh = cfg["model"].get("num_heads")
    if hd is not None and nh is not None and hd % nh != 0:
        raise ValueError(f"model.hidden_dim({hd}) は num_heads({nh}) で割り切れる必要があります")


def to_train_config(cfg: dict) -> TrainConfig:
    """設定 dict から TrainConfig を構築する。"""
    t = cfg["training"]
    device = t.get("device", "auto")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainConfig(
        epochs=t["epochs"],
        lr=t["lr"],
        batch_size=t["batch_size"],
        warmup_epochs=t["warmup_epochs"],
        beta_global_max=t["beta_global_max"],
        beta_var_max=t["beta_var_max"],
        free_bits=t.get("free_bits", 0.0),
        grad_clip=t.get("grad_clip", 1.0),
        patience=t.get("patience", 15),
        use_scheduler=t.get("use_scheduler", True),
        scheduler_factor=t.get("scheduler_factor", 0.5),
        scheduler_patience=t.get("scheduler_patience", 5),
        min_lr=t.get("min_lr", 1e-6),
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        device=device,
        log_every=t.get("log_every", 1),
    )


def model_kwargs_from(cfg: dict) -> dict:
    """設定 dict から CrossSectionVAE の引数 dict を構築する。"""
    m = cfg["model"]
    from . import constants as C
    return dict(
        in_dim=C.NODE_FEATURE_DIM,
        hidden_dim=m.get("hidden_dim", 64),
        z_global_dim=m.get("z_global_dim", 8),
        z_var_dim=m.get("z_var_dim", 16),
        z_fixed_dim=m.get("z_fixed_dim", 8),
        edge_dim=C.EDGE_FEATURE_DIM,
        num_heads=m.get("num_heads", 4),
        dropout=m.get("dropout", 0.1),
        max_var_nodes=m.get("max_var_nodes", 10),
        max_context_nodes=m.get("max_context_nodes", 16),
        use_anchor=m.get("use_anchor", False),
    )


def save_config(cfg: dict, path: str):
    """設定 dict を YAML として保存（実験の再現性のため出力先に残す用）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
