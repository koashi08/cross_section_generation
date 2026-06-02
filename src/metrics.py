"""評価指標。

学習損失（標準化空間）とは別に、人間が解釈できる「実スケールの再構成誤差」を計算する。

復元の流れ:
  標準化空間 pred/target
    -> normalizer.denormalize_offset で逆変換（正規化座標 = l_ref 単位）
    -> × l_ref で実スケール（元の Cartesian 座標）に復元
    -> 平均ユークリッド距離 / MSE を計算

注意: l_ref はサンプルごとに異なるので、可変ノードごとに対応する l_ref を
      ブロードキャストして使う。
"""

from __future__ import annotations

import torch


def _expand_lref_to_variable(data, var_nodes):
    """各可変ノードに、属するサンプルの l_ref を対応させる [N_var, 1] を返す。"""
    # data.l_ref: バッチ化されると [batch_size, 1]、batch[var_nodes] でサンプルID
    batch = data.batch if hasattr(data, "batch") and data.batch is not None \
        else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
    sample_id = batch[var_nodes]                 # [N_var]
    l_ref = data.l_ref.view(-1)                  # [batch_size]
    return l_ref[sample_id].unsqueeze(-1)        # [N_var, 1]


@torch.no_grad()
def reconstruction_metrics(outputs, data, normalizer=None):
    """実スケールの再構成誤差を計算する。

    Returns:
        dict:
          - mean_euclidean: 平均ユークリッド距離（実スケール）
          - mse_real: MSE（実スケール）
          - mean_euclidean_norm: 正規化座標（l_ref 単位）での平均ユークリッド距離
    """
    var_nodes = outputs["var_nodes"]
    pred_pos = outputs["pred_pos"]                  # [N_var, 2]（標準化前の正規化座標）
    anchor = outputs["anchor"]
    target = data.y[var_nodes]                      # [N_var, 2]（正規化座標 = x/l_ref, y/l_ref）

    # 注意: pred_pos / target は build_data 時点で「l_ref で割った正規化座標」。
    #       FeatureNormalizer.transform は x, edge_attr のみ標準化し、
    #       y/anchor は変えないので、ここでの pred_pos/target は正規化座標のまま。
    #       （損失計算でのみ normalizer.normalize_offset を通している）

    # 正規化座標（l_ref 単位）での誤差
    diff_norm = pred_pos - target                   # [N_var, 2]
    euclid_norm = torch.norm(diff_norm, dim=1)      # [N_var]

    # 実スケールに復元（× l_ref）
    l_ref_var = _expand_lref_to_variable(data, var_nodes)   # [N_var, 1]
    pred_real = pred_pos * l_ref_var
    target_real = target * l_ref_var
    diff_real = pred_real - target_real
    euclid_real = torch.norm(diff_real, dim=1)      # [N_var]

    return {
        "mean_euclidean": float(euclid_real.mean()),
        "mse_real": float((diff_real ** 2).mean()),
        "mean_euclidean_norm": float(euclid_norm.mean()),
    }
