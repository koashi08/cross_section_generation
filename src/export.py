"""予測結果の CSV 出力。

再構成（model.forward）・生成（model.generate）どちらの出力でも使えるよう、
モデル出力 dict と data を受け取って予測座標を実スケールに復元し CSV 化する。

出力スキーマ（可変ノード単位、1行 = 1可変ノード）:
  graph_id, node_index, pred_x, pred_y, target_x, target_y, euclidean_error
  - 座標は実スケール（× l_ref で元の Cartesian に復元）
  - target は再構成モードでのみ意味を持つ（生成モードでは入力形状の値）
  - node_index は元グラフ（バッチ前）でのローカル index
"""

from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.loader import DataLoader


def _rows_from_outputs(out, batch, normalizer=None, include_target=True):
    """1バッチのモデル出力から、可変ノード単位の行リストを作る（実スケール）。"""
    var_nodes = out["var_nodes"].cpu().numpy()        # 元グラフでの可変ノード index
    pred = out["pred_pos"].detach().cpu().numpy()     # [N_var, 2]（正規化座標）
    b = batch.batch.cpu().numpy()
    gid = batch.graph_id.view(-1).cpu().numpy()
    l_ref = batch.l_ref.view(-1).cpu().numpy()

    target = None
    if include_target:
        target = batch.y[out["var_nodes"]].detach().cpu().numpy()  # [N_var, 2]

    # 元グラフでの node_id（パディング前のローカル index に戻す）
    # batch.batch[var_nodes] でサンプルID、各サンプル内の通し番号を node_index とする
    rows = []
    sample_local_counter = {}
    for j, vn in enumerate(var_nodes):
        sid = int(b[vn])
        lref = float(l_ref[sid])
        local_idx = sample_local_counter.get(sid, 0)
        sample_local_counter[sid] = local_idx + 1

        row = {
            "graph_id": int(gid[sid]),
            "var_local_index": local_idx,          # サンプル内の可変ノード通し番号
            "pred_x": float(pred[j, 0] * lref),
            "pred_y": float(pred[j, 1] * lref),
        }
        if include_target and target is not None:
            tx = float(target[j, 0] * lref)
            ty = float(target[j, 1] * lref)
            err = ((pred[j] - target[j]) * lref)
            row["target_x"] = tx
            row["target_y"] = ty
            row["euclidean_error"] = float((err ** 2).sum() ** 0.5)
        rows.append(row)
    return rows


@torch.no_grad()
def export_predictions(model, dataset, out_csv, device="cpu",
                       normalizer=None, batch_size=64,
                       mode="reconstruct", z_global=None, z_var=None):
    """データセット全体の予測座標を CSV 出力する。

    Args:
        model: CrossSectionVAE
        dataset: 予測対象のデータセット
        out_csv: 出力 CSV パス
        device: 実行デバイス
        normalizer: 現状は未使用（pred_pos は正規化座標で得られ × l_ref で復元）。
                    将来 normalize_offset を経由する場合の拡張用。
        mode: "reconstruct"（forward, 再構成）または "generate"（z サンプリング生成）
        z_global, z_var: generate モードで z を明示指定する場合（None ならサンプリング）
    Returns:
        出力した CSV のパス
    """
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_rows = []
    for batch in loader:
        batch = batch.to(device)
        if mode == "reconstruct":
            out = model(batch)
            include_target = True
        elif mode == "generate":
            out = model.generate(batch, z_global=z_global, z_var=z_var)
            include_target = True   # 生成でも入力形状との比較用に target を残す
        else:
            raise ValueError(f"未知の mode: {mode}（reconstruct / generate）")

        all_rows.extend(_rows_from_outputs(out, batch, normalizer, include_target))

    df = pd.DataFrame(all_rows)
    # graph_id, var_local_index でソートして見やすく
    if not df.empty:
        df = df.sort_values(["graph_id", "var_local_index"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return out_csv
