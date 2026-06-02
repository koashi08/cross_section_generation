"""CSV (nodes / edges / graphs) -> RawSample パーサー。

カラム名は確認済みのスキーマに対応:
  nodes:  graph_id, node_id, node_type_code, theta, r, x, y, sin_theta, cos_theta
  edges:  graph_id, edge_id, edge_type_code, src_node_id, dst_node_id,
          edge_length, kappa_abs_mean
  graphs: graph_id, closed_flag, extremum_count
"""

from __future__ import annotations

import pandas as pd

from . import constants as C
from .raw_sample import RawNode, RawEdge, RawSample

# node_type_code -> 内部 ID のマッピング（実コードに合わせて要確認）。
# 現状は恒等（CSV のコードがそのまま内部 ID と一致する想定）。
NODE_TYPE_CODE_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
EDGE_TYPE_CODE_MAP = {0: 0, 1: 1}


def load_raw_samples(nodes_csv: str, edges_csv: str, graphs_csv: str) -> dict[int, RawSample]:
    """3つの CSV を読み、graph_id をキーにした RawSample 辞書を返す。"""
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    graphs_df = pd.read_csv(graphs_csv)

    node_groups = dict(tuple(nodes_df.groupby("graph_id")))
    edge_groups = dict(tuple(edges_df.groupby("graph_id"))) if len(edges_df) else {}
    graph_rows = graphs_df.set_index("graph_id")

    samples: dict[int, RawSample] = {}
    for gid, ndf in node_groups.items():
        gid = int(gid)
        ndf = ndf.sort_values("node_id")

        # ノードの node_id -> 連番ローカル index への対応（念のため）
        local_index = {int(nid): i for i, nid in enumerate(ndf["node_id"].tolist())}

        nodes = []
        node_type_ids = []
        for _, row in ndf.iterrows():
            ntype = NODE_TYPE_CODE_MAP[int(row["node_type_code"])]
            node_type_ids.append(ntype)
            nodes.append(RawNode(
                node_type=ntype,
                r=float(row["r"]),
                theta=float(row["theta"]),
                x=float(row["x"]),
                y=float(row["y"]),
                sin_theta=float(row["sin_theta"]),
                cos_theta=float(row["cos_theta"]),
            ))

        edges = []
        edf = edge_groups.get(gid)
        if edf is not None:
            for _, row in edf.iterrows():
                edges.append(RawEdge(
                    src=local_index[int(row["src_node_id"])],
                    dst=local_index[int(row["dst_node_id"])],
                    edge_type=EDGE_TYPE_CODE_MAP[int(row["edge_type_code"])],
                    edge_length=float(row["edge_length"]),
                    kappa_abs_mean=float(row["kappa_abs_mean"]),
                ))

        grow = graph_rows.loc[gid]
        has_extension = C.NODE_TYPE["extension"] in node_type_ids
        section_type = (C.SECTION_TYPE["closed"] if int(grow["closed_flag"]) == 1
                        else C.SECTION_TYPE["open"])

        samples[gid] = RawSample(
            graph_id=gid,
            nodes=nodes,
            edges=edges,
            extremum_count=int(grow["extremum_count"]),
            section_type=section_type,
            has_extension=has_extension,
        )
    return samples
