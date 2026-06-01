"""断面形状 VAE で共有する ID 定数。"""

# ---- ノードタイプ ----
# node_type_code の実際のマッピングは要確認。下記はデフォルト想定。
NODE_TYPE = {
    "fixed": 0,
    "variable": 1,
    "extension": 2,
    "dependent": 3,
}
NODE_TYPE_INV = {v: k for k, v in NODE_TYPE.items()}

VAR_ID = NODE_TYPE["variable"]
CONTEXT_TYPE_IDS = (
    NODE_TYPE["fixed"],
    NODE_TYPE["extension"],
    NODE_TYPE["dependent"],
)
NUM_NODE_TYPES = len(NODE_TYPE)

# ---- エッジタイプ ----
EDGE_TYPE = {
    "straight": 0,
    "bspline": 1,
}
EDGE_TYPE_INV = {v: k for k, v in EDGE_TYPE.items()}
NUM_EDGE_TYPES = len(EDGE_TYPE)

# ---- 断面タイプ（closed_flag）----
SECTION_TYPE = {
    "open": 0,
    "closed": 1,
}

# ---- 特徴量次元（CSV 実態に合わせた版）----
# ノード特徴量 x のレイアウト:
#   [type_onehot(4), r_norm(1), sin_theta(1), cos_theta(1),
#    x_norm(1), y_norm(1), incident_edge_len_mean(1), incident_edge_count_norm(1)]
NODE_FEATURE_DIM = NUM_NODE_TYPES + 3 + 2 + 2   # = 11

# エッジ特徴量 edge_attr のレイアウト:
#   [edge_length_norm(1), kappa_abs_mean_norm(1), edge_type_onehot(2)]
EDGE_FEATURE_DIM = 2 + NUM_EDGE_TYPES           # = 4

# ---- 標準化対象の列インデックス（one-hot/sin/cos は除外）----
# x の標準化対象: r_norm, x_norm, y_norm, incident_edge_len_mean, incident_edge_count_norm
NODE_NORM_COLS = [4, 7, 8, 9, 10]    # type(0-3), r(4), sin(5), cos(6), x(7), y(8), len(9), cnt(10)
# edge_attr の標準化対象: edge_length_norm, kappa_abs_mean_norm
EDGE_NORM_COLS = [0, 1]
