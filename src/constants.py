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

# ---- edge_attr の幾何量列（デコーダで可変-接続エッジをマスクする対象）----
# index 0=edge_length_norm, 1=kappa_abs_mean_norm が幾何量（実現形状に依存）。
# index 2-3=edge_type one-hot は構造情報なのでマスクしない。
EDGE_GEOM_COLS = [0, 1]

# ---- ノード特徴量 x のうち、可変ノードの位置を漏らす列（デコーダ slot 初期化でマスク）----
# レイアウト: [type(0-3), r(4), sin(5), cos(6), x(7), y(8), inc_len_mean(9), inc_cnt(10)]
# マスク対象: r, sin, cos, x, y（直接の位置）と inc_len_mean（隣接エッジ長→可変位置に依存）。
# 非マスク: type one-hot（タイプ情報）, inc_cnt（隣接エッジ数＝トポロジ、位置非依存）。
NODE_POS_LEAK_COLS = [4, 5, 6, 7, 8, 9]

# ---- スロット識別子（デコーダ slot 初期化に注入）----
# 一筆書き順の可変ノード通し番号 slot_id から以下を導出:
#   extremum_index = slot_id // 2   （どの極値ペアか）
#   left_right     = slot_id %  2   （ペアの左/右）
# 極値数は最大4なので、極値ペアは最大4。slot_id は最大 8（極値4×2点）。
MAX_EXTREMA = 4                      # 極値ペアの最大数（extremum_index の上限）
MAX_SLOTS = MAX_EXTREMA * 2          # slot_id の最大数（= max_var_nodes と整合）
SLOT_ID_EMBED_DIM = 8                # slot_id embedding の次元
EXTREMUM_EMBED_DIM = 4               # extremum_index embedding の次元
LEFT_RIGHT_EMBED_DIM = 2             # left_right embedding の次元
