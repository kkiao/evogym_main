import random
import numpy as np

# EvoGymが体を読むときの材料番号。数字ではなく名前で使う。
EMPTY = 0      # 空白。ブロックを置かない。
RIGID = 1      # 硬いブロック。骨や支えの役割。
SOFT = 2       # 柔らかいブロック。
H_ACT = 3      # 横方向に伸縮する部品（筋肉）。
V_ACT = 4      # 縦方向に伸縮する部品（筋肉）。

def make_walker_body():
    """固定体RL用：左右の脚に近い伸縮部品を持つ、横移動向けの体。"""
    body = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 3, 1, 3, 1],
        [0, 4, 0, 4, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)
    return body


def make_soft_walker_body():
    """開発を停止した旧形状。既存の過去実験を再現する場合にのみ使用する。"""
    body = np.array([
        [0, 0, 2, 0, 0],
        [0, 2, 1, 2, 0],
        [1, 3, 2, 3, 1],
        [0, 4, 1, 4, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)
    return body


def make_layered_walker_body():
    """層状Walker。上段は剛体、中段は横方向、下段は縦方向のアクチュエータ。"""
    body = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0],
    ], dtype=int)
    return body


def make_climber_seed_body():
    """体進化用：上下へ力を出す部品を持つ、Climberの親となる体。"""
    body = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 4, 1, 0],
        [1, 3, 1, 3, 1],
        [0, 1, 4, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=int)
    return body


#has_muscle == 筋肉がある
#row == 行,col == 列
def has_muscle(body):
    # 体の全マスを順に調べ、筋肉が一つでもあればTrueを返す。
    #shape[0]は行数、shape[1]は列数を返す。
    #body == (2(行),3(列))のようになっているから
    for row in range(body.shape[0]):
        for col in range(body.shape[1]):
            material = body[row][col]
            if material == H_ACT or material == V_ACT:
                return True
    return False
    

#mutate == 突然変異する
def mutate_body(parent_body):
    # copy()で親とは別の配列を作る。親の体を壊さないために必要。
    child_body = parent_body.copy()
    add_candidates = []
    occupied_candidates = []

    # 既存ブロックと、その上下左右の空白を調べる。
    for row in range(child_body.shape[0]):
        for col in range(child_body.shape[1]):
            if child_body[row][col] != EMPTY:
                occupied_candidates.append((row, col))

                for row_change, col_change in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    new_row = row + row_change
                    new_col = col + col_change
                    inside_body = 0 <= new_row < child_body.shape[0] and 0 <= new_col < child_body.shape[1]
                    if inside_body and child_body[new_row][new_col] == EMPTY:
                        add_candidates.append((new_row, new_col))

    material = random.choice([RIGID, SOFT, H_ACT, V_ACT])

    # 隣接する空白があるときは、60%の確率で新ブロックを追加する。
    if add_candidates and random.random() < 0.60:
        row, col = random.choice(add_candidates)
        child_body[row][col] = material
    else:
        # 追加できない、または40%のときは、既存ブロックの材料を変える。
        row, col = random.choice(occupied_candidates)
        child_body[row][col] = material

    # 筋肉が消えた体は動かせないので、選んだマスを筋肉にする。
    if not has_muscle(child_body):
        child_body[row][col] = H_ACT

    return child_body
