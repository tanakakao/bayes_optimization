import torch
from torch import Tensor
import numpy as np
from typing import List, Optional, Tuple, Sequence, Callable, Any

from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from bayes_optimization.models.objective import (
    get_single_objective_fn,
    get_multi_objective_fn,
    get_single_objective_with_risk,
    get_multi_objective_with_risk
)

def y_processing_for_constraints(
    train_Y: torch.Tensor,
    y_constraints_idx: Sequence[int],
    y_ops: Sequence[Optional[str]],
    y_thresholds: Sequence[float],
    y_directions: Sequence[str],
    y_weights: Sequence[float],
    dtype: torch.dtype,
) -> Tuple[List[int], List[Optional[str]], List[float], torch.Tensor, List[str]]:
    """
    目的変数側の制約を学習データに合わせて整形・正規化する。

    - 指定されていない目的にもダミー制約（op=None, thr=0）を付与して全目的を対象化
    - 目的インデックス順（0..D-1）にソートして整列
    - しきい値を train_Y の平均・標準偏差で正規化
    - 方向（min/max）と重みベクトルを最終整形

    Args:
        train_Y (Tensor): 学習目的 (N, D)。
        y_constraints_idx (Sequence[int]): 制約を与える目的のインデックス群。
        y_ops (Sequence[Optional[str]]): 各制約の演算子（'=', '>', '<' または None）。
        y_thresholds (Sequence[float]): 各制約のしきい値（元スケール）。
        y_directions (Sequence[str]): 各目的の方向（'min' or 'max'）。不足時は 'max' で補完。
        y_weights (Sequence[float]): 単一化（加重）のための重み。空なら均等重みを採用。
        dtype (torch.dtype): 重みテンソルの dtype。

    Returns:
        Tuple[
            List[int],                 # y_constraints_idx（0..D-1 に整列）
            List[Optional[str]],       # y_ops（整列 & 補完）
            List[float],               # y_thresholds（平均・分散で正規化済み）
            torch.Tensor,              # y_weights（長さ D、dtype 指定）
            List[str],                 # y_directions（長さ D、'min'/'max' に正規化）
        ]
    """
    if not isinstance(train_Y, torch.Tensor):
        raise TypeError("train_Y は torch.Tensor である必要があります。")
    if train_Y.ndim != 2:
        raise ValueError("train_Y は (N, D) の2次元テンソルである必要があります。")
    n_targets = int(train_Y.size(-1))

    # y_ops 値域チェック
    valid_ops = {None, "=", ">", "<"}
    for op in y_ops:
        if op not in valid_ops:
            raise ValueError(f"y_ops に不正な演算子があります: {op}")

    # インデックスの検証（範囲内）
    for idx in y_constraints_idx:
        if not (0 <= int(idx) < n_targets):
            raise ValueError(f"y_constraints_idx が範囲外です: {idx}（D={n_targets}）")

    # ── 辞書化 → 全目的へ補完 ──
    mapping = {int(i): (o, float(t)) for i, o, t in zip(y_constraints_idx, y_ops, y_thresholds)}
    for i in range(n_targets):
        mapping.setdefault(i, (None, 0.0))

    idx_sorted = list(range(n_targets))
    ops_sorted: List[Optional[str]] = []
    thr_sorted: List[float] = []
    for i in idx_sorted:
        op_i, thr_i = mapping[i]
        ops_sorted.append(op_i)
        thr_sorted.append(thr_i)

    thr_norm = thr_sorted

    # ── 方向（min/max）の整形 ──
    dirs = list(y_directions or [])
    if len(dirs) == 0:
        dirs = ["max"] * n_targets
    elif len(dirs) < n_targets:
        pad_val = dirs[-1] if len(dirs) > 0 else "max"
        dirs = dirs + [pad_val] * (n_targets - len(dirs))
    elif len(dirs) > n_targets:
        dirs = dirs[:n_targets]

    dirs = [d.lower() for d in dirs]
    for d in dirs:
        if d not in {"min", "max"}:
            raise ValueError(f"y_directions は 'min' または 'max' である必要があります（不正: {d}）。")

    # ── 重みベクトル ──
    if len(y_weights) == 0:
        w = torch.ones(n_targets, dtype=dtype)
    else:
        if len(y_weights) != n_targets:
            raise ValueError("y_weights の長さが目的変数の次元と異なります。")
        w = torch.tensor(list(y_weights), dtype=dtype)

    if not (len(idx_sorted) == len(ops_sorted) == len(thr_norm) == n_targets):
        raise ValueError("目的変数の制約リストの長さが一致していません。")

    return idx_sorted, ops_sorted, thr_norm, w, dirs


def make_constraints_y(
    constraints_idx: List[int],
    ops: List[Optional[str]],  # "<", ">", "=", or None
    thresholds: List[float],
    y_weights: List[float],
    y_directions: List[str],  # "min" or "max"
    risk_type: str = None, # "var" or "cvar"
    dtype: torch.dtype = torch.double,
) -> Tuple[Optional[List[Callable[[Tensor], Tensor]]], Any]:
    """
    BoTorch用の「目的関数」と「y側制約関数」を構築するユーティリティ。

    Args:
        constraints_idx: 対象とする目的列のインデックス。
        ops: 各目的に対する条件演算子。"<", ">", "=", None のいずれか。
        thresholds: 各目的に対応する閾値（すでに正規化済み前提でもOK）。
        y_weights: 各目的の重み。
        y_directions: 各目的が "min" or "max" のいずれか。
        dtype: Tensorに使用するデータ型。

    Returns:
        Tuple[
            Optional[List[Callable[[Tensor], Tensor]]],  # y制約用の関数リスト
            Callable / MultiObjective                     # 目的関数
        ]
    """
    constraints: List[Callable[[Tensor], Tensor]] = []
    eq_targets: List[Optional[float]] = [None] * len(constraints_idx)

    for i, (op, threshold) in enumerate(zip(ops, thresholds)):
        idx = constraints_idx[i]
        if op == "<":
            # E[Y_i] <= threshold  →  Y_i - t <= 0 →  constraints(Y) = Y_i - t
            constraints.append(lambda Y, i=idx, t=threshold: Y[..., i] - t)
        elif op == ">":
            # E[Y_i] >= threshold  →  t - Y_i <= 0 →  constraints(Y) = t - Y_i
            constraints.append(lambda Y, i=idx, t=threshold: t - Y[..., i])
        elif op == "=":
            # 等式目標は目的関数側で扱う（eq_targets）
            eq_targets[i] = threshold
        elif op is None:
            continue
        else:
            raise ValueError(f"Unsupported operator: {op}")

    constraints_out: Optional[List[Callable[[Tensor], Tensor]]] = constraints or None

    # 目的関数の作成
    signs = torch.tensor(
        [-1.0 if d == "min" else 1.0 for d in y_directions],
        dtype=dtype,
    )
    # weights_tensor = torch.tensor(y_weights, dtype=dtype)

    if len(y_directions) == 1:
        scalar_obj = get_single_objective_fn(
            idx=constraints_idx[0],
            weight=y_weights[0],
            sign=signs[0],
            eq_target=eq_targets[0],
        )
        
        objective = get_single_objective_with_risk(
            scalar_obj_fn=scalar_obj,
            risk_type=risk_type,
        )
    else:
        scalar_obj = get_multi_objective_fn(
            idx_list=constraints_idx,
            weights=y_weights,
            signs=signs,
            eq_targets=eq_targets,
        )
        objective = get_multi_objective_with_risk(
            scalar_obj_fn=scalar_obj,
            risk_type=risk_type
        )
        
    return constraints_out, scalar_obj, objective
