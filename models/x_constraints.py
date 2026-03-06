import torch
from torch import Tensor
from typing import List, Sequence, Tuple, Optional, Union


def inequality_const(
    constraint_idx: Union[Sequence[int], Tensor],
    constraint_coefs: Union[Sequence[float], Tensor],
    constraint_value: float,
    constraint_op: str,  # "<", "<=", ">", ">="
    dtype: torch.dtype = torch.double,
) -> Tuple[Tensor, Tensor, float]:
    """
    BoTorch optimize_acqf 用の不等式制約 (indices, coefficients, rhs) を 1 つ生成する。

    BoTorch 側の仕様:
        sum_j coefficients[j] * x[indices[j]] >= rhs

    これに合わせて:
        - sum(a_j x_j) >= c      →  そのまま
        - sum(a_j x_j) <= c      →  -sum(a_j x_j) >= -c に変換

    Args:
        constraint_idx: 制約に含める変数のインデックス。
        constraint_coefs: 各変数に対応する係数 a_j。
        constraint_value: 右辺 c。
        constraint_op: 人間が指定する向き ("<", "<=", ">", ">=")。
        dtype: 係数の dtype。

    Returns:
        (indices, coefficients, rhs) のタプル。
    """
    # Tensor / list のどちらでも受けられるように as_tensor で統一
    indices = torch.as_tensor(constraint_idx, dtype=torch.long)
    coefficients = torch.as_tensor(constraint_coefs, dtype=dtype)
    rhs = float(constraint_value)

    if indices.numel() == 0:
        raise ValueError("constraint_idx が空です。")

    # 「<=」は -1 をかけて「>=」形式に変換
    if constraint_op in ("<", "<="):
        coefficients = -coefficients
        rhs = -rhs
    elif constraint_op in (">", ">="):
        # そのままで OK
        pass
    else:
        raise ValueError(f"Unsupported constraint_op: {constraint_op}")

    return indices, coefficients, rhs


def make_constraints(
    constraint_idx: List[List[int]],
    constraint_coefs: List[List[float]],
    constraint_values: List[float],
    constraint_ops: List[str],
    dtype: torch.dtype = torch.double,
) -> Tuple[
    Optional[List[Tuple[Tensor, Tensor, Tensor]]],   # equality_constraints
    Optional[List[Tuple[Tensor, Tensor, float]]],    # inequality_constraints
]:
    """
    複数の線形等式・不等式制約を生成して BoTorch 形式で返す。

    各制約に対して、対象とする変数のインデックスリストと、係数、制約値、演算子を指定する。

    Args:
        constraint_idx: 各制約に使う変数インデックスのリスト（複数可）。
        constraint_coefs: 各制約に使う係数リスト。
        constraint_values: 各制約に対応する右辺値。
        constraint_ops: 各制約の演算子 ("<", "<=", ">", ">=", "=")。
        dtype: 使用するデータ型。

    Returns:
        (equality_constraints, inequality_constraints)
        それぞれ BoTorch の optimize_acqf にそのまま渡せる形式。

    Raises:
        ValueError: 入力の長さが不一致の場合。
    """
    if not (
        len(constraint_idx)
        == len(constraint_coefs)
        == len(constraint_values)
        == len(constraint_ops)
    ):
        raise ValueError("constraint_idx/constraint_coefs/constraint_values/constraint_ops の長さが一致している必要があります。")

    if len(constraint_idx) == 0:
        return None, None

    eq_const_list: List[Tuple[Tensor, Tensor, Tensor]] = []
    ineq_const_list: List[Tuple[Tensor, Tensor, float]] = []

    for idxs, coefs, value, op in zip(
        constraint_idx, constraint_coefs, constraint_values, constraint_ops
    ):
        if op == "=":
            # 等式: sum_j a_j x_j = c
            idx_tensor = torch.as_tensor(idxs, dtype=torch.long)
            coef_tensor = torch.as_tensor(coefs, dtype=dtype)
            rhs_tensor = torch.as_tensor(value, dtype=dtype)
            eq_const_list.append((idx_tensor, coef_tensor, rhs_tensor))
        else:
            # 不等式: sum_j a_j x_j {<=,>=} c
            ineq_const_list.append(
                inequality_const(
                    constraint_idx=idxs,
                    constraint_coefs=coefs,
                    constraint_value=value,
                    constraint_op=op,
                    dtype=dtype,
                )
            )

    return (eq_const_list or None), (ineq_const_list or None)
