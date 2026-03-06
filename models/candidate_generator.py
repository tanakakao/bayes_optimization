import torch
from torch import Tensor
from typing import List, Optional, Dict, Tuple, Any, Union

from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.acquisition import AcquisitionFunction

from bayes_optimization.models.x_constraints import make_constraints
from bayes_optimization.models.evo_acqf_optimization import (
    k_exact_sparse_transform_factory, diversify_within_q, make_k_sparse_linear_constraints_repair,
    optimize_acqf_evo, optimize_acqf_evo_mixed
)


def optimize_continuous_acqf(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    fixed_features: Optional[Dict[int, float]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, Tensor]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    sequential: bool = True,
    num_restarts: int = 5,
    raw_samples: int = 128,
    evo_method: Optional[str] = None,
    candidate_transform: Optional[Any] = None,
    repair_sum_eq: Optional[Any] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    BoTorch の optimize_acqf を用いた連続変数のみの候補点最適化。

    失敗時は num_restarts=1 でフォールバックを試みる。

    Args:
        acqf: 獲得関数。
        bounds: shape (2, d) の境界テンソル。
        q: 一度に提案する点数。
        fixed_features: 固定したい変数 {index: value}。
        equality_constraints: 等式制約 (indices, coefficients, rhs) のリスト。
        inequality_constraints: 不等式制約 (indices, coefficients, rhs) のリスト。
        sequential: q > 1 のとき逐次最適化するかどうか。
        num_restarts: 多点初期値の数。
        raw_samples: 初期候補生成に用いるサンプル数。
        options: optimize_acqf に渡すオプション dict。

    Returns:
        (candidates, acquisition_values)
    """
    base_options = {"batch_limit": 5, "maxiter": 400}
    opt_kwargs = options if options is not None else base_options
    
    if candidate_transform is not None and evo_method is None:
        raise ValueError("candidate_transformを使用する場合はevo_methodを指定してください。。")
    
    if evo_method is not None:
        return optimize_acqf_evo(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            fixed_features=fixed_features,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
            method=evo_method, # "ga", "pso", "cmaes" を切り替え可能
            options={
                "swarm_size": 80,
                "num_iterations": 200,
            },
            candidate_transform=candidate_transform,
            post_processing_func=repair_sum_eq
        )

    try:
        return optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            fixed_features=fixed_features,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=opt_kwargs,
            sequential=sequential,
        )
    except Exception:
        # フォールバック：restart を 1 に落として再実行
        return optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            fixed_features=fixed_features,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            num_restarts=1,
            raw_samples=raw_samples,
            options=opt_kwargs,
            sequential=sequential,
        )


def optimize_mixed_acqf(
    acqf: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    fixed_features,
    fixed_features_list: List[Dict[int, float]],
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, Tensor]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    num_restarts: int = 5,
    raw_samples: int = 256,
    evo_method: Optional[str] = None,
    candidate_transform: Optional[Any] = None,
    repair_sum_eq: Optional[Any] = None,
    categorical_features: Optional[Dict[int, List[int]]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    BoTorch の optimize_acqf_mixed を用いたカテゴリ変数を含む候補点最適化。

    Args:
        acqf: 獲得関数。
        bounds: shape (2, d) の境界テンソル。
        q: 一度に提案する点数。
        fixed_features_list: 各カテゴリ組合せに対する固定特徴 {index: value} のリスト。
        equality_constraints: 等式制約 (indices, coefficients, rhs) のリスト。
        inequality_constraints: 不等式制約 (indices, coefficients, rhs) のリスト。
        num_restarts: 多点初期値の数。
        raw_samples: 初期候補生成に用いるサンプル数。
        options: optimize_acqf_mixed に渡すオプション dict。

    Returns:
        (candidates, acquisition_values)
    """
    
    base_options = {"batch_limit": 5, "maxiter": 500}
    opt_kwargs = options if options is not None else base_options

    if candidate_transform is not None and evo_method is None:
        raise ValueError("candidate_transformを使用する場合はevo_methodを指定してください。。")
    
    if evo_method is not None:
        # if candidate_transform is None or candidate_transform=={}:
        #     raise ValueError("evo_methodを使用する場合はcandidate_transformを指定してください。。")
        return optimize_acqf_evo_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            fixed_features=fixed_features,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            categorical_features=categorical_features,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
            method=evo_method,           # "ga", "pso", "cmaes" を切り替え可能
            options={
                "swarm_size": 80,
                "num_iterations": 200,
            },
            candidate_transform=candidate_transform,
            post_processing_func=repair_sum_eq
        )

    return optimize_acqf_mixed(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        fixed_features_list=fixed_features_list,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=opt_kwargs,
    )


def get_candidate(
    acqf: AcquisitionFunction,
    constraint_idx: List[List[int]],
    constraint_coefs: List[List[float]],
    constraint_values: List[float],
    constraint_ops: List[str],
    fixed_features: Optional[Dict[str, float]],
    fixed_features_list: Optional[Union[List[Dict[int, float]], Dict[int, float]]],
    bounds: Tensor,
    n_cand: int,
    sequential: bool = True,
    evo_method: Optional[str] = None,
    k_sparse_spec: Optional[Tuple[List[int], int]] = None,
    categorical_features: Optional[Dict[int, List[int]]] = None,
    dtype: torch.dtype = torch.double,
) -> Tensor:
    """
    BoTorch の獲得関数に対して、線形制約付きで候補点を最適化するヘルパー。

    Args:
        acqf: 獲得関数。
        constraint_idx:
            各制約に使う変数インデックスのリスト（例: [[0,1], [2,3]]）。
        constraint_coefs:
            各制約に対応する係数のリスト（例: [[1.0,1.0], [1.0,-1.0]]）。
        constraint_values:
            各制約の右辺値（例: [1.0, 0.0]）。
        constraint_ops:
            各制約の演算子（"<", "<=", ">", ">=", "="）。
        categorical_cols:
            カテゴリカル変数の列名リスト（有無の判定にのみ使用）。
        fixed_features_list:
            - 連続のみの場合: {index: value} の dict
            - カテゴリありの場合: {index: value} の dict のリスト
        bounds:
            正規化空間での探索境界 (2, d)。
        n_cand:
            提案する候補点数 (q)。
        sequential:
            連続のみの場合に q-batch を逐次最適化するかどうか。
        dtype:
            制約係数に用いる dtype。

    Returns:
        shape (q, d) の候補点テンソル。
    """
    # 線形制約を BoTorch 形式（等式 / 不等式）に変換
    equality_constraints, inequality_constraints = make_constraints(
        constraint_idx=constraint_idx,
        constraint_coefs=constraint_coefs,
        constraint_values=constraint_values,
        constraint_ops=constraint_ops,
        dtype=dtype,
    )

    if k_sparse_spec is not None:
        if evo_method is None:
            raise ValueError("k_sparse_specを設定する場合は、evo_methodにga/pso/saを指定してください。")
        candidate_transform = k_exact_sparse_transform_factory(
            comp_idx=k_sparse_spec[0],
            k=k_sparse_spec[1],
            score="abs",
            min_active=0.0
        )
    else:
        candidate_transform = None

    if evo_method is not None and k_sparse_spec is not None:
        _repair_sum_eq = post_processing_func = make_k_sparse_linear_constraints_repair(
            bounds=bounds,
            comp_idx=k_sparse_spec[0],
            k=k_sparse_spec[1],
            score="abs",
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            max_iters=12,
        )
        repair_sum_eq = lambda X: diversify_within_q(
            X,
            _repair_sum_eq,
            bounds=bounds,
            frozen_idx=list(fixed_features.keys()),
            comp_idx=k_sparse_spec[0],
            mode="deterministic",   # 勾配最適化と相性が良い
            tol=None,               # bounds スケールから自動
            step=None,              # bounds スケールから自動
            max_tries=3,
        )
    else:
        repair_sum_eq = None
    
    # カテゴリカル変数あり → optimize_acqf_mixed
    if categorical_features is not None:
        if not isinstance(fixed_features_list, list):
            raise ValueError(
                "categorical_cols が存在する場合、fixed_features_list は List[Dict[int, float]] である必要があります。"
            )
        candidates, _ = optimize_mixed_acqf(
            acqf=acqf,
            bounds=bounds,
            q=n_cand,
            fixed_features=fixed_features,
            fixed_features_list=fixed_features_list,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            evo_method = evo_method,
            candidate_transform = candidate_transform,
            repair_sum_eq = repair_sum_eq,
            categorical_features = categorical_features,
        )
    # 連続のみ → optimize_acqf
    else:
        fixed_dict: Optional[Dict[int, float]]
        if isinstance(fixed_features_list, dict):
            fixed_dict = fixed_features_list
        elif fixed_features_list is None:
            fixed_dict = None
        else:
            # list が来た場合は「連続のみ」では通常ありえないのでエラーにしておく
            raise ValueError(
                "categorical_cols が空の場合、fixed_features_list は Dict[int, float] または None である必要があります。"
            )

        candidates, _ = optimize_continuous_acqf(
            acqf=acqf,
            bounds=bounds,
            q=n_cand,
            fixed_features=fixed_dict,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            sequential=sequential,
            evo_method = evo_method,
            candidate_transform = candidate_transform,
            repair_sum_eq = repair_sum_eq,
        )

    return candidates