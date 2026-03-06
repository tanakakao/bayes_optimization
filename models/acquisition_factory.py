import torch
from torch import Tensor
from typing import List, Optional, Union

from botorch.models.model_list_gp_regression import ModelListGP, ModelListGPyTorchModel
from botorch.models.multitask import MultiTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models import KroneckerMultiTaskGP

from botorch.acquisition import (
    qNegIntegratedPosteriorVariance,
    qUpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
)
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qLowerBoundMaxValueEntropy,
)
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qMultiObjectiveMaxValueEntropy,
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction

from botorch.acquisition.utils import get_optimal_samples
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.monte_carlo import qPosteriorStandardDeviation

from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling import SobolQMCNormalSampler

from botorch.utils.transforms import normalize, unnormalize

from bayes_optimization.models.objectives_and_y_constraints import (
    y_processing_for_constraints,
    make_constraints_y,
)
from bayes_optimization.models.acquisitions import (
    LogDetqStraddle,
    qICUAcquisition,
    qStraddle,
    qJointBoundaryVariance,
    LogDetqStraddleMultiCommon,
    qStraddleMultiCommon,
    BALDAcquisition,
    StraddleClassifierAcquisition,
    EntropyClassifierAcquisition,
    BALDMultiOutputAcquisition,
    JointStraddleClassifierAcquisition,
    EntropyMultiOutputAcquisition,
    qMaxVarianceMultiObj,
    make_logdetlike_variance_objective,
)
from bayes_optimization.models.acquisitions.robust import (
    MCJointRobustAcquisition,
    RobustqExpectedHypervolumeImprovement,
    compute_robust_train_y
)


def standardize_ref_point(
    model,
    ref_point_orig: torch.Tensor,
) -> torch.Tensor:
    """
    BoTorch モデルのスケールに合わせて ref_point を標準化する。

    Args:
        model: SingleTaskGP / ModelListGP など（outcome_transform を持つもの）。
        ref_point_orig: 元スケールでの ref_point（shape: [m]）。

    Returns:
        ref_point_std: 標準化された ref_point（shape: [m]）。
    """
    if isinstance(model, ModelListGP):
        return torch.stack(
            [
                (ref_point_orig[i] - model.models[i].outcome_transform.means)
                / model.models[i].outcome_transform.stds
                for i in range(len(model.models))
            ]
        )
    else:
        return (ref_point_orig - model.outcome_transform.means) / model.outcome_transform.stds

def get_pareto_sample(
    model,
    bounds: Tensor,
    n: int = 10,
):
    """
    パレートサンプルを取得するヘルパー関数。

    Args:
        model: 使用するモデル。
        bounds: 探索範囲 (2, d)。
        n: サンプル数。

    Returns:
        (pareto_sets, pareto_fronts)
    """
    optimizer_kwargs = {"pop_size": 2000, "max_tries": 200}
    return sample_optimal_points(
        model=model,
        bounds=bounds,
        num_samples=n,
        num_points=n,
        optimizer=random_search_optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )

def get_acqf(
    model,
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    acq_method: Optional[str] = "PI",
    y_constraints_idx: Optional[List[int]] = None,
    y_ops: Optional[List[Optional[str]]] = None,
    y_thresholds: Optional[List[float]] = None,
    y_weights: Optional[List[float]] = None,
    y_directions: Optional[List[str]] = None,
    h_lse: Optional[Union[List[float], Tensor]] = None,
    risk_type:str = None,
    n_cand: Optional[int] = 1,
    dtype: torch.dtype = torch.double,
) -> "AcquisitionFunction":
    """
    各種獲得関数（EI, PI, UCB, EHI, NEHI, NParEGO, PES, MVE, JES, AL, KG,
    Straddle 系, ICU, JBV, BALD, Entropy など）を共通インターフェースで生成する。

    Args:
        model: BoTorch モデル or ModelListGP。
        train_X: 学習入力データ (N, d)。
        train_Y: 学習出力（目的）データ (N, m)。
        bounds: 探索空間の境界 (2, d)。
        acq_method: 使用する獲得関数名。
        y_constraints_idx: 目的変数に対する制約のインデックス。
        y_ops: 目的変数制約の演算子（"<", ">", "=", None）。
        y_thresholds: 目的変数制約の閾値。
        y_weights: 目的変数の重み（スカラー化用）。
        y_directions: "min" or "max" のリスト。
        h_lse: レベルセット推定用の閾値 (Straddle 系等で使用)。
        n_cand: q-batch サイズ（一部の ACQF で使用）。
        dtype: 内部で使用する dtype。

    Returns:
        BoTorch の獲得関数インスタンス。

    Raises:
        ValueError: 不正な設定や未対応の組み合わせに対して。
    """
    y_constraints_idx = y_constraints_idx or []
    y_ops = y_ops or []
    y_thresholds = y_thresholds or []
    y_weights = y_weights or []
    y_directions = y_directions or []
    h_lse = h_lse or []

    # 何か 1 つでも有効な y 制約があれば True
    y_const = (len(y_ops) > 0) and any(op is not None for op in y_ops)

    # モデル種別の判定
    # - task_type: KroneckerMultiTaskGP かどうか（多タスク専用モデル）
    is_multitask_model = isinstance(model, KroneckerMultiTaskGP)

    output_type = "multi" if train_Y.size(-1) > 1 else "single"

    # ModelListGP or ModelListGPyTorchModel の場合、
    # 中身が MultiTaskGP / SingleTaskMultiFidelityGP なら multi_task=True
    if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):
        inner = model.models[0]
        multi_task = isinstance(inner, (MultiTaskGP, SingleTaskMultiFidelityGP))
    else:
        multi_task = False

    # 目的側制約の前処理
    (
        y_constraints_idx,
        y_ops,
        y_thresholds,
        y_weights,
        y_directions,
    ) = y_processing_for_constraints(
        train_Y,
        y_constraints_idx,
        y_ops,
        y_thresholds,
        y_directions,
        y_weights,
        dtype,
    )

    # 目的関数 & 制約を構築
    constraints, scalar_obj, objective = make_constraints_y(
        y_constraints_idx,
        y_ops,
        y_thresholds,
        y_weights,
        y_directions,
        risk_type,
        dtype,
    )

    # --- 不正な組み合わせチェック ---

    if output_type == "single" and acq_method in ["EHI", "NEHI", "NParEGO"]:
        raise ValueError(f"{acq_method} はシングルアウトプットでは使用できません。")

    if output_type == "multi" and acq_method in ["EI", "PI", "UCB"]:
        raise ValueError(f"{acq_method} はマルチアウトプットでは使用できません。")

    if is_multitask_model and acq_method not in ["EHI", "NEHI", "NParEGO"]:
        raise ValueError(f"{acq_method} は KroneckerMultiTaskGP（多タスク）では使用できません。")

    if y_const and acq_method in ["PES", "MVE", "JES", "AL"]:
        raise ValueError(f"{acq_method} では目的変数制約はサポートしていません。")

    if (not isinstance(model, ModelListGP)) and (output_type == "multi") and (acq_method == "KG"):
        raise ValueError("多目的 Knowledge Gradient は ModelListGP のみサポートしています。")

    # --- objective を当てた学習値 Y_obj を準備（best_f や ref_point の計算に使用） ---
    Y_obj = scalar_obj(train_Y)

    # sampler = SobolQMCNormalSampler(torch.Size([256]))
    sampler = SobolQMCNormalSampler(torch.Size([64]))

    # --- 獲得関数の分岐 ---

    if acq_method == "EI":
        best_f = Y_obj.max()*0.9
        # acqf = qExpectedImprovement(
        acqf = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )

    elif acq_method == "PI":
        best_f = Y_obj.max()*0.9
        acqf = qProbabilityOfImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )

    elif acq_method == "UCB":
        beta = 0.3
        acqf = qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=objective,
        )

    elif acq_method == "PES":
        if output_type == "single":
            num_samples = 32
            optimal_inputs, _ = get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_samples,
            )
            acqf = qPredictiveEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                threshold=1e-2,
            )
        else:
            ps, pf = get_pareto_sample(model, bounds, n=10)
            acqf = qMultiObjectivePredictiveEntropySearch(
                model=model,
                pareto_sets=ps,
            )

    elif acq_method == "MVE":
        if output_type == "single":
            candidate_set = torch.rand(1000, bounds.size(1), dtype=dtype, device=bounds.device)
            candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
            acqf = qLowerBoundMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
            )
        else:
            ps, pf = get_pareto_sample(model, bounds, n=10)
            hypercell_bounds = compute_sample_box_decomposition(pf)
            acqf = qLowerBoundMultiObjectiveMaxValueEntropySearch(
                model=model,
                hypercell_bounds=hypercell_bounds,
                estimation_type="LB",
            )

    elif acq_method == "JES":
        if output_type == "single":
            num_samples = 32
            optimal_inputs, optimal_outputs = get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_samples,
            )
            acqf = qJointEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
            )
        else:
            ps, pf = get_pareto_sample(model, bounds, n=10)
            hypercell_bounds = compute_sample_box_decomposition(pf)
            acqf = qLowerBoundMultiObjectiveJointEntropySearch(
                model=model,
                pareto_sets=ps,
                pareto_fronts=pf,
                hypercell_bounds=hypercell_bounds,
                estimation_type="LB",
            )

    elif acq_method == "EHI":
        ref_point = Y_obj.min(dim=0).values.to(dtype)
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=Y_obj,
        )
        # acqf = qExpectedHypervolumeImprovement(
        acqf = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )

    elif acq_method == "NEHI":
        ref_point = Y_obj.min(dim=0).values.to(dtype)
        # acqf = qNoisyExpectedHypervolumeImprovement(
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            prune_baseline=True,
            cache_root=True,
            incremental_nehvi=True,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )

    elif acq_method == "NParEGO":
        acqf = qLogNParEGO(
            model=model,
            X_baseline=train_X,
            prune_baseline=True,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
        )

    elif acq_method == "AL":
        if output_type == "single":
            mc_samples = draw_sobol_samples(bounds=bounds, n=64, q=1).squeeze(-2)
            acqf = qNegIntegratedPosteriorVariance(
                model=model,
                mc_points=mc_samples,
            )
        else:
            obj_var = make_logdetlike_variance_objective()
            acqf = qSimpleRegret(
                model=model,
                sampler=sampler,
                objective=obj_var,
            )

    elif acq_method == "KG" and not multi_task:
        if output_type == "single":
            acqf = qKnowledgeGradient(
                model=model,
                num_fantasies=64,
                current_value=None,
                sampler=SobolQMCNormalSampler(torch.Size([64])),
                objective=objective,
            )
        else:
            ref_point = (Y_obj.min(dim=0).values - 0.1).to(dtype)
            hv = Hypervolume(ref_point)
            current_value = hv.compute(Y_obj)
            acqf = qHypervolumeKnowledgeGradient(
                model=model,
                ref_point=ref_point,
                current_value=current_value,
                objective=objective,
            )

    elif acq_method == "Straddle":
        beta = 5.0
        if output_type == "single":
            h = h_lse[0]
            acqf = qStraddle(model, beta, h)
        else:
            acqf = qStraddleMultiCommon(model, beta, h_lse)

    elif acq_method == "LogStraddle":
        beta = 5.0
        if output_type == "single":
            h = h_lse[0]
            acqf = LogDetqStraddle(model, beta, h)
        else:
            acqf = LogDetqStraddleMultiCommon(model, beta, h_lse)

    elif acq_method == "ICU":
        if output_type == "single":
            h = h_lse[0]
            acqf = qICUAcquisition(model, h)
        else:
            raise ValueError("ICU は多目的には対応していません。")

    elif acq_method == "JBV":
        if output_type == "single":
            raise ValueError("JBV は単目的には対応していません。")
        else:
            acqf = qJointBoundaryVariance(model, h_lse)

    elif acq_method == "BALD":
        if output_type == "single":
            acqf = BALDAcquisition(model)
        else:
            acqf = BALDMultiOutputAcquisition(model)

    elif acq_method == "Straddle_cls":
        if output_type == "single":
            acqf = StraddleClassifierAcquisition(model)
        else:
            acqf = JointStraddleClassifierAcquisition(model)

    elif acq_method == "Entropy":
        if output_type == "single":
            acqf = EntropyClassifierAcquisition(model)
        else:
            acqf = EntropyMultiOutputAcquisition(model)

    elif acq_method == "STD":
        # Posterior Standard Deviation（不確実性最大化）
        # ・単目的：MultiObjective / GenericMCObjective をそのまま使う
        # ・多目的：分散の logdet を scalar にする objective に差し替え
        if output_type == "single":
            acqf = qPosteriorStandardDeviation(
                model=model,
                sampler=sampler,
                objective=objective,
                constraints=constraints,  # y制約があればそのまま効く
            )
        else:
            # 多目的のときは「分散の logdet」でスカラー化して
            # 「不確実性の大きい方向」を選ぶ
            variance_objective = make_logdetlike_variance_objective()
            acqf = qPosteriorStandardDeviation(
                model=model,
                sampler=sampler,
                objective=variance_objective,
                constraints=constraints,
            )
    elif acq_method=="Robust":
        if output_type == "single":
            acqf = MCJointRobustAcquisition(
                model,
                beta=2.,
                noise_penalty=2.0,
                objective=objective,
                # constraints=constraints,
            )
        else:
            ref_point = Y_obj.min(dim=0).values.to(dtype)
            robust_train_Y = compute_robust_train_y(model, train_X, noise_penalty=2.5)
            
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_point,
                Y=robust_train_Y,
            )
            
            # 4. 獲得関数の初期化
            acqf = RobustqExpectedHypervolumeImprovement(
                model=model, 
                ref_point=ref_point,
                # X_baseline=train_X,
                partitioning=partitioning, 
                beta=2.0,         # 探索の重み
                noise_penalty=2.5, # 安全性の重み
                objective=objective,
                # constraints=constraints,
            )
    
    else:
        raise ValueError(f"Unsupported acquisition function type: {acq_method}")

    # MultiTaskGP / SingleTaskMultiFidelityGP を ModelListGP に入れている場合は
    # 最後の次元を task index として 1 で固定するラッパをつける
    if multi_task:
        acqf = FixedFeatureAcquisitionFunction(
            acq_function=acqf,
            d=train_X.shape[-1],
            columns=[train_X.shape[-1] - 1],
            values=torch.tensor([1], dtype=dtype, device=train_X.device),
        )

    return acqf