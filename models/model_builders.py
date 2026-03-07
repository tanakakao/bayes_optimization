import torch
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

from botorch.models import (
    SingleTaskGP,
    MixedSingleTaskGP,
    KroneckerMultiTaskGP,
    ModelListGP,
    MultiTaskGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize, ChainedInputTransform

from .deep_gp import (
    DeepGPModel,
    DeepMixedGPModel,
    DeepKernelGPModel,
    DeepKernelMixedGPModel,
    DeepKernelDeepGPModel,
    DeepKernelDeepMixedGPModel,
)
from .robust_gp import (
    HeteroscedasticSingleTaskGP,
    HeteroscedasticMixedSingleTaskGP,
    RobustRelevancePursuitMixedSingleTaskGP,
    HeteroscedasticRobustRelevancePursuitSingleTaskGP,
    HeteroscedasticRobustRelevancePursuitMixedSingleTaskGP,
)
from .high_dim_gp import (
    MixedSaasFullyBayesianSingleTaskGP,
    REMBOSingleTaskGP,
    REMBOMixedSingleTaskGP,
    PCASingleTaskGP,
    PCAMixedSingleTaskGP
)
from .likelihoods import singletasklikelihood, multitasklikelihood
from .robust_gp.utils import setup_input_perturbation
import warnings

NoiseMode = Literal[
    "standard",            # 通常ノイズ（RRP/hetero なし）
    "rrp",                 # Robust Relevance Pursuit
    "heteroscedastic",     # ヘテロスケノイズ
    "heteroscedastic_rrp", # ヘテロスケ + RRP（必要なければ後で削ってOK）
]


# ---------------------------------------------------------------------
# Likelihood と設定バリデーション
# ---------------------------------------------------------------------
def get_likelihood(
    multi_model_type: Optional[str],
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    alpha: float,
    deep_gp: bool,
    deep_kernel: bool,
) -> Union[torch.nn.Module, List[torch.nn.Module]]:
    """モデルタイプに応じた Likelihood を返す。"""
    use_deep = deep_gp or deep_kernel
    n_output = train_Y.shape[-1]

    if multi_model_type == "multi_task_multi_output":
        return multitasklikelihood(
            train_X=train_X,
            train_Y=train_Y,
            alpha=alpha,
            deep=use_deep,
        )
    elif multi_model_type == "model_list":
        return [
            singletasklikelihood(
                train_X=train_X,
                train_Y=train_Y[:, i : i + 1],
                alpha=alpha,
                deep=use_deep,
            )
            for i in range(n_output)
        ]
    else:
        return singletasklikelihood(
            train_X=train_X,
            train_Y=train_Y,
            alpha=alpha,
            deep=use_deep,
        )


def _validate_regression_args(
    train_Y: torch.Tensor,
    multi_model_type: Optional[str],
    multi_task_type: Optional[str],
    categorical_idx: List[int],
    is_high_dim: bool,
    deep_gp: bool,
    deep_kernel: bool,
) -> str:
    """回帰モデル構築用の引数整合性チェックと multi_model_type の補正を行う。"""
    n_output = train_Y.shape[-1]
    is_deep = deep_gp or deep_kernel

    # 出力が1次元のとき、マルチタスク指定は不可
    if n_output == 1:
        if multi_model_type in [
            "single_task_multi_output",
            "multi_task_multi_output",
            "model_list",
        ]:
            if multi_task_type is not None:
                raise ValueError(
                    "train_Y が 1 次元のときはマルチタスク用の multi_model_type は指定できません。"
                    "multi_model_type を None にしてください。"
                )

    # カテゴリカル変数 + マルチタスクは非Deep系では非対応
    if not is_deep and len(categorical_idx) > 0:
        if multi_model_type == "multi_task_multi_output" or multi_task_type is not None:
            raise ValueError(
                "カテゴリカル変数を含む場合、マルチタスクは使用できません。"
                "'single_task_multi_output' か 'model_list' を使用してください。"
            )

    # multi_model_type が未指定で多出力の場合は single_task_multi_output をデフォルト採用
    if multi_model_type is None and n_output > 1:
        multi_model_type = "single_task_multi_output"

    # Deep 系モデルで多出力の場合は multi_task_multi_output として扱う
    if is_deep and n_output > 1:
        multi_model_type = "multi_task_multi_output"

    return multi_model_type


# ---------------------------------------------------------------------
# InputTransform (Normalize + optional Perturbation)
# ---------------------------------------------------------------------
def build_input_transform(
    train_X: torch.Tensor,
    bounds: torch.Tensor,
    perturbation: bool,
    categorical_idx: List[int],
) -> ChainedInputTransform:
    """Normalize と必要に応じた Perturbation をまとめて構築。"""
    dim = train_X.shape[-1]

    # Normalize
    tf_normalize = Normalize(d=dim, bounds=bounds)

    # Perturbation がないなら Normalize のみ
    if not perturbation:
        return tf_normalize

    # Perturbation を追加（Normalize 後を [0,1] 空間とみなす）
    pert_bounds = torch.tensor(
        [[0.0] * dim, [1.0] * dim],
        dtype=train_X.dtype,
        device=train_X.device,
    )
    tf_perturb = setup_input_perturbation(
        dim=dim,
        bounds=pert_bounds,
        perturbation=perturbation,
        cat_dims=categorical_idx,
    )
    return ChainedInputTransform(normalize=tf_normalize, perturb=tf_perturb)

# ---------------------------------------------------------------------
# SaaS(Sparse Axis-Aligned Subspace)
# ---------------------------------------------------------------------
def _build_saas_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    categorical_idx: List[int],
    multi_model_type: str = "model_list",
    likelihood: Any = None,
    input_transform: Optional[Any] = None,
) -> Any:
    """SAAS Fully Bayesian GP (SaasFullyBayesianSingleTaskGP) の構築。

    方針:
      - categorical_idx は未対応: 将来拡張用に NotImplementedError で停止
      - likelihood は無視（SAAS は外部から likelihood 注入しない）
      - 多目的(n_output>1)は ModelListGP のみ: 指定が違っても warning で ModelList に寄せる
    """
    n_output = train_Y.shape[-1]

    # --- likelihood は受け取っても適用しない（warning のみ） ---
    if likelihood is not None:
        warnings.warn(
            "SAAS(SaasFullyBayesianSingleTaskGP) は likelihood を外部注入しません。"
            " 渡された likelihood は無視して処理を続行します。",
            category=UserWarning,
            stacklevel=2,
        )

    # --- 多目的は ModelList のみ対応：指定が違っても warning して ModelList に強制 ---
    if n_output > 1 and multi_model_type != "model_list":
        warnings.warn(
            f"SAAS の多目的(n_output={n_output})は multi_model_type='model_list' のみ対応です。"
            f" 指定値 '{multi_model_type}' は無視し、内部的に 'model_list' として構築します。",
            category=UserWarning,
            stacklevel=2,
        )
        multi_model_type = "model_list"

    # --- 単目的で single_task_multi_output / kronecker 等が来ても SAAS は単体で扱う（warning） ---
    if n_output == 1 and multi_model_type not in ("single_task_multi_output", "model_list"):
        warnings.warn(
            f"SAAS 単目的(n_output=1)では multi_model_type='{multi_model_type}' は実質無関係です。"
            " 指定は無視して単一 SAAS モデルとして構築します。",
            category=UserWarning,
            stacklevel=2,
        )

    # --- model_list（多目的 / 単目的どちらでも可） ---
    if categorical_idx:
        if n_output > 1 and multi_model_type == "model_list":
            models = [
                MixedSaasFullyBayesianSingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, i : i + 1],  # (n, 1)
                    cat_dims=categorical_idx,
                    outcome_transform=Standardize(m=1),
                    input_transform=input_transform,
                )
                for i in range(n_output)
            ]
            return ModelListGP(*models)
        else:
            return MixedSaasFullyBayesianSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,  # (n, 1)
                cat_dims=categorical_idx,
                outcome_transform=Standardize(m=1),
                input_transform=input_transform,
            )
    else:
        if n_output > 1 and multi_model_type == "model_list":
            models = [
                SaasFullyBayesianSingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, i : i + 1],  # (n, 1)
                    outcome_transform=Standardize(m=1),
                    input_transform=input_transform,
                )
                for i in range(n_output)
            ]
            return ModelListGP(*models)
    
        # --- 単目的（single model） ---
        # n_output == 1 前提（上で多目的は model_list に寄せている）
        return SaasFullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,  # (n, 1)
            outcome_transform=Standardize(m=1),
            input_transform=input_transform,
        )

# ---------------------------------------------------------------------
# embedding(PCA-BP, REMBO)
# ---------------------------------------------------------------------
def _build_embed_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    categorical_idx: List[int],
    multi_model_type: str = "model_list",
    embed_type = "pca",
    n_components = 3,
    likelihood: Any = None,
    input_transform: Optional[Any] = None,
) -> Any:
    """SAAS Fully Bayesian GP (SaasFullyBayesianSingleTaskGP) の構築。

    方針:
      - categorical_idx は未対応: 将来拡張用に NotImplementedError で停止
      - likelihood は無視（SAAS は外部から likelihood 注入しない）
      - 多目的(n_output>1)は ModelListGP のみ: 指定が違っても warning で ModelList に寄せる
    """
    n_output = train_Y.shape[-1]

    if embed_type == "pca":
        if categorical_idx:
            return PCAMixedSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=categorical_idx,
                outcome_transform=Standardize(m=train_Y.shape[-1]),
                likelihood=likelihood,
                n_components=n_components
            )
        else:
            return PCASingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                outcome_transform=Standardize(m=train_Y.shape[-1]),
                likelihood=likelihood,
                n_components=n_components
            )
    elif embed_type == "rembo":
        if categorical_idx:
            return REMBOMixedSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=categorical_idx,
                outcome_transform=Standardize(m=train_Y.shape[-1]),
                likelihood=likelihood,
                n_components=n_components
            )
        else:
            return REMBOSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                outcome_transform=Standardize(m=train_Y.shape[-1]),
                likelihood=likelihood,
                n_components=n_components
            )
    else:
        raise ValueError("'pca'か'rembo'を指定してください。")


# ---------------------------------------------------------------------
# Deep 系モデル
# ---------------------------------------------------------------------
def _build_deep_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    categorical_idx: List[int],
    likelihood: Any,
    deep_gp: bool,
    deep_kernel: bool,
    input_transform: Optional[Any] = None,
    list_hidden_dims: Optional[List[int]] = None,
) -> Optional[Any]:
    """Deep GP / Deep Kernel 系モデルの構築。"""
    m = train_Y.shape[-1]
    outcome_transform = Standardize(m=m)
    if list_hidden_dims is None:
        list_hidden_dims = [10, 5]
    if len(list_hidden_dims) == 0:
        raise ValueError("list_hidden_dims は1つ以上の要素を指定してください。")

    if deep_gp and deep_kernel:
        if categorical_idx:
            return DeepKernelDeepMixedGPModel(
                train_X,
                train_Y,
                cat_dims=categorical_idx,
                input_transform=input_transform,
                outcome_transform=outcome_transform,
                likelihood=likelihood,
                hidden_dim=list_hidden_dims[0],
            )
        return DeepKernelDeepGPModel(
            train_X,
            train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            likelihood=likelihood,
            list_hidden_dims=list_hidden_dims,
        )

    elif deep_gp:
        if categorical_idx:
            return DeepMixedGPModel(
                train_X,
                train_Y,
                cat_dims=categorical_idx,
                input_transform=input_transform,
                outcome_transform=outcome_transform,
                likelihood=likelihood,
                hidden_dim=list_hidden_dims[0],
            )
        return DeepGPModel(
            train_X,
            train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            list_hidden_dims=list_hidden_dims,
        )

    elif deep_kernel:
        if categorical_idx:
            return DeepKernelMixedGPModel(
                train_X,
                train_Y,
                input_transform=input_transform,
                outcome_transform=outcome_transform,
                likelihood=likelihood,
                cat_dims=categorical_idx,
            )
        return DeepKernelGPModel(
            train_X,
            train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            likelihood=likelihood,
        )
    return None


# ---------------------------------------------------------------------
# MultiTask / MultiFidelity (専用モデル)
# ---------------------------------------------------------------------
def _build_specialized_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    multi_task_type: str,
    multi_model_type: str,
    input_transform: Optional[Any] = None,
) -> Tuple[Any, str]:
    """MultiTaskGP / MultiFidelityGP (ModelList) の構築。"""
    n_output = train_Y.shape[-1]

    if multi_task_type == "multi_task":
        models = [
            MultiTaskGP(
                train_X[~torch.isnan(train_Y[:, [i]]).view(-1)],
                train_Y[~torch.isnan(train_Y[:, [i]]).view(-1), i : i + 1],
                task_feature=-1,
                outcome_transform=Standardize(m=1),
                input_transform=input_transform,
            )
            for i in range(n_output)
        ]
        return ModelListGP(*models), "model_list"

    elif multi_task_type == "multi_fidelity":
        models = [
            SingleTaskMultiFidelityGP(
                train_X[~torch.isnan(train_Y[:, [i]]).view(-1)],
                train_Y[~torch.isnan(train_Y[:, [i]]).view(-1), i : i + 1],
                data_fidelities=[-1],
                outcome_transform=Standardize(m=1),
                input_transform=input_transform,
            )
            for i in range(n_output)
        ]
        return ModelListGP(*models), "model_list"

    return None, multi_model_type


# ---------------------------------------------------------------------
# 標準 GP (SingleTask / Mixed / KroneckerMultiTask)
# ---------------------------------------------------------------------
def _build_standard_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    categorical_idx: List[int],
    multi_model_type: str,
    likelihood: Any,
    input_transform: Optional[Any] = None,
) -> Any:
    """通常 GP (SingleTask, Mixed, KroneckerMultiTask) の構築。"""
    n_output = train_Y.shape[-1]

    # --- カテゴリカル変数あり ---
    if categorical_idx:
        if multi_model_type == "single_task_multi_output" or n_output == 1:
            return MixedSingleTaskGP(
                train_X,
                train_Y,
                cat_dims=categorical_idx,
                outcome_transform=Standardize(m=n_output),
                likelihood=likelihood,
                input_transform=input_transform,
            )
        elif multi_model_type == "model_list":
            models = [
                MixedSingleTaskGP(
                    train_X,
                    train_Y[:, i : i + 1],
                    cat_dims=categorical_idx,
                    outcome_transform=Standardize(m=1),
                    likelihood=likelihood[i],
                    input_transform=input_transform,
                )
                for i in range(n_output)
            ]
            return ModelListGP(*models)
        else:
            raise ValueError(
                "カテゴリカル変数を含む場合、KroneckerMultiTaskGP などは使用できません。"
            )

    # --- カテゴリカル変数なし ---
    else:
        if multi_model_type == "single_task_multi_output" or n_output == 1:
            return SingleTaskGP(
                train_X,
                train_Y,
                likelihood=likelihood,
                outcome_transform=Standardize(m=n_output),
                input_transform=input_transform,
            )
        elif multi_model_type == "model_list":
            models = [
                SingleTaskGP(
                    train_X,
                    train_Y[:, i : i + 1],
                    likelihood=likelihood[i],
                    outcome_transform=Standardize(m=1),
                    input_transform=input_transform,
                )
                for i in range(n_output)
            ]
            return ModelListGP(*models)
        else:
            return KroneckerMultiTaskGP(
                train_X,
                train_Y,
                likelihood=likelihood,
                outcome_transform=Standardize(m=n_output),
                input_transform=input_transform,
            )


# ---------------------------------------------------------------------
# Robust / Heteroscedastic / RRP 系
# ---------------------------------------------------------------------
def _build_noise_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    categorical_idx: List[int],
    multi_model_type: str,
    input_transform: Optional[Any],
    noise_mode: NoiseMode,
    is_high_dim: bool,
    deep_gp: bool,
    deep_kernel: bool,
    multi_task_type: Optional[str],
) -> Any:
    """
    NoiseMode に応じて Robust / Heteroscedastic / 両方 のモデルを構築する。

    NoiseMode:
        - "standard"           : ここでは何も作らず None を返す（呼び出し側で通常 GP）
        - "rrp"                : RobustRelevancePursuitSingleTaskGP 系
        - "heteroscedastic"    : HeteroscedasticSingleTaskGP 系
        - "heteroscedastic_rpp": HeteroscedasticRobustRelevancePursuitSingleTaskGP 系
    """
    if noise_mode == "standard":
        return None

    # 制約チェック（以前の _build_rrp_or_hetero_model と同様）
    if is_high_dim:
        raise ValueError(
            "NoiseMode='rrp' / 'heteroscedastic' / 'heteroscedastic_rpp' "
            "は SaasFullyBayesianSingleTaskGP とは併用できません。"
        )
    if deep_gp or deep_kernel:
        raise ValueError(
            "NoiseMode='rrp' / 'heteroscedastic' / 'heteroscedastic_rpp' "
            "は DeepGP / DeepKernel モデルとは併用できません。"
        )
    if multi_task_type is not None:
        raise ValueError(
            "NoiseMode='rrp' / 'heteroscedastic' / 'heteroscedastic_rpp' "
            "は multi_task / multi_fidelity 設定とは併用できません。"
        )

    n_output = train_Y.shape[-1]

    # 共通 kwargs
    common_kwargs: Dict[str, Any] = {}
    if input_transform is not None:
        common_kwargs["input_transform"] = input_transform

    # ------- NoiseMode ごとの ModelClass 選択 -------

    if noise_mode == "rrp":
        # Robust Relevance Pursuit
        common_kwargs["prior_mean_of_support"] = 2.0
        if categorical_idx:
            ModelClass = RobustRelevancePursuitMixedSingleTaskGP
            common_kwargs["cat_dims"] = categorical_idx
        else:
            ModelClass = RobustRelevancePursuitSingleTaskGP

    elif noise_mode == "heteroscedastic":
        # Heteroscedastic GP
        if categorical_idx:
            ModelClass = HeteroscedasticMixedSingleTaskGP
            common_kwargs["cat_dims"] = categorical_idx
        else:
            ModelClass = HeteroscedasticSingleTaskGP

    elif noise_mode == "heteroscedastic_rpp":
        # Heteroscedastic + RRP のハイブリッド
        # すでに動作確認済みとのことなので素直に両方を噛ませるクラスを想定
        common_kwargs["prior_mean_of_support"] = 2.0
        if categorical_idx:
            ModelClass = HeteroscedasticRobustRelevancePursuitMixedSingleTaskGP
            common_kwargs["cat_dims"] = categorical_idx
        else:
            ModelClass = HeteroscedasticRobustRelevancePursuitSingleTaskGP

    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    # ------- モデル構築 (単一 / model_list) -------

    if multi_model_type == "model_list" and n_output > 1:
        models = []
        for i in range(n_output):
            kwargs = common_kwargs.copy()
            kwargs["outcome_transform"] = Standardize(m=1)
            models.append(
                ModelClass(
                    train_X,
                    train_Y[:, i : i + 1],
                    **kwargs,
                )
            )
        return ModelListGP(*models)
    else:
        kwargs = common_kwargs.copy()
        kwargs["outcome_transform"] = Standardize(m=n_output)
        return ModelClass(train_X, train_Y, **kwargs)