import torch
from typing import Any, List, Optional

from gpytorch.mlls import (
    ExactMarginalLogLikelihood,
    DeepApproximateMLL,
    VariationalELBO,
    SumMarginalLogLikelihood,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.model_list_gp_regression import ModelListGPyTorchModel
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.relevance_pursuit import forward_relevance_pursuit

from .deep_gp import fit_deepgp_mll, fit_deepkernel_mll
from .classifier_gp import (
    ClassifierGPBinaryFromMulticlass,
    ClassifierMixedGPBinaryFromMulticlass,
    fit_classifier_mll,
)

from .high_dim_gp.mll import fit_saas_nuts
from .model_builders import (
    get_likelihood,
    _validate_regression_args,
    _build_deep_model,
    _build_specialized_model,
    _build_standard_model,
    _build_noise_model,
    _build_saas_model,
    _build_embed_model,
    build_input_transform,
    NoiseMode
)

def _wrap_as_model_list(models: List[Any]) -> Any:
    """
    バージョン差・モデル種別差に耐える ModelList ラッパーを返す。
    - 可能なら ModelListGP（batch_shape 等が揃う）を使う
    - 無理なら ModelListGPyTorchModel を使い、batch_shape を付与する
    """
    # まず ModelListGP を試す（これが最も無難）
    try:
        from botorch.models.model_list_gp_regression import ModelListGP
        return ModelListGP(*models)
    except Exception:
        pass

    # フォールバック：ModelListGPyTorchModel
    from botorch.models.gpytorch import ModelListGPyTorchModel
    ml = ModelListGPyTorchModel(*models)

    # ★ あなたの環境では batch_shape が無いので付与（最重要）
    if not hasattr(ml, "batch_shape"):
        bs = getattr(models[0], "batch_shape", torch.Size())
        ml.batch_shape = bs if bs is not None else torch.Size()

    # ★ 念のため num_outputs も付与（環境差対策）
    if not hasattr(ml, "num_outputs"):
        ml.num_outputs = sum(getattr(m, "num_outputs", 1) for m in models)

    return ml

def _legacy_flags_to_noise_mode(
    robust: bool,
    heteroscedastic: bool,
) -> NoiseMode:
    """
    既存の robust / heteroscedastic フラグを NoiseMode にマッピングするヘルパー。

    - robust=True, heteroscedastic=False  -> "rrp"
    - robust=False, heteroscedastic=True  -> "heteroscedastic"
    - robust=True, heteroscedastic=True   -> "heteroscedastic_rpp"
    - 両方 False                           -> "standard"
    """
    if robust and heteroscedastic:
        return "heteroscedastic_rpp"
    if robust:
        return "rrp"
    if heteroscedastic:
        return "heteroscedastic"
    return "standard"

# ---------------------------------------------------------------------
# MLL の最適化
# ---------------------------------------------------------------------
def fit_mll(
    multi_model_type: str,
    model: torch.nn.Module,
    train_X: torch.Tensor = None,
    robust: bool = False,
    hd_model: str = None,
    deep_gp: bool = False,
    deep_kernel: bool = False,
    lr: float = 1e-2,
    epoch: int = 300,
):
    """モデルに応じた MLL の最適化を実行する。"""
    robust_params = {}
    if robust:
        robust_params = {
            "numbers_of_outliers": [0, 1, 2, 3],
            "reset_parameters": False,
            "relevance_pursuit_optimizer": forward_relevance_pursuit,
        }

    if deep_gp:
        mll = DeepApproximateMLL(
            VariationalELBO(model.likelihood, model, len(train_X), beta=0.01)
        )
        fit_deepgp_mll(mll, lr=lr, epoch=epoch)

    elif deep_kernel:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_deepkernel_mll(mll, lr=lr, epoch=epoch)

    elif hd_model=="saas":
        fit_saas_nuts(
            model,
            **{
                "warmup_steps": 128,
                "num_samples":64,
                "thinning":16,
                "disable_progbar":False
            }
        )
    elif multi_model_type == "model_list":
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, **robust_params)

    else:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, **robust_params)


# ---------------------------------------------------------------------
# 回帰モデル
# ---------------------------------------------------------------------
def fit_regression_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor = None,
    multi_model_type: str = None,
    multi_task_type: str = None,
    categorical_idx: Optional[List[int]] = None,
    robust: bool = False,
    perturbation: bool = False,
    heteroscedastic: bool = False,
    is_high_dim: bool = False,
    hd_model: str=None, # saas, pca, rembo
    n_components: int = 2, 
    deep_gp: bool = False,
    deep_kernel: bool = False,
    alpha: float = 1e-1,
    lr: float = 1e-2,
    epoch: int = 300,
    list_hidden_dims: Optional[List[int]] = None,
    build_only: bool = False,
) -> Any:
    """回帰モデルの構築と学習を一括で行う。"""
    if categorical_idx is None:
        categorical_idx = []

    # 0. robust / heteroscedastic → NoiseMode へ変換（互換レイヤ）
    noise_mode: NoiseMode = _legacy_flags_to_noise_mode(
        robust=robust,
        heteroscedastic=heteroscedastic,
    )
    use_custom_noise = noise_mode != "standard"  # RRP / Hetero / Hetero+RRP を使うかどうか

    # ★ オプション：RRP 系と InputPerturbation の併用を禁止する場合
    if noise_mode in ("rrp", "heteroscedastic_rpp") and perturbation:
        raise ValueError(
            "NoiseMode='rrp' / 'heteroscedastic_rpp' と InputPerturbation の併用は "
            "SparseOutlierNoise まわりで不安定になりがちなので、どちらか一方のみ有効にしてください。"
        )

    # 1. バリデーション & multi_model_type の補正
    multi_model_type = _validate_regression_args(
        train_Y=train_Y,
        multi_model_type=multi_model_type,
        multi_task_type=multi_task_type,
        categorical_idx=categorical_idx,
        is_high_dim=is_high_dim,
        deep_gp=deep_gp,
        deep_kernel=deep_kernel,
    )

    # 2. InputTransform (Normalize + Perturbation)
    input_tf = build_input_transform(
        train_X=train_X,
        bounds=bounds,
        perturbation=perturbation,
        categorical_idx=categorical_idx,
    )

    # 3. Likelihood の準備
    likelihood = None
    if (
        hd_model != "saas"
        and multi_task_type not in ["multi_task", "multi_fidelity"]
        and not use_custom_noise
    ):
        likelihood = get_likelihood(
            multi_model_type=multi_model_type,
            train_X=train_X,
            train_Y=train_Y,
            alpha=alpha,
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
        )

    # 4. モデル構築
    model = None

    # (A) NoiseMode (Hetero / RRP / Hetero+RRP)
    if use_custom_noise:
        model = _build_noise_model(
            train_X=train_X,
            train_Y=train_Y,
            categorical_idx=categorical_idx,
            multi_model_type=multi_model_type,
            input_transform=input_tf,
            noise_mode=noise_mode,
            is_high_dim=is_high_dim,
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
            multi_task_type=multi_task_type,
        )

    # (B) Deep 系
    elif deep_gp or deep_kernel:
        model = _build_deep_model(
            train_X=train_X,
            train_Y=train_Y,
            categorical_idx=categorical_idx,
            likelihood=likelihood,
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
            input_transform=input_tf,
            list_hidden_dims=list_hidden_dims,
        )

    # (C) SAAS (高次元)
    elif is_high_dim:
        if hd_model == "saas":
            model = _build_saas_model(
                train_X=train_X,
                train_Y=train_Y,
                categorical_idx=categorical_idx,
                input_transform=input_tf,
            )
        elif hd_model in ["pca", "rembo"]:
            model = _build_embed_model(
                train_X=train_X,
                train_Y=train_Y,
                categorical_idx=categorical_idx,
                likelihood=likelihood,
                embed_type = hd_model,
                n_components = n_components,
                input_transform=input_tf,
            )
        else:
            raise ValueError("高次元のモデルは'saas','pca','rembo'から選択してください。")

    # (D) MultiTask / MultiFidelity
    elif multi_task_type in ["multi_task", "multi_fidelity"]:
        model, multi_model_type = _build_specialized_model(
            train_X=train_X,
            train_Y=train_Y,
            multi_task_type=multi_task_type,
            multi_model_type=multi_model_type,
            input_transform=input_tf,
        )

    # (E) 標準 GP
    else:
        model = _build_standard_model(
            train_X=train_X,
            train_Y=train_Y,
            categorical_idx=categorical_idx,
            multi_model_type=multi_model_type,
            likelihood=likelihood,
            input_transform=input_tf,
        )

    # 5. モデル学習
    if not build_only and model is not None:
        # RRP 最適化を使うかどうかは NoiseMode から判定
        #   - "rrp"
        #   - "heteroscedastic_rpp" のときのみ RRP を走らせる
        is_rrp_fit = noise_mode in ("rrp", "heteroscedastic_rpp")

        fit_mll(
            multi_model_type=multi_model_type,
            model=model,
            train_X=train_X,
            robust=is_rrp_fit,  # ← FitGPyTorchMLL の RRP 分岐に渡すフラグ
            hd_model=hd_model,
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
            lr=lr,
            epoch=epoch,
        )

    return model


# ---------------------------------------------------------------------
# 分類モデル（heteroscedastic 補助）
# ---------------------------------------------------------------------
def _fit_classification_noise_model(
    model: Any,
    train_X: torch.Tensor,
    categorical_idx: Optional[List[int]] = None,
) -> Any:
    """分類モデルに対して入力依存ノイズ（log-variance）モデルを学習して付与する。"""
    if categorical_idx is None:
        categorical_idx = []

    with torch.no_grad():
        model.eval()
        model.likelihood.eval()
        base_post = model.posterior(train_X, observation_noise=False)
        p = base_post.mean
        y = model.train_targets
        if y.ndim < p.ndim:
            y = y.unsqueeze(-1)
        if p.ndim < y.ndim:
            p = p.unsqueeze(-1)
        residual_sq = (p - y).pow(2).clamp_min(1e-6)
        train_Y_log_var = torch.log(residual_sq)

    transformed_train_X = model.input_transform(train_X) if getattr(model, "input_transform", None) else train_X

    if categorical_idx:
        noise_model = MixedSingleTaskGP(
            transformed_train_X,
            train_Y_log_var,
            cat_dims=categorical_idx,
        )
    else:
        noise_model = SingleTaskGP(
            transformed_train_X,
            train_Y_log_var,
        )

    mll_noise = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
    fit_gpytorch_mll(mll_noise)

    model.noise_model = noise_model.to(train_X)
    model.noise_model_uses_transformed_inputs = True
    return model


# ---------------------------------------------------------------------
# 分類モデル（robust 補助）
# ---------------------------------------------------------------------
def _refit_classification_model_with_rrp_style_filter(
    model: Any,
    train_X: torch.Tensor,
    numbers_of_outliers: Optional[List[int]] = None,
) -> Any:
    """
    分類モデル向けの簡易 Robust Relevance Pursuit 近似。

    1) まず通常学習済みモデルで学習点の誤差 |p-y| を評価
    2) 指定個数の外れ候補（既定: [0,1,2,3] の最大値）を除外
    3) inlier のみで再学習

    回帰側の `numbers_of_outliers` パラメータ構成に合わせるため、
    ここでは最大値を採用する簡易実装としている。
    """
    if numbers_of_outliers is None or len(numbers_of_outliers) == 0:
        numbers_of_outliers = [0, 1, 2, 3]

    with torch.no_grad():
        model.eval()
        model.likelihood.eval()
        post = model.posterior(train_X, observation_noise=False)
        p = post.mean
        y = model.train_targets
        if y.ndim < p.ndim:
            y = y.unsqueeze(-1)
        if p.ndim < y.ndim:
            p = p.unsqueeze(-1)
        err = (p - y).abs().reshape(-1)

    n = err.numel()
    n_out = int(max(numbers_of_outliers))
    n_out = max(0, min(n_out, max(0, n - 1)))
    if n_out == 0:
        return model

    out_idx = torch.topk(err, k=n_out, largest=True).indices
    inlier_mask = torch.ones(n, dtype=torch.bool, device=train_X.device)
    inlier_mask[out_idx] = False

    # wrapper 側の学習データ（raw）を更新
    train_X_inlier = train_X[inlier_mask]
    train_y_inlier = model.train_targets.reshape(-1)[inlier_mask]
    model.set_train_data(inputs=train_X_inlier, targets=train_y_inlier, strict=False)

    # latent GP 側の学習データ（transform 後）を更新
    transformed_X_inlier = (
        model.input_transform(train_X_inlier)
        if getattr(model, "input_transform", None)
        else train_X_inlier
    )
    model.model.train_inputs = transformed_X_inlier
    model.model.train_targets = train_y_inlier
    model.model._train_inputs_transformed = transformed_X_inlier
    model.model._train_targets = train_y_inlier

    mll = VariationalELBO(
        model.likelihood,
        model.model,
        num_data=train_X_inlier.shape[0],
    )
    fit_classifier_mll(mll)
    return model


# ---------------------------------------------------------------------
# 分類モデル
# ---------------------------------------------------------------------
def fit_classification_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    categorical_idx: Optional[List[int]] = None,
    target_class: Optional[List[Any]] = None,
    bounds: torch.Tensor = None,
    deep_gp: bool = False,
    deep_kernel: bool = False,
    robust: bool = False,
    perturbation: bool = False,
    heteroscedastic: bool = False,
    list_hidden_dims: Optional[List[int]] = None,
    build_only: bool = False,
) -> Any:
    """単一のバイナリ分類 GP モデルを構築・学習する。"""
    if categorical_idx is None:
        categorical_idx = []
    if target_class is None:
        target_class = []

    input_tf = build_input_transform(
        train_X=train_X,
        bounds=bounds,
        perturbation=perturbation,
        categorical_idx=categorical_idx,
    )


    if categorical_idx:
        model = ClassifierMixedGPBinaryFromMulticlass(
            train_X,
            train_Y,
            target_class,
            categorical_idx,
            input_transform=input_tf,
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
            list_hidden_dims=list_hidden_dims,
        )
    else:
        model = ClassifierGPBinaryFromMulticlass(
            train_X,
            train_Y,
            target_class,
            input_transform=input_tf,
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
            list_hidden_dims=list_hidden_dims,
        )

    if not build_only:
        mll = VariationalELBO(model.likelihood, model.model, num_data=train_Y.shape[0])
        fit_classifier_mll(mll)

        if robust:
            model = _refit_classification_model_with_rrp_style_filter(
                model=model,
                train_X=train_X,
                numbers_of_outliers=[0, 1, 2, 3],
            )

        if heteroscedastic:
            noise_train_X = model.train_inputs[0] if isinstance(model.train_inputs, tuple) else model.train_inputs
            model = _fit_classification_noise_model(
                model=model,
                train_X=noise_train_X,
                categorical_idx=categorical_idx,
            )

    return model


def fit_classification_models(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    categorical_idx: Optional[List[int]] = None,
    target_classes: Optional[List[List[Any]]] = None,
    bounds: torch.Tensor = None,
    deep_gp: bool = False,
    deep_kernel: bool = False,
    robust: bool = False,
    perturbation: bool = False,
    heteroscedastic: bool = False,
    list_hidden_dims: Optional[List[int]] = None,
    build_only: bool = False,
) -> List[Any]:
    """複数のターゲットに対するバイナリ分類モデル群を構築する。"""
    if target_classes is None:
        target_classes = []

    models = [
        fit_classification_model(
            train_X=train_X,
            train_Y=train_Y,
            categorical_idx=categorical_idx,
            target_class=t_class,
            bounds=bounds,
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
            robust=robust,
            perturbation=perturbation,
            heteroscedastic=heteroscedastic,
            list_hidden_dims=list_hidden_dims,
            build_only=build_only,
        )
        for t_class in target_classes
    ]
    return models


# ---------------------------------------------------------------------
# 統合インターフェース（回帰＋分類）
# ---------------------------------------------------------------------
def fit_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor = None,
    multi_model_type: str = None,
    multi_task_type: str = None,
    categorical_idx: Optional[List[int]] = None,
    cat_targets_idx: Optional[List[int]] = None,
    robust: bool = False,
    perturbation: bool = False,
    heteroscedastic: bool = False,
    is_high_dim: bool = False,
    hd_model: str=None, # saas, pca, rembo
    n_components: int = 2, 
    deep_gp: bool = False,
    deep_kernel: bool = False,
    alpha: float = 1e-1,
    lr: float = 1e-2,
    epoch: int = 300,
    list_hidden_dims: Optional[List[int]] = None,
    build_only: bool = False,
) -> Any:
    """統合インターフェース：回帰と分類（混合含む）を自動で振り分ける。"""
    if categorical_idx is None:
        categorical_idx = []

    # すべて回帰の場合
    if not cat_targets_idx:
        model = fit_regression_model(
            train_X=train_X,
            train_Y=train_Y,
            bounds=bounds,
            multi_model_type=multi_model_type,
            multi_task_type=multi_task_type,
            categorical_idx=categorical_idx,
            robust=robust,
            perturbation=perturbation,
            heteroscedastic=heteroscedastic,
            is_high_dim=is_high_dim,
            hd_model = hd_model, # saas, pca, rembo
            n_components = n_components, 
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
            alpha=alpha,
            lr=lr,
            epoch=epoch,
            list_hidden_dims=list_hidden_dims,
            build_only=build_only,
        )
        return model

    # 回帰と分類が混在する場合 (ModelListGPyTorchModel でラップ)
    models: List[Any] = []
    n_output = train_Y.shape[-1]

    for i in range(n_output):
        if i in cat_targets_idx:
            # 分類ターゲット（1 クラス vs その他）
            cls_model = fit_classification_models(
                train_X=train_X,
                train_Y=train_Y[:, [i]],
                categorical_idx=categorical_idx,
                target_classes=[[1.0]],
                bounds=bounds,
                deep_gp=deep_gp,
                deep_kernel=deep_kernel,
                robust=robust,
                perturbation=perturbation,
                heteroscedastic=heteroscedastic,
                list_hidden_dims=list_hidden_dims,
                build_only=build_only,
            )[0]
            models.append(cls_model)
        else:
            # 回帰ターゲット
            reg_model = fit_regression_model(
                train_X=train_X,
                train_Y=train_Y[:, [i]],
                bounds=bounds,
                # multi_model_type="single_task_multi_output",
                categorical_idx=categorical_idx,
                robust=robust,
                perturbation=perturbation,
                heteroscedastic=heteroscedastic,
                is_high_dim=is_high_dim,
                hd_model = hd_model, # saas, pca, rembo
                n_components = n_components, 
                deep_gp=deep_gp,
                deep_kernel=deep_kernel,
                alpha=alpha,
                lr=lr,
                epoch=epoch,
                list_hidden_dims=list_hidden_dims,
                build_only=build_only,
            )
            models.append(reg_model)

    if n_output > 1:
        # return ModelListGPyTorchModel(*models)
        return ModelListGP(*models)
    else:
        return models[0]
