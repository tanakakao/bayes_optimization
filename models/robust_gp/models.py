#--------------------------------------------------------------------------------------------
# heteroscedastic.py
#--------------------------------------------------------------------------------------------
import torch
from torch import Tensor
from botorch.models import SingleTaskGP, MixedSingleTaskGP, KroneckerMultiTaskGP
from gpytorch.distributions import MultitaskMultivariateNormal
from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP, RobustRelevancePursuitMixin
from botorch.utils.types import _DefaultType, DEFAULT
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.means.mean import Mean
from gpytorch.module import Module

from botorch.models.transforms.input import (
    Normalize,
    InputPerturbation,
    ChainedInputTransform,
)
from botorch.models.transforms.input import InputTransform

import warnings
warnings.simplefilter(action="ignore")

def _align_like(t: Tensor, ref: Tensor) -> Tensor:
    while t.dim() < ref.dim():
        t = t.unsqueeze(0)
    if t.shape == ref.shape:
        return t
    if t.transpose(-1, -2).shape == ref.shape:
        return t.transpose(-1, -2)
    if t.numel() == ref.numel():
        return t.view_as(ref)
    return t.expand_as(ref)

class IdentityInputTransform(InputTransform):
    def transform(self, X):
        return X
    def untransform(self, X):
        return X
    def preprocess_transform(self, X):
        return X

class RobustRelevancePursuitMixedSingleTaskGP(MixedSingleTaskGP, RobustRelevancePursuitMixin):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        convex_parameterization: bool = True,
        prior_mean_of_support: float | None = None,
        cache_model_trace: bool = False,
    ) -> None:
        r"""A robust mixed single-task GP model that toggles the relevance pursuit algorithm
            during model fitting via `fit_gpytorch_mll`.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            cat_dims: A list of indices corresponding to the categorical columns in
                the input `train_X`.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            likelihood: A base likelihood.
            covar_module: The module computing the covariance (Kernel) matrix.
            mean_module: The mean function to be used.
            outcome_transform: An outcome transform.
            input_transform: An input transform.
            convex_parameterization: If True, use a convex parameterization of the
                sparse noise model.
            prior_mean_of_support: The mean value for the default exponential prior
                distribution over the support size.
            cache_model_trace: If True, cache the model trace during relevance pursuit.
        """
        self._original_X = train_X
        self._original_Y = train_Y
        self._cat_dims = cat_dims  # Store cat_dims for reconstruction

        # Initialize the MixedSingleTaskGP
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        # Apply the Robust Logic (Mixin)
        RobustRelevancePursuitMixin.__init__(
            self,
            base_likelihood=self.likelihood,
            dim=train_X.shape[-2],
            prior_mean_of_support=prior_mean_of_support,
            convex_parameterization=convex_parameterization,
            cache_model_trace=cache_model_trace,
        )

    def to_standard_model(self) -> Model:
        """Returns a standard MixedSingleTaskGP with the same parameters as this model.
        This is used to avoid recursion through the fit_gpytorch_mll dispatch."""
        is_training = self.training
        
        # Instantiate a standard MixedSingleTaskGP using the stored params
        model = MixedSingleTaskGP(
            train_X=self._original_X,
            train_Y=self._original_Y,
            cat_dims=self._cat_dims, # Pass the stored categorical dimensions
            train_Yvar=None,
            likelihood=self.likelihood,
            outcome_transform=getattr(self, "outcome_transform", None),
            input_transform=getattr(self, "input_transform", None),
        )
        
        if not is_training:
            model.eval()
        return model

# ==============================================================================
# 2. HeteroscedasticSingleTaskGP
# ==============================================================================
class HeteroscedasticSingleTaskGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, outcome_transform=None, input_transform=None):
        # 型変換とコピー
        _X = train_X.detach().clone().double()
        _Y = train_Y.detach().clone().double()
        if _Y.ndim == 1:
            _Y = _Y.unsqueeze(-1)

        # inner モデル用の Normalize だけを抽出
        normalize_tf_for_inner = None
        if isinstance(input_transform, ChainedInputTransform):
            for key in input_transform.keys():
                tf = input_transform[key]
                if isinstance(tf, Normalize):
                    normalize_tf_for_inner = tf
                    break
        elif isinstance(input_transform, Normalize):
            normalize_tf_for_inner = input_transform
        # InputPerturbation 単体やその他の場合は inner では transform しない
            
        # 1. Base GP
        base_model = SingleTaskGP(
            _X, _Y, outcome_transform=outcome_transform, input_transform=normalize_tf_for_inner
        )
        mll_base = ExactMarginalLogLikelihood(base_model.likelihood, base_model)
        fit_gpytorch_mll(mll_base)
        
        # 2. Residuals
        with torch.no_grad():
            posterior = base_model.posterior(_X)
            mean = posterior.mean
            if mean.shape != _Y.shape and mean.shape == _Y.t().shape:
                 mean = mean.t()
            residuals_sq = (mean - _Y).pow(2) + 1e-6
            train_Y_log_var = torch.log(residuals_sq)

        # 3. Noise Model
        noise_model = SingleTaskGP(_X, train_Y_log_var, input_transform=normalize_tf_for_inner) 
        mll_noise = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        fit_gpytorch_mll(mll_noise)

        # 4. Predict Noise for Init
        with torch.no_grad():
            predicted_log_var = noise_model.posterior(_X).mean
            if predicted_log_var.shape != _Y.shape and predicted_log_var.shape == _Y.t().shape:
                predicted_log_var = predicted_log_var.t()
            predicted_noise_var = torch.exp(predicted_log_var)

        # 5. Init Parent
        super().__init__(
            train_X=train_X, 
            train_Y=train_Y,
            train_Yvar=predicted_noise_var.to(train_X), 
            outcome_transform=outcome_transform,
            input_transform=input_transform
        )
        self.noise_model = noise_model

    def _get_normalize_only_transform(self):
        """Normalize のみを抽出。なければ None を返す"""
        tf = getattr(self, "input_transform", None)
        if tf is None:
            return None

        if isinstance(tf, Normalize):
            return tf

        if isinstance(tf, ChainedInputTransform):
            for name, sub_tf in tf.named_children():
                if isinstance(sub_tf, Normalize):
                    return sub_tf
        return None

    def predict_noise_logvar(self, X: Tensor) -> Tensor:
        # noise_model 自体が Normalize のみ（推奨）なので、X は raw のままでOK
        logvar = self.noise_model.posterior(X).mean
        # shape を base posterior.mean に合わせる
        base_mean = self.posterior(X, observation_noise=False).mean
        return _align_like(logvar, base_mean)

    def predict_noise_var(self, X: Tensor) -> Tensor:
        # var = exp(logvar)
        return self.predict_noise_logvar(X).exp().clamp_min(1e-12)

    def predict_noise_std(self, X: Tensor) -> Tensor:
        # std = sqrt(var)
        return self.predict_noise_var(X).sqrt()

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        # 1) base posterior は必ず latent f（observation_noise=False）で取る
        orig_tf = getattr(self, "input_transform", None)
        eval_tf = self._get_normalize_only_transform()

        if eval_tf is not None:
            self.input_transform = eval_tf
        else:
            self.input_transform = IdentityInputTransform()

        try:
            base_posterior = super().posterior(
                X,
                output_indices=output_indices,
                observation_noise=False,  # ★固定：二重加算回避
                **kwargs,
            )
        finally:
            self.input_transform = orig_tf

        if not observation_noise:
            return base_posterior

        # 2) observation_noise=True のときだけ、heteroskedastic variance を対角に足す
        orig_dist = base_posterior.distribution
        mean = orig_dist.mean

        noise_var = self.predict_noise_var(X)
        noise_var = _align_like(noise_var, mean)

        # event 次元を flatten して一律に diag を作る（multitask/バッチでも安全）
        event_ndim = len(orig_dist.event_shape)
        noise_flat = noise_var.reshape(*noise_var.shape[:-event_ndim], -1)
        noise_diag = torch.diag_embed(noise_flat)

        new_covar = orig_dist.covariance_matrix + noise_diag

        if isinstance(orig_dist, MultitaskMultivariateNormal):
            new_dist = MultitaskMultivariateNormal(mean, new_covar)
        else:
            new_dist = MultivariateNormal(mean, new_covar)

        return base_posterior.__class__(new_dist)

# ==============================================================================
# 3. HeteroscedasticMixedSingleTaskGP
# ==============================================================================
class HeteroscedasticMixedSingleTaskGP(MixedSingleTaskGP):
    def __init__(self, train_X, train_Y, cat_dims, outcome_transform=None, input_transform=None):
        _X = train_X.detach().clone().double()
        _Y = train_Y.detach().clone().double()
        if _Y.ndim == 1:
            _Y = _Y.unsqueeze(-1)

        # inner モデル用 Normalize 抽出
        normalize_tf_for_inner = None
        if isinstance(input_transform, ChainedInputTransform):
            for key in input_transform.keys():
                tf = input_transform[key]
                if isinstance(tf, Normalize):
                    normalize_tf_for_inner = tf
                    break
        elif isinstance(input_transform, Normalize):
            normalize_tf_for_inner = input_transform
        # InputPerturbation除外
            
        # 1. Base GP
        base_model = MixedSingleTaskGP(
            _X, _Y, cat_dims, outcome_transform=outcome_transform, input_transform=normalize_tf_for_inner
        )
        mll_base = ExactMarginalLogLikelihood(base_model.likelihood, base_model)
        fit_gpytorch_mll(mll_base)
        
        # 2. Residuals
        with torch.no_grad():
            posterior = base_model.posterior(_X)
            mean = posterior.mean
            if mean.shape != _Y.shape and mean.shape == _Y.t().shape:
                 mean = mean.t()
            residuals_sq = (mean - _Y).pow(2) + 1e-6
            train_Y_log_var = torch.log(residuals_sq)

        # 3. Noise Model
        noise_model = MixedSingleTaskGP(_X, train_Y_log_var, cat_dims) 
        mll_noise = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        fit_gpytorch_mll(mll_noise)

        # 4. Predict Noise for Init
        with torch.no_grad():
            predicted_log_var = noise_model.posterior(_X).mean
            if predicted_log_var.shape != _Y.shape and predicted_log_var.shape == _Y.t().shape:
                predicted_log_var = predicted_log_var.t()
            predicted_noise_var = torch.exp(predicted_log_var)

        # 5. Init Parent
        super().__init__(
            train_X=train_X, 
            train_Y=train_Y,
            cat_dims=cat_dims,
            train_Yvar=predicted_noise_var.to(train_X), 
            outcome_transform=outcome_transform,
            input_transform=input_transform
        )
        self.noise_model = noise_model

    # 【追加】Helperメソッドをこちらにも実装（継承元が違うため共有されない場合を考慮して明示的に定義）
    def _get_normalize_only_transform(self):
        tf = getattr(self, "input_transform", None)
        if tf is None:
            return None
        if isinstance(tf, Normalize):
            return tf
        if isinstance(tf, ChainedInputTransform):
            for name, sub_tf in tf.named_children():
                if isinstance(sub_tf, Normalize):
                    return sub_tf
        return None

    def _predict_noise_logvar(self, X: Tensor) -> Tensor:
        return self.noise_model.posterior(X).mean

    def _predict_noise_var(self, X: Tensor) -> Tensor:
        return self._predict_noise_logvar(X).exp().clamp_min(1e-12)

    def _predict_noise_std(self, X: Tensor) -> Tensor:
        return self._predict_noise_logvar(X).exp().clamp_min(1e-12).sqrt()

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        orig_tf = getattr(self, "input_transform", None)
        eval_tf = self._get_normalize_only_transform()

        if eval_tf is not None:
            self.input_transform = eval_tf
        else:
            self.input_transform = IdentityInputTransform()

        try:
            base_posterior = super().posterior(
                X,
                output_indices=output_indices,
                observation_noise=False,  # ★固定：二重加算回避
                **kwargs,
            )
        finally:
            self.input_transform = orig_tf

        if not observation_noise:
            return base_posterior

        orig_dist = base_posterior.distribution
        mean = orig_dist.mean

        noise_var = self._predict_noise_var(X)
        noise_var = _align_like(noise_var, mean)

        event_ndim = len(orig_dist.event_shape)
        noise_flat = noise_var.reshape(*noise_var.shape[:-event_ndim], -1)
        noise_diag = torch.diag_embed(noise_flat)

        covar = orig_dist.covariance_matrix
        new_covar = covar + noise_diag

        if isinstance(orig_dist, MultitaskMultivariateNormal):
            new_dist = MultitaskMultivariateNormal(mean, new_covar)
        else:
            new_dist = MultivariateNormal(mean, new_covar)

        return base_posterior.__class__(new_dist)

class HeteroscedasticRobustRelevancePursuitSingleTaskGP(
    RobustRelevancePursuitSingleTaskGP
):
    """
    RobustRelevancePursuitSingleTaskGP に
    ヘテロスケダスティック（入力依存ノイズ）を組み合わせたモデル。

    流れ:
        1. 一旦 RobustRelevancePursuitSingleTaskGP でベースモデルをフィット
        2. 残差² から log-variance を作り、SingleTaskGP でノイズモデルを学習
        3. ノイズモデルの予測分散を train_Yvar として持つ
           RobustRelevancePursuitSingleTaskGP を初期化（←これが本体）

    このクラスのインスタンスに対しては、
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    のように通常どおりフィットすればよい。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        outcome_transform=None,
        input_transform=None,
        convex_parameterization: bool = True,
        prior_mean_of_support: Tensor | None = None,
        cache_model_trace: bool = False,
    ) -> None:

        # ---- 型変換 & 2次元化（コピーして安全側に倒す） ----
        X = train_X.detach().clone().double()
        Y = train_Y.detach().clone().double()
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        # =====================================================
        # 0. inner モデル用の Normalize だけを抽出
        # =====================================================
        normalize_tf_for_inner = None

        if isinstance(input_transform, ChainedInputTransform):
            # 典型的には normalize=Normalize(...), perturb=InputPerturbation(...)
            # という形を想定
            for key in input_transform.keys():
                tf = input_transform[key]
                if isinstance(tf, Normalize):
                    normalize_tf_for_inner = tf
                    break
        elif isinstance(input_transform, Normalize):
            # Normalize 単体が渡されている場合
            normalize_tf_for_inner = input_transform
        else:
            # InputPerturbation 単体やその他の場合は inner では transform しない
            normalize_tf_for_inner = None
        
        # =====================================================
        # 1. ベースの「ロバスト GP」モデルで一度フィット
        # =====================================================
        base_model = RobustRelevancePursuitSingleTaskGP(
            train_X=X,
            train_Y=Y,
            outcome_transform=outcome_transform,
            input_transform=normalize_tf_for_inner,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
            cache_model_trace=cache_model_trace,
        )

        mll_base = ExactMarginalLogLikelihood(base_model.likelihood, base_model)
        fit_gpytorch_mll(mll_base)

        # =====================================================
        # 2. 残差² からノイズの log-variance を推定
        # =====================================================
        with torch.no_grad():
            posterior = base_model.posterior(X)
            mean = posterior.mean  # shape: (n, 1) or (n, m)

            # 転置が必要なケースへの対処（安全のため）
            if mean.shape != Y.shape and mean.shape == Y.t().shape:
                mean = mean.t()

            residuals_sq = (mean - Y).pow(2) + 1e-6  # 数値安定化のために ε を足す
            train_Y_log_var = torch.log(residuals_sq)

        # =====================================================
        # 3. ノイズモデル: log-variance を SingleTaskGP で学習
        # =====================================================
        noise_model = SingleTaskGP(X, train_Y_log_var)
        mll_noise = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        fit_gpytorch_mll(mll_noise)

        # =====================================================
        # 4. ノイズモデルの予測分散を使って RRP-GP を初期化
        # =====================================================
        with torch.no_grad():
            predicted_log_var = noise_model.posterior(X).mean

            if predicted_log_var.shape != Y.shape and predicted_log_var.shape == Y.t().shape:
                predicted_log_var = predicted_log_var.t()

            predicted_noise_var = torch.exp(predicted_log_var)  # variance (σ^2)

        # ここからが「本体」の RobustRelevancePursuitSingleTaskGP
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=predicted_noise_var.to(train_X),  # dtype/device を train_X に合わせる
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
            cache_model_trace=cache_model_trace,
        )

        # 後段のロバスト獲得関数などから使うために保持しておく
        self.noise_model = noise_model

    # ★ここが今回の「エラー潰し」ポイント
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        RRP の _fit_rrp から standard_model の state_dict をロードするとき、
        standard_model 側は noise_model.* を持っていないので missing になる。
        それを無視するために strict=False で読み込み、
        noise_model.* 以外に missing がある場合だけエラーにする。
        """
        incompatible: _IncompatibleKeys = super().load_state_dict(
            state_dict, strict=False
        )

        # noise_model.* 以外で missing があれば拾う
        missing_non_noise = [
            k for k in incompatible.missing_keys
            if not k.startswith("noise_model.")
        ]

        if strict and len(missing_non_noise) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict (excluding noise_model.*): "
                + ", ".join(missing_non_noise)
            )

        # RRP 側は戻り値を使っていないので、そのまま返しておけばOK
        return incompatible

class HeteroscedasticRobustRelevancePursuitMixedSingleTaskGP(
    RobustRelevancePursuitMixedSingleTaskGP
):
    """
    RobustRelevancePursuitMixedSingleTaskGP に
    ヘテロスケダスティック（入力依存ノイズ）を組み合わせたモデル。

    流れ:
        1. 一旦 RobustRelevancePursuitMixedSingleTaskGP でベースモデルをフィット
        2. 残差² から log-variance を作り、MixedSingleTaskGP でノイズモデルを学習
        3. ノイズモデルの予測分散を train_Yvar として持つ
           RobustRelevancePursuitMixedSingleTaskGP を初期化（←これが本体）

    このクラスのインスタンスに対しては、
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    のように通常どおりフィットすればよい。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims,
        outcome_transform=None,
        input_transform=None,
        convex_parameterization: bool = True,
        prior_mean_of_support: Tensor | None = None,
        cache_model_trace: bool = False,
    ) -> None:

        # ---- 型変換 & 2次元化（コピーして安全側に倒す） ----
        X = train_X.detach().clone().double()
        Y = train_Y.detach().clone().double()
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        # =====================================================
        # 0. inner モデル用の Normalize だけを抽出
        # =====================================================
        normalize_tf_for_inner = None

        if isinstance(input_transform, ChainedInputTransform):
            # 典型的には normalize=Normalize(...), perturb=InputPerturbation(...)
            # という形を想定
            for key in input_transform.keys():
                tf = input_transform[key]
                if isinstance(tf, Normalize):
                    normalize_tf_for_inner = tf
                    break
        elif isinstance(input_transform, Normalize):
            # Normalize 単体が渡されている場合
            normalize_tf_for_inner = input_transform
        else:
            # InputPerturbation 単体やその他の場合は inner では transform しない
            normalize_tf_for_inner = None
        
        # =====================================================
        # 1. ベースの「ロバスト Mixed GP」モデルで一度フィット
        # =====================================================
        base_model = RobustRelevancePursuitMixedSingleTaskGP(
            train_X=X,
            train_Y=Y,
            cat_dims=cat_dims,
            outcome_transform=outcome_transform,
            input_transform=normalize_tf_for_inner,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
            cache_model_trace=cache_model_trace,
        )

        mll_base = ExactMarginalLogLikelihood(base_model.likelihood, base_model)
        fit_gpytorch_mll(mll_base)

        # =====================================================
        # 2. 残差² からノイズの log-variance を推定
        # =====================================================
        with torch.no_grad():
            posterior = base_model.posterior(X)
            mean = posterior.mean  # shape: (n, 1) or (n, m)

            # 転置が必要なケースへの対処（安全のため）
            if mean.shape != Y.shape and mean.shape == Y.t().shape:
                mean = mean.t()

            residuals_sq = (mean - Y).pow(2) + 1e-6  # 数値安定化のために ε を足す
            train_Y_log_var = torch.log(residuals_sq)

        # =====================================================
        # 3. ノイズモデル: log-variance を MixedSingleTaskGP で学習
        # =====================================================
        noise_model = MixedSingleTaskGP(
            X,
            train_Y_log_var,
            cat_dims=cat_dims,
        )
        mll_noise = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        fit_gpytorch_mll(mll_noise)

        # =====================================================
        # 4. ノイズモデルの予測分散を使って RRP-MixedGP を初期化
        # =====================================================
        with torch.no_grad():
            noise_post = noise_model.posterior(X)
            predicted_log_var = noise_post.mean

            if predicted_log_var.shape != Y.shape and predicted_log_var.shape == Y.t().shape:
                predicted_log_var = predicted_log_var.t()

            predicted_noise_var = torch.exp(predicted_log_var)  # variance (σ^2)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=predicted_noise_var.to(train_X),  # dtype/device を train_X に合わせる
            cat_dims=cat_dims,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
            cache_model_trace=cache_model_trace,
        )

        # 後段のロバスト獲得関数などから使うために保持しておく
        self.noise_model = noise_model

    # ★ここが今回の「エラー潰し」ポイント
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        RRP の _fit_rrp から standard_model の state_dict をロードするとき、
        standard_model 側は noise_model.* を持っていないので missing になる。
        それを無視するために strict=False で読み込み、
        noise_model.* 以外に missing がある場合だけエラーにする。
        """
        incompatible: _IncompatibleKeys = super().load_state_dict(
            state_dict, strict=False
        )

        # noise_model.* 以外で missing があれば拾う
        missing_non_noise = [
            k for k in incompatible.missing_keys
            if not k.startswith("noise_model.")
        ]

        if strict and len(missing_non_noise) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict (excluding noise_model.*): "
                + ", ".join(missing_non_noise)
            )

        # RRP 側は戻り値を使っていないので、そのまま返しておけばOK
        return incompatible