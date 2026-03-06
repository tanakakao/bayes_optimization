# robust_acqf.py
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective,
    IdentityMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform


# =========================================================
# 共通ヘルパ
# =========================================================
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
    
def get_noise_sigma(
    model: Model,
    X: Tensor,
    *,
    mean_like: Tensor,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
) -> Tensor:
    r"""観測ノイズの標準偏差 σ(x) を推定して mean_like と shape を揃えて返す。

    優先順位:
      1) model.noise_model があれば、その posterior.mean から推定
      2) model.likelihood.noise が取れれば、それを利用
      3) default_sigma を一様ノイズとして利用

    Args:
        model: BoTorch Model
        X: (..., q, d)
        mean_like: posterior.mean と同 shape を想定 (..., q, m)
        default_sigma: noise_model が無い場合の一様 σ
        noise_is_log_var: noise_model の mean が log(σ^2) である前提なら True

    Returns:
        σ(x): mean_like と同 shape (..., q, m)
    """
    # noise_model 側だけ input_transform を手動適用（model.posterior 側は内部適用される前提）
    X_noise = X
    if hasattr(model, "input_transform") and model.input_transform is not None:
        X_noise = model.input_transform(X_noise)

    # 1) heteroskedastic noise_model
    if hasattr(model, "noise_model") and model.noise_model is not None:
        with torch.no_grad():
            noise_post = model.noise_model.posterior(X_noise)
            nm = noise_post.mean
            sigma = nm.exp().clamp_min(1e-12).sqrt() if noise_is_log_var else nm
        return _align_like(sigma, mean_like)

    # 2) likelihood.noise（取れれば）
    sigma_val = torch.as_tensor(default_sigma, dtype=mean_like.dtype, device=mean_like.device)
    if hasattr(model, "likelihood") and model.likelihood is not None:
        try:
            noise = getattr(model.likelihood, "noise", None)
        except Exception:
            noise = None
        if noise is not None:
            # noise は variance 前提が多い
            sigma_val = noise.to(mean_like).clamp_min(1e-12).sqrt()

    return sigma_val * torch.ones_like(mean_like)


def robustify_samples(
    model: Model,
    X: Tensor,
    samples: Tensor,
    *,
    beta: float = 0.0,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    posterior: Optional[object] = None,
) -> Tensor:
    r"""MC サンプルをロバスト化して返す。

    - βスケーリング（UCB 的）:  beta != 0 のときのみ適用
        s' = μ + beta (s - μ)
      ※ beta=0 のときは「何もしない」（=元サンプルのまま）

    - ノイズペナルティ: noise_penalty != 0 のときのみ適用
        s'' = s' - noise_penalty * σ(x)

    Args:
        model: BoTorch Model
        X: (..., q, d)
        samples: (S, ..., q, m)
        posterior: 既に計算済みなら渡してよい（model.posterior(X)）

    Returns:
        robust_samples: (S, ..., q, m)
    """
    if posterior is None:
        posterior = model.posterior(X)

    mu = posterior.mean  # (..., q, m)

    # mu を samples と同 rank に
    mu_exp = mu
    while mu_exp.dim() < samples.dim():
        mu_exp = mu_exp.unsqueeze(0)

    out = samples

    # βスケーリング（beta=0 なら無効化）
    if beta != 0.0:
        out = mu_exp + torch.as_tensor(beta, device=out.device, dtype=out.dtype) * (out - mu_exp)

    # ノイズペナルティ（noise_penalty=0 なら無効化）
    if noise_penalty != 0.0:
        sigma = get_noise_sigma(
            model,
            X,
            mean_like=mu,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
        )  # (..., q, m)
        sigma_exp = sigma
        while sigma_exp.dim() < out.dim():
            sigma_exp = sigma_exp.unsqueeze(0)
        sigma_exp = _align_like(sigma_exp, out)
        out = out - torch.as_tensor(noise_penalty, device=out.device, dtype=out.dtype) * sigma_exp

    return out


def compute_robust_train_y(
    model: Model,
    train_X: Tensor,
    *,
    noise_penalty: float = 2.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
) -> Tensor:
    r"""学習点 train_X のロバスト目的値（μ - λσ）を計算する（Pareto 前処理等に利用）.

    Returns:
        robust_Y: (N, m) または (batch..., N, m)
    """
    with torch.no_grad():
        post = model.posterior(train_X)
        mu = post.mean
        sigma = get_noise_sigma(
            model,
            train_X,
            mean_like=mu,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
        )
        return mu - noise_penalty * sigma


# =========================================================
# Objective ラッパ（qNEHVI 用）
# =========================================================

class RobustMCMultiOutputObjective(MCMultiOutputObjective):
    r"""base_objective の前にサンプルをロバスト変換する Objective.

    qNEHVI は「objective(samples, X)」経由で値を使うので、ここで robustify するのが自然です。
    """

    def __init__(
        self,
        base_objective: MCMultiOutputObjective,
        model: Model,
        *,
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
    ) -> None:
        super().__init__()
        self.base_objective = base_objective
        self.model = model
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if X is None:
            # 念のため（通常 qNEHVI では X が渡される）
            return self.base_objective(samples, X=None)

        post = self.model.posterior(X)
        robust_samples = robustify_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
            posterior=post,
        )
        return self.base_objective(robust_samples, X=X)


# =========================================================
# 1) 単目的（ロバスト MC-UCB 風）
# =========================================================

class MCJointRobustAcquisition(MCAcquisitionFunction):
    r"""単目的向けロバスト獲得関数（MC + UCB 風 + ノイズペナルティ）.

    robust sample:
        s' = μ + beta (s - μ)
        score = s' - noise_penalty * σ(x)
    """

    def __init__(
        self,
        model: Model,
        *,
        beta: float = 2.0,
        noise_penalty: float = 2.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        sampler: Optional[SobolQMCNormalSampler] = None,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        super().__init__(model=model, sampler=sampler, **kwargs)

        # device/dtype は forward 入力に合わせて都度 to() されるので float buffer で保持
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("noise_penalty", torch.as_tensor(noise_penalty))
        self.register_buffer("default_sigma", torch.as_tensor(default_sigma))
        self.noise_is_log_var = bool(noise_is_log_var)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)  # (S, batch, q, 1) 想定

        robust = robustify_samples(
            self.model,
            X,
            samples,
            beta=float(self.beta),
            noise_penalty=float(self.noise_penalty),
            default_sigma=float(self.default_sigma),
            noise_is_log_var=self.noise_is_log_var,
            posterior=post,
        )  # (S, batch, q, 1)

        # q 内 max → MC 平均
        best_q = robust.max(dim=-2).values  # (S, batch, 1)
        return best_q.mean(dim=0).squeeze(-1)  # (batch,)


# =========================================================
# 2) 多目的（Decoupled 近似 qEHVI）
# =========================================================

class DecoupledRobustqEHVI(MCAcquisitionFunction):
    r"""多目的向け Decoupled 近似版 ロバスト qEHVI.

    注意:
      - q>1 で Inclusion-Exclusion をしない近似（元コード踏襲）
      - 厳密 qEHVI が必要なら RobustqExpectedHypervolumeImprovement を使用
    """

    def __init__(
        self,
        model: Model,
        partitioning: FastNondominatedPartitioning,
        *,
        beta: float = 2.0,
        noise_penalty: float = 2.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        sampler: Optional[SobolQMCNormalSampler] = None,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        super().__init__(
            model=model,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            **kwargs,
        )
        self.partitioning = partitioning

        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("noise_penalty", torch.as_tensor(noise_penalty))
        self.register_buffer("default_sigma", torch.as_tensor(default_sigma))
        self.noise_is_log_var = bool(noise_is_log_var)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)  # (S, batch, q, m)

        robust = robustify_samples(
            self.model,
            X,
            samples,
            beta=float(self.beta),
            noise_penalty=float(self.noise_penalty),
            default_sigma=float(self.default_sigma),
            noise_is_log_var=self.noise_is_log_var,
            posterior=post,
        )  # (S, batch, q, m)

        m = robust.shape[-1]
        # hypercell_bounds: (2, n_cells, m)
        cell_lower = self.partitioning.hypercell_bounds[0].view(1, 1, 1, -1, m)
        cell_upper = self.partitioning.hypercell_bounds[1].view(1, 1, 1, -1, m)

        pts = robust.unsqueeze(-2)  # (S, batch, q, 1, m)
        overlap_upper = torch.min(pts, cell_upper)
        overlap = (overlap_upper - cell_lower).clamp_min(0.0)
        vol = overlap.prod(dim=-1)          # (S, batch, q, n_cells)
        hvi_per_point = vol.sum(dim=-1)    # (S, batch, q)

        hvi = hvi_per_point.sum(dim=-1)    # (S, batch)
        return hvi.mean(dim=0)             # (batch,)


# =========================================================
# 3) 多目的（厳密 qEHVI をロバスト化）
# =========================================================

class RobustqExpectedHypervolumeImprovement(qExpectedHypervolumeImprovement):
    r"""ロバスト版 qEHVI（q>1 も厳密計算）。

    ロバスト化したサンプルを _compute_qehvi に投入する。
    """

    def __init__(
        self,
        model: Model,
        ref_point: Union[Tensor, list[float]],
        partitioning: FastNondominatedPartitioning,
        *,
        beta: float = 2.0,
        noise_penalty: float = 2.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: list = [],
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints if constraints is not None else [],
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )
        self.register_buffer("beta_rb", torch.as_tensor(beta))
        self.register_buffer("noise_penalty_rb", torch.as_tensor(noise_penalty))
        self.register_buffer("default_sigma_rb", torch.as_tensor(default_sigma))
        self.noise_is_log_var = bool(noise_is_log_var)
        self.eta = eta

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)  # (S, batch, q, m)

        robust = robustify_samples(
            self.model,
            X,
            samples,
            beta=float(self.beta_rb),
            noise_penalty=float(self.noise_penalty_rb),
            default_sigma=float(self.default_sigma_rb),
            noise_is_log_var=self.noise_is_log_var,
            posterior=post,
        )

        return self._compute_qehvi(samples=robust, X=X)


# =========================================================
# 4) 多目的（qNEHVI を objective ラップでロバスト化）
# =========================================================

class RobustqNEHVI(qNoisyExpectedHypervolumeImprovement):
    r"""ロバスト版 qNEHVI.

    qNEHVI は objective 経由で出力を使うため、
    RobustMCMultiOutputObjective で objective をラップしてロバスト化する。
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        X_baseline: Tensor,
        *,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: list = [],
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = True,
        marginalize_dim: Optional[int] = None,
        # robust params
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints if constraints is not None else [],
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
            marginalize_dim=marginalize_dim,
        )

        base_obj = self.objective
        self.base_objective = base_obj
        self.objective = RobustMCMultiOutputObjective(
            base_objective=base_obj,
            model=model,
            beta=beta,
            noise_penalty=noise_penalty,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
        )


# =========================================================
# 5) 多目的（ロバスト qNParEGO：ParEGO スカラー化 + ロバスト qEI）
# =========================================================

class RobustqNParEGO(MCAcquisitionFunction):
    r"""ロバスト版 qNParEGO (ParEGO + qEI).

    手順:
      1) multi-output サンプルを robustify
      2) ParEGO 重みでスカラー化（WeightedMCMultiOutputObjective）
      3) q 内 max をとって EI（best_value 超過分）を MC 期待値化
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Tensor,  # interface 互換用（m の取得に使用）
        *,
        weights: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        X_baseline = X_baseline.detach()
        tkwargs = {"dtype": X_baseline.dtype, "device": X_baseline.device}

        m = int(ref_point.numel())

        # ParEGO weights
        if weights is None:
            w = torch.rand(m, **tkwargs)
            weights = w / w.sum()
        else:
            weights = weights.to(**tkwargs)
            weights = weights / weights.sum()

        base_objective = WeightedMCMultiOutputObjective(weights=weights)

        super().__init__(model=model, sampler=sampler, objective=base_objective)

        self.base_objective = base_objective
        self.X_baseline = X_baseline
        self.ref_point = ref_point.to(**tkwargs)

        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)

        # baseline の best_value（ロバスト + スカラー化）を事前計算
        with torch.no_grad():
            robust_Y = compute_robust_train_y(
                model=model,
                train_X=X_baseline,
                noise_penalty=noise_penalty,
                default_sigma=default_sigma,
                noise_is_log_var=noise_is_log_var,
            )  # (n, m)

            # objective は (S, batch, q, m) を想定 → S=1, batch=1 で流す
            robust_Y_mc = robust_Y.unsqueeze(0).unsqueeze(0)  # (1, 1, n, m)
            obj_train = self.base_objective(
                robust_Y_mc,
                X=X_baseline.unsqueeze(0),  # (1, n, d)
            ).squeeze(0).squeeze(0)  # (n,)

            best = obj_train.max()

        self.register_buffer("best_value", best)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)  # (S, ..., q, m)

        robust = robustify_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
            posterior=post,
        )

        # ParEGO scalarization: (S, ..., q)
        scalarized = self.base_objective(robust, X=X)

        # qEI: q 内 max → baseline best_value 超過分
        best_q = scalarized.max(dim=-1).values  # (S, ...)
        improv = (best_q - self.best_value).clamp_min(0.0)
        return improv.mean(dim=0)  # (...,)
