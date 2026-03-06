import torch
from torch import cdist
from torch.distributions import Normal
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.transforms import unnormalize, normalize
from botorch.models.model import Model

from torch import Tensor
from typing import List, Optional, Union, Tuple, Any, Dict

class LogDetqStraddle(AcquisitionFunction):
    """
    レベルセット境界探索のための獲得関数。
    目的関数の予測値がしきい値 h に近い点を好みつつ、分布の分散の「広がり（logdet）」も考慮する。

    |μ - h| を最小化するように探索し、かつ共分散行列の log det を最大化する。

    Args:
        model (Model): モデル
        beta (float): logdet にかけるスケール係数
        h (float): レベルセットのしきい値
    """

    def __init__(self, model, beta: float, h: float):
        super().__init__(model)
        self.beta = beta
        self.h = h
        self.penalty_scale = 20.0
        self.X_pending = None  # 探索済み点などによるペナルティ対象

    @t_batch_mode_transform()
    def forward(self, X):
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        covar = posterior.mvn.covariance_matrix  # 共分散行列 (batch, q, q)

        # μ から h までの距離（小さい方がよい）
        dist_to_h = torch.abs(mean - self.h)
        mean_term = -dist_to_h.sum(dim=-1)

        # log det 計算（数値安定化のため小さい値を加える）
        eps = 1e-4 * torch.eye(covar.size(-1), device=covar.device)
        chol = torch.linalg.cholesky(covar + eps)
        logdet = 2 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=-1)

        score = mean_term + self.beta * logdet

        # 探索済み点への距離によるペナルティ
        if self.X_pending is not None:
            dists = cdist(X.mean(dim=1), self.X_pending)
            min_dists = dists.min(dim=1).values
            penalty = torch.exp(-min_dists * self.penalty_scale)
            score = score - penalty

        return score

def contour_uncertainty(
    model: Model,
    X: torch.Tensor,
    h: float
) -> torch.Tensor:
    """
    予測分布における、指定値 h を閾値とする「不確実性（境界上の分布の分散）」を計算。

    Args:
        model (Model): モデル
        X (Tensor): 入力点（(batch, q, d)）
        h (float): 境界値（レベルセット）

    Returns:
        Tensor: 不確実性スコア（(batch, q)）
    """
    posterior = model.posterior(X)
    mean = posterior.mean
    std = posterior.variance.sqrt().clamp_min(1e-9)

    if mean.shape[-1] == 1:
        mean = mean.squeeze(-1)
        std = std.squeeze(-1)

    h_tensor = torch.tensor(h, dtype=mean.dtype, device=mean.device).expand_as(mean)
    normal = torch.distributions.Normal(mean, std)
    prob = normal.cdf(h_tensor)
    return prob * (1 - prob)

class qICUAcquisition(AcquisitionFunction):
    """
    Integrated Contour Uncertainty (ICU) に基づくレベルセット探索獲得関数。

    Args:
        model (Model): モデル
        h (float): レベルセットのしきい値
        reduction (str): q点に対する集約方法（"mean" または "sum"）
    """

    def __init__(self, model: Model, h: float, reduction: str = "mean"):
        super().__init__(model)
        self.h = h
        self.reduction = reduction
        self.X_pending = None
        self.penalty_scale = 50.0

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        score = contour_uncertainty(self.model, X, self.h)

        if self.X_pending is not None:
            dists = cdist(X.mean(dim=1), self.X_pending)
            penalty = torch.exp(-dists.min(dim=1).values * self.penalty_scale)
            score = score - penalty

        if self.reduction == "mean":
            return score.mean(dim=-1)
        elif self.reduction == "sum":
            return score.sum(dim=-1)
        else:
            raise ValueError("Unknown reduction method")

class qStraddle(AcquisitionFunction):
    """
    Straddle 方式に基づくレベルセット探索の獲得関数。

    予測平均が h に近く、標準偏差が大きい点を好む。

    Args:
        model (Model): モデル
        beta (float): 標準偏差への重み
        h (float): レベルセットのしきい値
    """

    def __init__(self, model, beta: float, h: float):
        super().__init__(model)
        self.beta = beta
        self.h = h
        self.X_pending = None

    @t_batch_mode_transform()
    def forward(self, X):
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1)
        score = -torch.abs(mean - self.h) + self.beta * std
        score = score.mean(dim=-1)

        if self.X_pending is not None:
            dists = cdist(X.mean(dim=1), self.X_pending)
            penalty = torch.exp(-dists.min(dim=1).values * 100.0)
            score = score - penalty

        return score

class qJointBoundaryVariance(AcquisitionFunction):
    """
    複数目的に対する「レベルセット境界上の同時不確実性（分散）」を評価する獲得関数。

    各目的関数に対して h_i を指定し、それぞれの CDF を使って成功確率を計算、
    それらの積から分散を導出。

    Args:
        model (Model): モデル（複数出力を想定）
        h (List[float]): 各目的に対するレベルセット境界
    """

    def __init__(self, model, h: List[float]):
        super().__init__(model)
        self.h = torch.tensor(h, dtype=torch.float32)
        self.X_pending = None
        self.penalty_scale = 20.0

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean                  # (batch, q, m)
        sigma = posterior.variance.clamp_min(1e-9).sqrt()

        probs = []
        for i in range(self.h.numel()):
            mean_i = mean[..., i]
            sigma_i = sigma[..., i]
            z = (mean_i - self.h[i]) / sigma_i
            p_i = torch.distributions.Normal(0, 1).cdf(z)
            probs.append(p_i.prod(dim=-1))     # 各 q 点での成功確率の積

        joint_prob = torch.stack(probs, dim=-1).prod(dim=-1)  # 各目的の積
        variance = joint_prob * (1 - joint_prob)              # Bernoulli 分布の分散

        if self.X_pending is not None:
            dists = torch.cdist(X.mean(dim=1), self.X_pending)
            penalty = torch.exp(-dists.min(dim=1).values * self.penalty_scale)
            variance = variance - penalty

        return variance

class LogDetqStraddleMultiCommon(AcquisitionFunction):
    """
    多目的分類または回帰モデルに対する共通領域の境界探索用 LogDetqStraddle。

    - 各目的の予測平均がしきい値以上である領域を探索
    - 予測共分散の log det を加味して不確実性も考慮

    Args:
        model: BoTorch互換の多出力モデル（posterior.mean ∈ (0,1) or 実数）
        beta: 共分散の logdet への重み
        thresholds: 各目的のしきい値 (shape: (m,))
    """

    def __init__(self, model, beta: float, h: List[float]):
        super().__init__(model)
        self.beta = beta
        self.thresholds = torch.tensor(h).view(1, 1, -1)  # (1, 1, m)
        self.X_pending = None
        self.penalty_scale = 20.0

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean  # (B, q, m)
        covar = posterior.mvn.covariance_matrix  # (B, q*m, q*m)

        # --- (1) 共通満足度スコア ---
        # 各目的に対して h_i - μ_i を計算（正のときだけ罰：=しきい値未満）
        margin = torch.relu(self.thresholds.to(X.device) - mean)  # (B, q, m)
        mean_term = -margin.sum(dim=(1, 2))  # 満足していればゼロに近づく → 高スコア

        # --- (2) log det（共分散行列の広がり） ---
        eps = 1e-4 * torch.eye(covar.size(-1), device=covar.device)
        chol = torch.linalg.cholesky(covar + eps)
        logdet = 2 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=-1)

        # --- (3) スコア計算 ---
        score = mean_term + self.beta * logdet

        # --- (4) 探索済み点への距離によるペナルティ ---
        if self.X_pending is not None:
            dists = cdist(X.mean(dim=1), self.X_pending)
            min_dists = dists.min(dim=1).values
            penalty = torch.exp(-min_dists * self.penalty_scale)
            score = score - penalty

        return score

class qStraddleMultiCommon(AcquisitionFunction):
    """
    多目的の共通満足領域（すべての目的がしきい値以上）を探索するための qStraddle 獲得関数。

    - ReLU(threshold - mean) により、しきい値を満たしていない部分に罰を与える
    - 分散が大きいほど不確実性が高く好ましい

    Args:
        model (Model): BoTorch互換の多出力モデル（posterior.mean ∈ ℝ）
        beta (float): 分散へのスケール係数
        thresholds (Tensor): 各目的のしきい値 (shape: (m,))
    """

    def __init__(self, model, beta: float, h: List[float]):
        super().__init__(model)
        self.beta = beta
        self.thresholds = torch.tensor(h).view(1, 1, -1)  # (1, 1, m)
        self.X_pending = None
        self.penalty_scale = 10.0

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean  # (B, q, m)
        std = posterior.variance.clamp_min(1e-9).sqrt()  # (B, q, m)

        # --- 共通満足度スコア（しきい値未満にペナルティ） ---
        margin = torch.relu(self.thresholds.to(X.device) - mean)  # (B, q, m)
        mean_term = -margin.sum(dim=(1, 2))  # すべて満たせば 0、満たさないほどマイナス

        # --- 標準偏差スコア（大きいほど良い） ---
        std_term = std.sum(dim=(1, 2))

        score = mean_term + self.beta * std_term  # (B,)

        # --- ペナルティ（既探索点に近い場合） ---
        if self.X_pending is not None:
            dists = cdist(X.mean(dim=1), self.X_pending)
            penalty = torch.exp(-dists.min(dim=1).values * self.penalty_scale)
            score = score - penalty

        return score