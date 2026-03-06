import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.transforms import t_batch_mode_transform
from torch import cdist

def make_logdetlike_variance_objective(
    weights: torch.Tensor | None = None,
    eps: float = 1e-9
    ):
    def _obj(Y, X=None):
        # (1,b,q,m) の分散（unbiased=False で安定）
        if weights is not None:
            w = weights.view(1, 1, 1, -1)
            var = ((Y - Y.mean(dim=0, keepdim=True))**2 * w).mean(dim=0, keepdim=True)
        else:
            var = (Y - Y.mean(dim=0, keepdim=True)).pow(2).mean(dim=0, keepdim=True)

        score = torch.log(var + eps).sum(dim=-1)  # (1, b, q)
        return score.expand(Y.shape[0], -1, -1)   # (mc, b, q) に整形
    return GenericMCObjective(_obj)

class qMaxVarianceMultiObj(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        obj_weights: torch.Tensor = None,
        diversity_strength: float = 100.0,
        X_pending: torch.Tensor = None,
        X_observed: torch.Tensor = None,
        mc_samples: int = 256,
        objective=None,
        posterior_transform=None
    ):
        super().__init__(
            model=model,
            sampler=SobolQMCNormalSampler(torch.Size([mc_samples])),
            objective=objective,
            posterior_transform=posterior_transform
        )
        self.obj_weights = obj_weights
        self.diversity_strength = diversity_strength
        self.X_pending = X_pending
        self.X_observed = X_observed

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: shape = (..., q, d)
        Returns:
            shape = (...,)
        """
        posterior = self.model.posterior(X)
        var = posterior.variance  # (..., q, m)
        mean_var_across_q = var.mean(dim=-2)  # (..., m)

        # 多目的対応（重みあり/なし）
        if self.obj_weights is None:
            acquisition = mean_var_across_q.mean(dim=-1)
        else:
            w = self.obj_weights / self.obj_weights.sum()
            acquisition = (mean_var_across_q * w).sum(dim=-1)

        # --- 既存点との距離ペナルティ ---
        if self.X_pending is not None or self.X_observed is not None:
            X_ref_list = []
            if self.X_pending is not None:
                X_ref_list.append(self.X_pending)
            if self.X_observed is not None:
                X_ref_list.append(self.X_observed)

            if X_ref_list:
                X_ref = torch.cat(X_ref_list, dim=0)  # (N_ref, d)
                dists = cdist(X.mean(dim=-2), X_ref)  # (..., N_ref)
                min_dist = dists.min(dim=-1).values

                # ゼロ割防止
                lengthscale = torch.clamp(torch.median(dists), min=1e-6)
                penalty = torch.exp(-min_dist / (0.5 * lengthscale + 1e-8))

                acquisition = acquisition - self.diversity_strength * penalty

        # --- バッチ内距離ペナルティ（新追加） ---
        if X.size(-2) > 1:  # q > 1 のとき
            dists_within = cdist(X, X)  # (q, q)
            # 自分自身を大きな値でマスク
            dists_within += torch.eye(X.size(-2), device=X.device) * 1e6
            min_dist_within = dists_within.min(dim=-1).values
            lengthscale_within = torch.clamp(torch.median(dists_within), min=1e-6)
            penalty_within = torch.exp(-min_dist_within / (0.5 * lengthscale_within))
            acquisition = acquisition - self.diversity_strength * penalty_within

        # --- 安全ラッパー ---
        acquisition = torch.nan_to_num(acquisition, nan=0.0, posinf=1e6, neginf=0.0)
        acquisition = torch.clamp(acquisition, min=0.0)

        return acquisition