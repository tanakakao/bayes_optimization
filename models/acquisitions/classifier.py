import torch
from torch import Tensor
from torch import cdist
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition.bayesian_active_learning import qBayesianActiveLearningByDisagreement
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.model_list_gp_regression import ModelListGPyTorchModel

# class BALDAcquisition(AcquisitionFunction):
#     """
#     Bayesian Active Learning by Disagreement (BALD) に基づく獲得関数。

#     モデル予測のエントロピー（平均）と条件付きエントロピーの差により、
#     モデルが「最も意見が分かれている」点を選択する。

#     Args:
#         model: BoTorch モデル（rsample が実装された分類用モデルを想定）
#         num_samples (int): サンプル数（Monte Carlo 推定用）
#         reduction (str): バッチ内のスコアの集約方法 ("mean" または "sum")
#     """

#     def __init__(
#         self,
#         model,
#         num_samples: int = 16,
#         reduction: str = "mean"
#     ):
#         if isinstance(model, ModelListGP)|isinstance(model, ModelListGPyTorchModel):
#             model = model.models[0]
#         super().__init__(model)
#         self.num_samples = num_samples
#         self.reduction = reduction
#         self.X_pending = None  # 探索済み点へのペナルティ対象

#     @t_batch_mode_transform()
#     def forward(
#         self,
#         X: Tensor
#     ) -> Tensor:
        
#         self.model.eval()
#         self.model.likelihood.eval()

#         # モデルの事後分布から num_samples 個のサンプルを取得
#         dist = self.model(X)
#         f_samples = dist.rsample(torch.Size([self.num_samples]))  # shape: (S, B, q)

#         # sigmoid を通して確率に変換（分類）
#         probs = f_samples.clamp(1e-6, 1 - 1e-6)  # shape: (S, B, q)

#         # 条件付きエントロピー（各サンプルのエントロピーを平均）
#         entropy_per_sample = - (probs * probs.log() + (1 - probs) * (1 - probs).log())  # (S, B, q)
#         entropy_conditional = entropy_per_sample.mean(dim=0)  # (B, q)

#         # 平均予測に基づくエントロピー（情報利得との比較用）
#         mean_prob = probs.mean(dim=0)
#         mean_entropy = - (mean_prob * mean_prob.log() + (1 - mean_prob) * (1 - mean_prob).log())

#         # BALD スコア = 平均エントロピー - 条件付きエントロピー
#         bald_score = mean_entropy - entropy_conditional  # shape: (B, q)
        
#         # 🔁 探索済み点との距離によるペナルティ（多様性確保）
#         if self.X_pending is not None:
#             dists = cdist(X.mean(dim=1), self.X_pending)        # (B, N_pending)
#             min_dist = dists.min(dim=1).values                  # (B,)
#             penalty = torch.exp(-min_dist * 10.0)               # 小さいほど強くペナルティ
#             bald_score = bald_score - penalty.unsqueeze(-1)     # shape: (B, q)

#         # q点（複数クエリ）を集約
#         if self.reduction == "mean":
#             return bald_score.mean(dim=-1)  # shape: (B,)
#         elif self.reduction == "sum":
#             return bald_score.sum(dim=-1)
#         else:
#             raise ValueError("Unknown reduction")

class BALDAcquisition(AcquisitionFunction):
    """
    BALD: H[ E_y p(y|x, w) ] - E_w H[p(y|x, w)] を MC で推定（binary）。

    前提：
      - self.model が「分類用ラッパー」で、潜在GPが self.model.model、
        尤度が self.model.likelihood としてアクセスできることを推奨
      - それ以外の場合は下の _get_latent_and_transform を調整してください
    """

    def __init__(self, model, num_samples: int = 16, reduction: str = "mean"):
        # ModelList の場合は 1つ目を使う（あなたの既存方針を踏襲）
        if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):
            model = model.models[0]

        super().__init__(model)
        self.num_samples = int(num_samples)
        self.reduction = reduction
        self.set_X_pending(None)  # ★BoTorch互換（sequential最適化で内部から設定される）

    def _get_latent_and_transform(self, X: Tensor) -> Tensor:
        """
        X を input_transform したうえで flatten し、潜在GPの分布を返す。
        """
        # X: batch_shape x q x d（t_batch_mode_transform 後）
        # 2D が来たら q=1 扱いに寄せる（保険）
        if X.ndim == 2:
            X = X.unsqueeze(-2)

        # input_transform があるなら適用（ラッパー/BoTorch標準の両対応）
        Xt = X
        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)

        orig = Xt.shape[:-1]             # = batch_shape x q
        Xf = Xt.reshape(-1, Xt.shape[-1])  # (B_total*q, d)

        # 潜在GPにアクセス（ラッパー想定）
        latent_gp = getattr(self.model, "model", None)
        if latent_gp is None:
            # もし self.model 自体が gpytorch model ならこちら
            latent_gp = self.model

        latent_dist = latent_gp(Xf)  # gpytorch distribution
        return latent_dist, orig

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # 推論モード
        self.model.eval()
        like = getattr(self.model, "likelihood", None)
        if like is not None:
            like.eval()

        latent_dist, orig = self._get_latent_and_transform(X)
        # latent_dist は (B_total*q) 点に対する分布
        # rsample -> (S, B_total*q)
        f_samples = latent_dist.rsample(torch.Size([self.num_samples]))

        # logits -> prob
        probs = torch.sigmoid(f_samples).clamp(1e-6, 1 - 1e-6)

        # (S, B_total*q) -> (S, *batch_shape, q)
        probs = probs.reshape(self.num_samples, *orig)

        # 条件付きエントロピー E_w[ H(p(y|x,w)) ]
        ent_each = -(probs * probs.log() + (1 - probs) * (1 - probs).log())  # (S, *batch, q)
        entropy_conditional = ent_each.mean(dim=0)                             # (*batch, q)

        # 予測平均に基づくエントロピー H(E_w[p])
        mean_prob = probs.mean(dim=0)                                          # (*batch, q)
        mean_entropy = -(mean_prob * mean_prob.log() + (1 - mean_prob) * (1 - mean_prob).log())

        bald_score = mean_entropy - entropy_conditional                        # (*batch, q)

        # X_pending ペナルティ（shape を壊さない）
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            # 候補の代表点：(*batch, d)
            Xc = X.mean(dim=-2)

            # X_pending を (N_pending_flat, d) に潰す
            Xp2d = Xp.reshape(-1, Xp.shape[-1])

            # torch.cdist は 2D を要求するので flatten
            Xc2d = Xc.reshape(-1, Xc.shape[-1])                     # (B_total, d)
            dists = torch.cdist(Xc2d, Xp2d)                          # (B_total, Np)
            min_dist = dists.min(dim=-1).values                      # (B_total,)
            penalty = torch.exp(-min_dist * 10.0).reshape(*orig[:-1]) # (*batch,)

            bald_score = bald_score - penalty.unsqueeze(-1)          # (*batch, q)

        # q 集約（必ず batch_shape を返す）
        if self.reduction == "mean":
            out = bald_score.mean(dim=-1)   # (*batch,)
        elif self.reduction == "sum":
            out = bald_score.sum(dim=-1)    # (*batch,)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # shape 契約チェック：out は X.shape[:-2] と一致
        expected = X.shape[:-2]
        if out.shape != expected:
            raise RuntimeError(f"BALD output shape mismatch: expected {tuple(expected)}, got {tuple(out.shape)}")
        return out

# class StraddleClassifierAcquisition(AcquisitionFunction):
#     """
#     分類GPモデルにおけるStraddle Acquisition Function。
#     確率の不確実性（p * (1 - p)）を指標として最大化する。

#     Args:
#         model (Model): BoTorch互換の分類モデル（posterior.mean ∈ (0,1)）
#         reduction (str): "mean" or "sum"（q > 1 時の集約方法）
#     """

#     def __init__(
#         self,
#         model,
#         reduction: str = "mean"
#     ):
#         super().__init__(model)
#         self.reduction = reduction
#         self.X_pending = None

#     @t_batch_mode_transform()
#     def forward(self, X: Tensor) -> Tensor:
#         # 確率予測
#         posterior = self.model.posterior(X)
#         p = posterior.mean.squeeze(-1).clamp(1e-6, 1 - 1e-6)  # shape: (B, q)

#         # Straddle score: 不確実性 (p * (1 - p))
#         score = p * (1 - p)  # shape: (B, q)
#         # score = score[:,torch.newaxis]
        
#         # 近傍ペナルティ（任意）
#         if self.X_pending is not None:
#             dists = cdist(X.mean(dim=1), self.X_pending)  # shape: (B, N_pending)
#             min_dist = dists.min(dim=1).values
#             penalty = torch.exp(-min_dist * 10.0)
#             score = score - penalty.unsqueeze(-1)         # shape: (B, q)

#         # q点の集約
#         if self.reduction == "mean":
#             return score.mean(dim=-1)  # shape: (B,)
#         elif self.reduction == "sum":
#             return score.sum(dim=-1)
#         else:
#             raise ValueError(f"Unknown reduction mode: {self.reduction}")

class StraddleClassifierAcquisition(AcquisitionFunction):
    def __init__(self, model, reduction: str = "mean"):
        super().__init__(model)
        self.reduction = reduction
        self.X_pending = None

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # X: batch_shape x q x d
        posterior = self.model.posterior(X)
        p = posterior.mean

        # 多くの分類 posterior は最後に output 次元が付く想定（... x 1）
        if p.ndim >= 1 and p.shape[-1] == 1:
            p = p.squeeze(-1)

        p = p.clamp(1e-6, 1 - 1e-6)

        score = p * (1 - p)  # 形はモデル次第で (batch_shape, q) か (batch_shape,)

        # 近傍ペナルティ（shape を壊さない実装）
        if getattr(self, "X_pending", None) is not None:
            Xc = X.mean(dim=-2)  # batch_shape x d

            Xp = self.X_pending
            # X_pending は (n, d) / (n, 1, d) / (n, q, d) 等があり得るので潰す
            if Xp.ndim >= 2:
                Xp = Xp.reshape(-1, Xp.shape[-1])  # (N_pending_flat, d)

            dists = torch.cdist(Xc, Xp)  # batch_shape x N_pending_flat
            min_dist = dists.min(dim=-1).values  # batch_shape
            penalty = torch.exp(-min_dist * 10.0)  # batch_shape

            # score が (batch_shape, q) のときだけ q 側にブロードキャストして引く
            if score.ndim == X.ndim - 1:  # = batch_shape + q
                score = score - penalty.unsqueeze(-1)
            else:
                score = score - penalty  # score が batch_shape のみならそのまま

        # --- ここが本件の修正点 ---
        # score.ndim == X.ndim - 1 なら q 次元あり（batch_shape x q）なので reduction
        # score.ndim == X.ndim - 2 なら既に batch_shape（q が潰れている）なので reduction しない
        if score.ndim == X.ndim - 1:
            if self.reduction == "mean":
                out = score.mean(dim=-1)
            elif self.reduction == "sum":
                out = score.sum(dim=-1)
            else:
                raise ValueError(f"Unknown reduction mode: {self.reduction}")
        elif score.ndim == X.ndim - 2:
            out = score
        else:
            raise RuntimeError(
                f"Unexpected score shape. X.shape={tuple(X.shape)}, score.shape={tuple(score.shape)}"
            )

        # 最終契約チェック：out は batch_shape と一致する必要がある
        batch_shape = X.shape[:-2]
        if out.shape != batch_shape:
            raise RuntimeError(
                f"Acqf output must have shape {batch_shape}, got {tuple(out.shape)} "
                f"(X.shape={tuple(X.shape)}, score.shape={tuple(score.shape)})"
            )
        return out

# class EntropyClassifierAcquisition(AcquisitionFunction):
#     """
#     分類予測におけるエントロピーに基づく獲得関数。

#     最も不確実性の高い（確率が0.5に近い）点を選びやすくなる。
#     単一の予測平均を使うので、計算は軽い。

#     Args:
#         model (Model): 予測確率を返すモデル（Posterior.mean ∈ (0, 1) を想定）
#         reduction (str): q点の集約方法 ("mean" または "sum")
#     """

#     def __init__(
#         self,
#         model,
#         reduction: str = "mean"
#     ):
#         super().__init__(model)
#         self.reduction = reduction
#         self.X_pending = None

#     @t_batch_mode_transform()
#     def forward(self, X: Tensor) -> Tensor:
#         # モデルの予測確率（平均）を取得
#         posterior = self.model.posterior(X)
#         prob = posterior.mean.squeeze(-1).clamp(1e-6, 1 - 1e-6)  # shape: (B, q)

#         # Bernoulli エントロピー計算
#         entropy = - (prob * prob.log() + (1 - prob) * (1 - prob).log())  # shape: (B, q)
#         # entropy = entropy[:,torch.newaxis]

#         # 🔁 X_pending によるペナルティ（距離が近い点を抑制）
#         if self.X_pending is not None:
#             dists = cdist(X.mean(dim=1), self.X_pending)       # (B, N_pending)
#             min_dist = dists.min(dim=1).values
#             penalty = torch.exp(-min_dist * 10.0)
#             entropy = entropy - penalty.unsqueeze(-1)          # shape: (B, q)

#         # q点を集約
#         if self.reduction == "mean":
#             return entropy.mean(dim=-1)  # 明示的に shape: (B,)
#         elif self.reduction == "sum":
#             return entropy.sum(dim=-1)
#         else:
#             raise ValueError(f"Unknown reduction mode: {self.reduction}")

class EntropyClassifierAcquisition(AcquisitionFunction):
    """
    分類予測におけるエントロピーに基づく獲得関数。

    Args:
        model (Model): 予測確率を返すモデル（Posterior.mean ∈ (0, 1) を想定）
        reduction (str): q点の集約方法 ("mean" または "sum")
    """

    def __init__(self, model, reduction: str = "mean"):
        super().__init__(model)
        self.reduction = reduction
        self.set_X_pending(None)  # <- ここが重要（AttributeError防止 + BoTorch互換）

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # X: batch_shape x q x d
        posterior = self.model.posterior(X)
        prob = posterior.mean

        # posterior.mean が ... x 1 の場合は最後だけ落とす（バッチやqを潰さない）
        if prob.ndim >= 1 and prob.shape[-1] == 1:
            prob = prob.squeeze(-1)

        prob = prob.clamp(1e-6, 1 - 1e-6)

        # Bernoulli entropy
        entropy = -(prob * prob.log() + (1 - prob) * (1 - prob).log())

        # X_pending ペナルティ（shapeを壊さない）
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            # X の代表点（q方向を平均）：batch_shape x d
            Xc = X.mean(dim=-2)

            # X_pending を (N_pending_flat, d) に整形
            if Xp.ndim >= 2:
                Xp2d = Xp.reshape(-1, Xp.shape[-1])
            else:
                # 異常系：ここには通常来ない想定
                raise RuntimeError(f"Unexpected X_pending shape: {tuple(Xp.shape)}")

            dists = torch.cdist(Xc, Xp2d)              # batch_shape x N_pending_flat
            min_dist = dists.min(dim=-1).values        # batch_shape
            penalty = torch.exp(-min_dist * 10.0)      # batch_shape

            # entropy が (batch_shape, q) のときは q にブロードキャストして引く
            if entropy.ndim == X.ndim - 1:  # = batch_shape + q
                entropy = entropy - penalty.unsqueeze(-1)
            else:
                # entropy がすでに batch_shape のみならそのまま引く
                entropy = entropy - penalty

        # --- shape-safe reduction ---
        # entropy.ndim == X.ndim - 1 なら q 次元あり（batch_shape x q）なので reduction
        # entropy.ndim == X.ndim - 2 なら既に batch_shape（q が潰れている）なので reduction しない
        if entropy.ndim == X.ndim - 1:
            if self.reduction == "mean":
                out = entropy.mean(dim=-1)
            elif self.reduction == "sum":
                out = entropy.sum(dim=-1)
            else:
                raise ValueError(f"Unknown reduction mode: {self.reduction}")
        elif entropy.ndim == X.ndim - 2:
            out = entropy
        else:
            raise RuntimeError(
                f"Unexpected entropy shape. X.shape={tuple(X.shape)}, entropy.shape={tuple(entropy.shape)}"
            )

        # 最終契約チェック：out は batch_shape と一致する必要がある
        batch_shape = X.shape[:-2]
        if out.shape != batch_shape:
            raise RuntimeError(
                f"Acqf output must have shape {batch_shape}, got {tuple(out.shape)} "
                f"(X.shape={tuple(X.shape)}, entropy.shape={tuple(entropy.shape)})"
            )
        return out

class BALDMultiOutputAcquisition(AcquisitionFunction):
    """
    多出力分類モデルにおいて「全ての出力が1になる同時確率」のBALDスコアを計算。

    各出力は独立と仮定し、p_all = Π_i p_i に対してエントロピー差（BALD）を評価する。
    """

    def __init__(
        self,
        model,
        num_samples: int = 16,
        reduction: str = "mean",
        penalty_scale: float = 10.0,
    ):
        super().__init__(model)
        self.num_samples = num_samples
        self.reduction = reduction
        self.penalty_scale = penalty_scale
        self.X_pending = None

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self.model.eval()

        posterior = self.model.posterior(X)
        f_samples = posterior.rsample(torch.Size([self.num_samples]))  # shape: (S, B, q, m)
        probs = f_samples.clamp(1e-6, 1 - 1e-6)          # shape: (S, B, q, m)

        # 同時確率 p_all = ∏_m p_i
        p_all = probs.prod(dim=-1)  # shape: (S, B, q)

        # エントロピー項：-p log p - (1 - p) log (1 - p)
        def binary_entropy(p: Tensor) -> Tensor:
            return - (p * p.log() + (1 - p) * (1 - p).log())

        # 条件付きエントロピーの平均（各サンプルのエントロピーの平均）
        entropy_conditional = binary_entropy(p_all).mean(dim=0)  # shape: (B, q)

        # 平均確率に対するエントロピー
        p_mean = p_all.mean(dim=0)  # shape: (B, q)
        mean_entropy = binary_entropy(p_mean)  # shape: (B, q)

        # BALDスコア
        bald_score = mean_entropy - entropy_conditional  # shape: (B, q)

        # 距離ペナルティ（X_pending）
        if self.X_pending is not None:
            dists = cdist(X.mean(dim=1), self.X_pending)  # shape: (B, N_pending)
            penalty = torch.exp(-dists.min(dim=1).values * self.penalty_scale)  # (B,)
            bald_score = bald_score - penalty.unsqueeze(-1)

        # q点の集約
        if self.reduction == "mean":
            return bald_score.mean(dim=-1)  # shape: (B,)
        elif self.reduction == "sum":
            return bald_score.sum(dim=-1)
        else:
            raise ValueError(f"Unknown reduction mode: {self.reduction}")

# class JointStraddleClassifierAcquisition(AcquisitionFunction):
#     """
#     多出力分類GPモデルにおけるStraddle Acquisition Function（共通領域版）。
#     各出力が1となる同時確率の不確実性（p_joint * (1 - p_joint)）を最大化する。

#     Args:
#         model (Model): BoTorch互換の分類モデル（posterior.mean ∈ (0,1), shape: (B, q, m)）
#         reduction (str): "mean" or "sum"（q > 1 時の集約方法）
#     """

#     def __init__(
#         self,
#         model,
#         reduction: str = "mean",
#         penalty_scale: float = 10.0,
#     ):
#         super().__init__(model)
#         self.reduction = reduction
#         self.penalty_scale = penalty_scale
#         self.X_pending = None

#     @t_batch_mode_transform()
#     def forward(self, X: Tensor) -> Tensor:
#         # モデル予測：p.shape = (B, q, m)
#         posterior = self.model.posterior(X)
#         p = posterior.mean.clamp(1e-6, 1 - 1e-6)

#         # 各点に対する「すべて1になる確率」: p_joint = ∏ p_i
#         p_joint = p.prod(dim=-1)  # shape: (B, q)

#         # その不確実性（Straddle-like）：最大で p_joint ≈ 0.5 のとき
#         joint_uncertainty = p_joint * (1 - p_joint)  # shape: (B, q)

#         # 近傍ペナルティ
#         if self.X_pending is not None:
#             dists = cdist(X.mean(dim=1), self.X_pending)
#             min_dist = dists.min(dim=1).values
#             penalty = torch.exp(-min_dist * self.penalty_scale)
#             joint_uncertainty = joint_uncertainty - penalty.unsqueeze(-1)

#         # q点集約
#         if self.reduction == "mean":
#             return joint_uncertainty.mean(dim=-1)
#         elif self.reduction == "sum":
#             return joint_uncertainty.sum(dim=-1)
#         else:
#             raise ValueError(f"Unknown reduction mode: {self.reduction}")

def _to_batch_q_m(p: Tensor, X: Tensor) -> Tensor:
    """
    posterior.mean の形が揺れても必ず (batch_shape, q, m_flat) に変換する。

    想定して吸収する例:
      (B,q,m), (B,q,1,m), (B,m,q), (B,q,m,1), (B,q,1,m,1), ...
      ※ q=1 の場合も含む（今回の (B,m,1) を (B,1,m) に直す）
    """
    q = X.shape[-2]
    batch_shape = X.shape[:-2]
    bnd = len(batch_shape)

    # 末尾の singleton は全部落とす（ただし q=1 をここで消すのは後で復元できる）
    while p.ndim > 0 and p.shape[-1] == 1:
        p = p.squeeze(-1)

    # batch 次元を X に合わせて expand（1 であれば拡張可能）
    if p.ndim < bnd + 1:
        raise RuntimeError(f"posterior.mean has too few dims: p.shape={tuple(p.shape)}, X.shape={tuple(X.shape)}")

    if tuple(p.shape[:bnd]) != tuple(batch_shape):
        # broadcast 可能なら expand
        ok = True
        for a, b in zip(p.shape[:bnd], batch_shape):
            if not (a == b or a == 1):
                ok = False
                break
        if not ok:
            # ここが出る場合は「モデルが batch model 扱い」になっている可能性が高い
            raise RuntimeError(f"Batch dims mismatch: p.shape={tuple(p.shape)}, X batch={tuple(batch_shape)}")
        p = p.expand(*batch_shape, *p.shape[bnd:])

    tail = p.shape[bnd:]  # batch 以降

    # tail の中から "q 次元" を決める。
    # q=1 でも迷わないよう、「その軸を q とみなしたときに残りの m が最大になる軸」を選ぶ。
    cand = [i for i, s in enumerate(tail) if s == q]
    if not cand:
        raise RuntimeError(f"Cannot locate q-dim={q} in p.shape={tuple(p.shape)}")

    def rem_prod(i: int) -> int:
        prod = 1
        for j, s in enumerate(tail):
            if j != i:
                prod *= s
        return prod

    q_axis_in_tail = max(cand, key=rem_prod)  # 残りが最大になる軸を q とする

    # q 軸を -2 に移動し、残りを m_flat に畳む
    q_axis = bnd + q_axis_in_tail
    p = p.movedim(q_axis, -2).contiguous()  # q を末尾から2番目へ

    # ここで p は (batch_shape, ..., q, ...) になっているので、
    # q 以外の残りの tail を全部まとめて m_flat にする
    p = p.reshape(*batch_shape, q, -1)  # (batch_shape, q, m_flat)

    return p

class JointStraddleClassifierAcquisition(AcquisitionFunction):
    def __init__(self, model, reduction: str = "mean", penalty_scale: float = 10.0):
        super().__init__(model)
        self.reduction = reduction
        self.penalty_scale = float(penalty_scale)
        self.set_X_pending(None)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        p = posterior.mean.clamp(1e-6, 1 - 1e-6)

        # ★ここで必ず (batch_shape, q, m_flat) に揃える
        p = _to_batch_q_m(p, X)

        # joint（m_flat 方向に積）
        p_joint = p.prod(dim=-1)                    # (batch_shape, q)
        joint_uncertainty = p_joint * (1.0 - p_joint)

        # X_pending penalty（必要なら）
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            Xc = X.mean(dim=-2)                     # (batch_shape, d)
            Xc2d = Xc.reshape(-1, Xc.shape[-1])
            Xp2d = Xp.reshape(-1, Xp.shape[-1])
            dists = torch.cdist(Xc2d, Xp2d)
            min_dist = dists.min(dim=-1).values.reshape(*X.shape[:-2])
            penalty = torch.exp(-min_dist * self.penalty_scale)
            joint_uncertainty = joint_uncertainty - penalty.unsqueeze(-1)

        # q 集約 → batch_shape
        if self.reduction == "mean":
            out = joint_uncertainty.mean(dim=-1)
        elif self.reduction == "sum":
            out = joint_uncertainty.sum(dim=-1)
        else:
            raise ValueError(f"Unknown reduction mode: {self.reduction}")

        # t_batch_mode_transform 契約：out は X.shape[:-2]（例: (B,)）で返す
        expected = X.shape[:-2]
        while out.ndim > len(expected) and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if out.shape != expected:
            # ここに入る場合は「モデルが batch model 扱い」(model.batch_shapeが空でない) の可能性が高い
            raise RuntimeError(f"Acqf output shape mismatch: expected {tuple(expected)}, got {tuple(out.shape)}")

        return out

class EntropyMultiOutputAcquisition(AcquisitionFunction):
    """
    多出力分類モデルにおける「全ての出力が1となる確率」のエントロピーに基づく獲得関数。

    各出力が独立と仮定し、joint probability: p_joint = Π_i p_i
    そのエントロピーをスコアとする。

    Args:
        model (Model): BoTorch互換の分類モデル（posterior.mean ∈ (0,1), shape: (B, q, m)）
        reduction (str): q点の集約方法 ("mean" または "sum")
    """

    def __init__(
        self,
        model,
        reduction: str = "mean",
        penalty_scale: float = 10.0
    ):
        super().__init__(model)
        self.reduction = reduction
        self.penalty_scale = penalty_scale
        self.X_pending = None

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        prob = posterior.mean.clamp(1e-6, 1 - 1e-6)  # shape: (B, q, m)

        # 🔁 各出力が「同時に 1」となる確率
        joint_prob = prob.prod(dim=-1)  # shape: (B, q)

        # 🔁 Bernoulli エントロピー： H(p) = -p log p - (1 - p) log(1 - p)
        entropy = - (joint_prob * joint_prob.log() + (1 - joint_prob) * (1 - joint_prob).log())  # shape: (B, q)

        # 近傍ペナルティ（任意）
        if self.X_pending is not None:
            dists = cdist(X.mean(dim=1), self.X_pending)  # (B, N_pending)
            penalty = torch.exp(-dists.min(dim=1).values * self.penalty_scale)
            entropy = entropy - penalty.unsqueeze(-1)     # shape: (B, q)

        # q点の集約
        if self.reduction == "mean":
            return entropy.mean(dim=-1)  # shape: (B,)
        elif self.reduction == "sum":
            return entropy.sum(dim=-1)
        else:
            raise ValueError(f"Unknown reduction mode: {self.reduction}")