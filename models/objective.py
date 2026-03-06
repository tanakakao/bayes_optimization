from typing import List, Optional, Callable, Literal
import torch
from torch import Tensor

# BoTorch imports
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    GenericMCObjective,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.risk_measures import VaR, CVaR

# ==========================================
# 0. 定数・型定義
# ==========================================
STD_DEV = 0.1
N_W = 16
RiskType = Literal[None, "var", "cvar"]

# ==========================================
# 1. クラス定義 (Core Logic)
# ==========================================

class LinearMCObjective(MCMultiOutputObjective):
    r"""
    モデルの出力を線形変換（重み付け、符号反転）および等式制約評価を行うクラス。
    """
    def __init__(
        self,
        constraints_idx: List[int],
        weights: Tensor,
        signs: Tensor,
        eq_targets: List[Optional[float]],
    ):
        super().__init__()
        self.constraints_idx = constraints_idx
        self.register_buffer("weights", weights)
        self.register_buffer("signs", signs)
        self.eq_targets = eq_targets

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # デバイスと型を入力に合わせる（register_bufferしてあれば通常は不要だが念のため）
        weights = self.weights.to(samples)
        signs = self.signs.to(samples)

        result = []
        for i, idx in enumerate(self.constraints_idx):
            y_j = samples[..., idx]
            target = self.eq_targets[i]
            
            if target is not None:
                # 等式制約: 0に近いほど良い -> マイナス絶対値で最大化問題へ
                val = -torch.abs(y_j - target) * weights[i]
            else:
                # 通常: 最大化(sign=1) or 最小化(sign=-1)
                val = y_j * signs[i] * weights[i]
            
            result.append(val)
        
        # shape: [..., M_objectives]
        return torch.stack(result, dim=-1)


class MultiOutputRiskMeasure(MCMultiOutputObjective):  # ### FIX: 名前を変更して明確化
    r"""
    InputPerturbationによって拡大された次元(n_w)を集約し、各目的変数ごとにリスク指標を計算するラッパー。
    """
    def __init__(
        self,
        inner_obj_fn: MCMultiOutputObjective,
        n_w: int,
        alpha: float = 0.5,
        risk_type: RiskType = None,
        maximize: bool = True,
    ):
        super().__init__()
        self.inner_obj_fn = inner_obj_fn
        self.n_w = int(n_w)
        self.alpha = float(alpha)
        self.risk_type = risk_type
        self.maximize = maximize
        
        if risk_type in ["var", "cvar"] and not (0.0 < alpha <= 1.0):
             raise ValueError("alpha must be in (0.0, 1.0] for risk measures.")

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # 1. 内部の目的関数（LinearMCObjective等）を適用
        # transformed shape: [S, B, q_expanded, M_obj]
        transformed = self.inner_obj_fn.forward(samples, X)
        
        shape = transformed.shape
        q_expanded = shape[-2]

        # InputPerturbation未適用、またはRisk指定なしの場合はそのまま返す
        if self.risk_type is None:
             if q_expanded < self.n_w or q_expanded % self.n_w != 0:
                 return transformed

        # 2. 次元の分解 [..., q * n_w, M] -> [..., q, n_w, M]
        if q_expanded % self.n_w != 0:
            raise ValueError(
                f"Shape mismatch: q_expanded ({q_expanded}) must be divisible by n_w ({self.n_w})."
            )
        
        q_original = q_expanded // self.n_w
        new_shape = shape[:-2] + (q_original, self.n_w, shape[-1])
        reshaped = transformed.view(new_shape)

        # 3. リスク計算 (n_w 次元を集約)
        
        # --- Mean (RiskType: None) ---
        if self.risk_type is None:
            return reshaped.mean(dim=-2)

        # --- VaR / CVaR ---
        # maximize=True (報酬) -> 値が小さい方がリスク -> 昇順ソートして先頭(低報酬)を取得
        # maximize=False (コスト) -> 値が大きい方がリスク -> 降順ソートして先頭(高コスト)を取得
        descending = not self.maximize
        sorted_vals, _ = torch.sort(reshaped, dim=-2, descending=descending)
        
        k = int(self.n_w * self.alpha)
        if k < 1: k = 1
        
        # tail: [..., k, M] (ワースト側のk個)
        tail = sorted_vals[..., :k, :] 

        if self.risk_type == "cvar":
            return tail.mean(dim=-2)
        elif self.risk_type == "var":
            # ワーストk個の中での境界値 (tailの末尾)
            return tail[..., -1, :]
            
        raise ValueError(f"Unknown risk_type: {self.risk_type}")


# ==========================================
# 2. ファクトリ関数: 単目的 / スカラー
# ==========================================

# def get_single_objective_fn(
#     idx: int,
#     weight: float,
#     sign: float,
#     eq_target: Optional[float] = None,
# ) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
#     """
#     GenericMCObjective等で使用するスカラー変換関数。
#     戻り値: [..., q] (最後の次元は削除される)
#     """
#     def scalar_obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
#         y = Y[..., idx]
#         if eq_target is not None:
#             return -torch.abs(y - eq_target) * weight
#         else:
#             return y * sign * weight

#     return scalar_obj

def get_single_objective_fn(
    idx: int,
    weight: float,
    sign: float,
    eq_target: Optional[float] = None,
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    def scalar_obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # Y が (n,) のときは idx=0 しか意味を持たない
        if Y.ndim == 1:
            if idx != 0:
                raise ValueError(f"Y is 1D but idx={idx}. Ensure Y has shape (n, m).")
            y = Y
        else:
            y = Y[..., idx]  # (...,)

        if eq_target is not None:
            return -torch.abs(y - eq_target) * weight
        else:
            return y * sign * weight

    return scalar_obj

def get_single_objective_with_risk(
    scalar_obj_fn: Callable[[Tensor, Optional[Tensor]], Tensor],
    risk_type: RiskType = None,
    alpha: float = 0.5,
    n_w: int = N_W,
) -> MCAcquisitionObjective:
    """
    単目的最適化用のObjectiveを作成。
    """
    if risk_type is None:
        # GenericMCObjectiveは [..., q] の入力を期待するため、そのまま渡す
        return GenericMCObjective(scalar_obj_fn)

    # ### FIX: VaR/CVaR 用のラッパー
    # BoTorchのRiskMeasureクラスは [..., q, m] の形状を期待するため
    # スカラー関数が出力した [..., q] を [..., q, 1] に unsqueeze する必要がある
    def preprocessing_function(samples: Tensor) -> Tensor:
        obj_val = scalar_obj_fn(samples) # shape: [..., q*n_w]
        return obj_val     # shape: [..., q*n_w, 1]

    if risk_type == "var":
        return VaR(
            alpha=alpha,
            n_w=n_w,
            preprocessing_function=preprocessing_function,
        )
    elif risk_type == "cvar":
        return CVaR(
            alpha=alpha,
            n_w=n_w,
            preprocessing_function=preprocessing_function,
        )
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}")


# ==========================================
# 3. ファクトリ関数: 多目的
# ==========================================

# def get_multi_objective_fn(
#     idx_list: List[int],
#     weights: List[float],
#     signs: List[float],
#     eq_targets: List[Optional[float]] = None,
# ) -> LinearMCObjective:
#     if eq_targets is None:
#         eq_targets = [None] * len(idx_list)
    
#     if not (len(idx_list) == len(weights) == len(signs)):
#         raise ValueError("Inputs lists must have the same length.")

#     return LinearMCObjective(
#         constraints_idx=idx_list,
#         weights=torch.tensor(weights, dtype=torch.float),
#         signs=torch.tensor(signs, dtype=torch.float),
#         eq_targets=eq_targets,
#     )
class MultiObjective(MCMultiOutputObjective):
    def __init__(
        self,
        constraints_idx: List[int],
        weights: torch.Tensor,
        signs: torch.Tensor,
        eq_targets: List[Optional[float]]
    ):
        """
        複数の目的を扱うカスタムObjective関数。

        Args:
            constraints_idx (List[int]): 使用する目的変数のインデックス。
            weights (torch.Tensor): 各目的に対する重み (shape: [m]).
            signs (torch.Tensor): 最大化(+1) / 最小化(-1) の符号 (shape: [m]).
            eq_targets (List[Optional[float]]): 等式目標（なければ None）。
        """
        super().__init__()
        self.constraints_idx = constraints_idx
        self.weights = weights
        self.signs = signs
        self.eq_targets = eq_targets

    def forward(self, samples: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        MCサンプルに対して目的関数を評価する。

        Args:
            samples (torch.Tensor): shape [S, B, q, m] の目的サンプル

        Returns:
            torch.Tensor: shape [S, B, q, m'] の変換後目的サンプル
        """
        result = []
        for j, idx in enumerate(self.constraints_idx):
            y_j = samples[..., idx]
            if self.eq_targets[j] is not None:
                val = -torch.abs(y_j - self.eq_targets[j]) * self.weights[j]
            else:
                val = y_j * self.signs[j] * self.weights[j]
            result.append(val)
        return torch.stack(result, dim=-1)

def get_multi_objective_fn(
    idx_list: List[int],
    weights: List[float],
    signs: List[float],
    eq_targets: Optional[List[Optional[float]]] = None,
    *,
    kind: Literal["multi", "linear"] = "multi",   # ★追加：EHI/EHVIなら "multi"
    ref_tensor: Optional[Tensor] = None,          # ★追加：dtype/device参照（任意）
):
    """
    - kind="multi": 常に MultiObjective（...×q×m' を返す） -> EHI/EHVI向け
    - kind="linear": LinearMCObjective（実装次第でスカラー化の可能性） -> 1目的向け
    """
    if eq_targets is None:
        eq_targets = [None] * len(idx_list)

    if not (len(idx_list) == len(weights) == len(signs) == len(eq_targets)):
        raise ValueError("idx_list / weights / signs / eq_targets must have the same length.")

    dtype = ref_tensor.dtype if ref_tensor is not None else torch.float64
    device = ref_tensor.device if ref_tensor is not None else None

    w = torch.as_tensor(weights, dtype=dtype, device=device)
    s = torch.as_tensor(signs, dtype=dtype, device=device)

    if kind == "multi":
        # EHI/EHVI: 目的次元を潰さない objective を返す
        return MultiObjective(
            constraints_idx=idx_list,
            weights=w,
            signs=s,
            eq_targets=eq_targets,
        )

    # 1目的（PI/EI/UCB等）でスカラー化したい場合のみ
    return LinearMCObjective(
        constraints_idx=idx_list,
        weights=w,
        signs=s,
        eq_targets=eq_targets,
    )

def get_multi_objective_with_risk(
    scalar_obj_fn: MCMultiOutputObjective,
    risk_type: RiskType,
    alpha: float = 0.1,
    maximize: bool = True,
    n_w: int = N_W,
) -> MultiOutputRiskMeasure: # ### FIX: クラス名変更を反映
    return MultiOutputRiskMeasure(
        inner_obj_fn=scalar_obj_fn,
        n_w=n_w,
        alpha=alpha,
        risk_type=risk_type,
        maximize=maximize,
    )