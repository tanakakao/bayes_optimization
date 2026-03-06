from typing import List, Optional, Callable, Union, Literal
import torch
from torch import Tensor

# BoTorch imports
from botorch.acquisition import qSimpleRegret
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    GenericMCObjective,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.risk_measures import VaR, CVaR
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.models.transforms.input import InputPerturbation

# 定数定義 (必要に応じて値を設定してください)
STD_DEV = 0.1
N_W = 16
# RiskTypeの型定義
RiskType = Literal["none", "var", "cvar"]

# ==========================================
# 1. 入力の摂動 (Input Perturbation)
# ========================================== 

def setup_input_perturbation(
    dim: int,
    bounds: Tensor,
    n: int = N_W,
    std: float = STD_DEV,
    perturbation: bool = False,
    cat_dims: Optional[List[int]] = None,
    **tkwargs
) -> Optional[InputPerturbation]:
    """
    入力空間における不確実性（摂動）を設定する。
    
    Args:
        dim: 入力次元数
        n: サンプル数 (n_w)
        bounds: パラメータの境界
        std: 摂動の標準偏差
        perturbation: 摂動を有効にするかどうか
        cat_dims: カテゴリカル変数のインデックスリスト（摂動を与えない）
    """
    if perturbation:
        # draw_sobol_normal_samples は botorch.utils.sampling からインポートが必要
        raw_perturbation_set = draw_sobol_normal_samples(
            d=dim,
            n=n,
            **tkwargs
        ) * std
        
        perturbation_set = raw_perturbation_set.clone()
        
        if cat_dims:
            perturbation_set[:, cat_dims] = 0.0
            
        return InputPerturbation(
            perturbation_set=perturbation_set,
            bounds=bounds,
        )
    else:
        return None