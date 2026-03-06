import torch
from torch import Tensor
from typing import List, Optional, Union
import gpytorch
from gpytorch.models.deep_gps import DeepGP
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.input import InputTransform  # ★ 追加
from botorch.utils.types import DEFAULT
from botorch.utils.transforms import normalize_indices
from .deepgp import DeepGPModel, DeepMixedGPModel
from ..layers import DeepKernelDeepGPHiddenLayer, DeepKernelDeepMixedGPHiddenLayer  # `your_module` は適切なモジュールに置き換えてください
import warnings
warnings.simplefilter('ignore')

class DeepKernelDeepGPModel(DeepGPModel):
    """
    Deep Gaussian Processモデルクラス。

    このモデルは複数の隠れ層を持つ深層ガウス過程を実現します。

    Args:
        train_X (Tensor): 訓練データの入力。
        train_Y (Tensor): 訓練データの出力。
        train_Yvar (Optional[Tensor]): 観測ノイズの分散。デフォルトはNone。
        outcome_transform (Union[str, Standardize, None]): 出力変換。デフォルトは"DEFAULT"。
        list_hidden_dims (list): 隠れ層の次元リスト。デフォルトは[10, 10]。
    """
    def __init__(
        self,
        train_X,
        train_Y,
        train_Yvar=None,
        likelihood = None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",   # ★
        outcome_transform: Union[str, OutcomeTransform, None] = "DEFAULT",  # ★
        list_hidden_dims=[10, 10],
    ):
        super().__init__(
            train_X,
            train_Y,
            train_Yvar,
            likelihood,
            input_transform,
            outcome_transform,
            list_hidden_dims,
        )
        num_outputs = train_Y.shape[-1]
        # 最終層を定義
        self.last_layer = DeepKernelDeepGPHiddenLayer(
            input_dims=list_hidden_dims[-1],
            output_dims=None if num_outputs == 1 else num_outputs,
            mean_type="constant",  # 定数平均関数を使用
        )

class DeepKernelDeepMixedGPModel(DeepMixedGPModel):
    """
    Deep Gaussian Processモデル（混合データ対応）。

    このモデルは、カテゴリデータと連続データの混在した入力を扱う深層ガウス過程モデルを実現します。

    Args:
        train_X (Tensor): 訓練データの入力。
        train_Y (Tensor): 訓練データの出力。
        cat_dims (Sequence[int]): 入力のカテゴリ次元のインデックス。
        train_Yvar (Optional[Tensor]): 観測ノイズの分散。デフォルトはNone。
        outcome_transform (Union[str, Standardize, None]): 出力変換。デフォルトは"DEFAULT"。
        list_hidden_dims (list): 隠れ層の次元リスト。デフォルトは[10, 10]。
    """
    def __init__(
        self,
        train_X,
        train_Y,
        cat_dims,
        train_Yvar=None,
        likelihood=None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",   # ★
        outcome_transform: Union[str, OutcomeTransform, None] = "DEFAULT",  # ★
        hidden_dim=10,
    ):
        super().__init__(train_X,train_Y,cat_dims,train_Yvar,likelihood,input_transform,outcome_transform,hidden_dim)

        input_dim = train_X.shape[-1]
        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.input_layer = DeepKernelDeepMixedGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dim,
            ord_dims=ord_dims,
            cat_dims=cat_dims,
            num_inducing=128,
            mean_type="linear",
            input_data=train_X
        )