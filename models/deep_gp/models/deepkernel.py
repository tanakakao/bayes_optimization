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
from ..layers import DeepKernel, DeepKernelMixed  # `your_module` は適切なモジュールに置き換えてください
import warnings
warnings.simplefilter('ignore')

class DeepKernelGPModel(DeepGP, GPyTorchModel):
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
        likelihood=None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",   # ★
        outcome_transform: Union[str, OutcomeTransform, None] = "DEFAULT",  # ★
    ):
        super().__init__()

        # 入力次元数と出力次元数を取得
        input_dim = train_X.shape[-1]
        num_outputs = train_Y.shape[-1] if (len(train_Y.shape)>1)&(train_Y.shape[-1]!=1) else None
        # モデルの出力数と尤度を設定
        self._num_outputs = train_Y.shape[-1] if (len(train_Y.shape)>1)&(train_Y.shape[-1]!=1) else 1
        
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        # ===== outcome_transform の設定 =====
        if isinstance(outcome_transform, str):  # ★ "DEFAULT" / "NONE" を文字列で受ける
            if outcome_transform.upper() == "DEFAULT":
                outcome_transform = Standardize(
                    m=train_Y.shape[-1],
                    batch_shape=train_X.shape[:-2],
                )
            elif outcome_transform.upper() in ("NONE", ""):
                outcome_transform = None
            else:
                raise ValueError(f"Unknown outcome_transform: {outcome_transform}")

        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
            self.outcome_transform = outcome_transform  # ★ 属性として保持

        # ===== input_transform の設定 =====
        if isinstance(input_transform, str):  # ★ "DEFAULT" / "NONE" サポート
            if input_transform.upper() == "DEFAULT":
                input_transform = Normalize(d=input_dim)
            elif input_transform.upper() in ("NONE", ""):
                input_transform = None
            else:
                raise ValueError(f"Unknown input_transform: {input_transform}")

        self.input_transform = input_transform
        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)
            _ = self.input_transform(train_X)
            self.input_transform.eval()

        # 入力データの検証
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        if train_Y.shape[-1] == 1:
            train_Y = train_Y.ravel()

        train_X = train_X.to(torch.float32)
        train_Y = train_Y.to(torch.float32)
        
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

        if likelihood is None:
            if num_outputs is None:
                self.likelihood = GaussianLikelihood()  # 単変量の場合
            else:
                self.likelihood = MultitaskGaussianLikelihood(num_tasks=train_Y.shape[-1])  # 多変量の場合

        else:
            self.likelihood = likelihood

        self.deepkernel = DeepKernel(train_X, train_Y, self.likelihood)

        # 出力変換を設定
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

    # ★ BoTorchが内部で呼び出すために必要
    def set_train_data(self, inputs=None, targets=None, strict: bool = True):
        """
        BoTorch の Posterior 計算時に呼び出されるメソッド。
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
        if targets is not None:
            self.train_targets = targets
            
    # def forward(self, inputs, batch_mean: bool = True):
    #     """
    #     入力データを処理し、多変量正規分布を計算。
    #     """
    #     # GPyTorch / BoTorch から (X,) で来る場合に対応
    #     if isinstance(inputs, tuple):
    #         inputs = inputs[0]

    #     # ★ 重要: 学習時のみ transform_inputs をここで適用
    #     # eval() -> posterior() のときは Model.posterior 側で transform_inputs が呼ばれる
    #     if self.training:
    #         inputs = self.transform_inputs(inputs)

    #     return self.deepkernel.forward(inputs)
    def forward(self, inputs, batch_mean: bool = True):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
    
        if self.training:
            # ここは prior（学習用）でOK
            inputs = self.transform_inputs(inputs)
            return self.deepkernel.forward(inputs)
        else:
            # ここが重要：posterior（条件付け）を返す
            # なお eval 時は BoTorch の posterior 側が transform_inputs を呼ぶ前提なら
            # ここで transform しなくて良い
            return self.deepkernel(inputs)

    def predict(self, inputs):
        """
        モデルによる予測を計算。

        Args:
            x (Tensor): 予測対象の入力データ。

        Returns:
            Tuple[Tensor, Tensor]: 予測の平均と分散。
        """
        preds = self.likelihood(self(inputs))
        y_pred = preds.mean
        y_var = preds.variance
        y_pred, y_var = self.outcome_transform.untransform(y_pred, y_var)
        return y_pred, y_var

    @property
    def num_outputs(self) -> int:
        """
        モデルの出力次元数を取得。

        Returns:
            int: 出力次元数。
        """
        return self._num_outputs

class DeepKernelMixedGPModel(DeepGP, GPyTorchModel):
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
        cat_dims=None,
        likelihood=None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",   # ★
        outcome_transform: Union[str, OutcomeTransform, None] = "DEFAULT",  # ★
        list_hidden_dims=[10, 10],
    ):
        super().__init__()

        # 入力次元数と出力次元数を取得
        input_dim = train_X.shape[-1]
        num_outputs = train_Y.shape[-1] if (len(train_Y.shape)>1)&(train_Y.shape[-1]!=1) else None
        # モデルの出力数と尤度を設定
        self._num_outputs = train_Y.shape[-1] if (len(train_Y.shape)>1)&(train_Y.shape[-1]!=1) else 1
        
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        # ===== outcome_transform の設定 =====
        if isinstance(outcome_transform, str):  # ★ "DEFAULT" / "NONE" を文字列で受ける
            if outcome_transform.upper() == "DEFAULT":
                outcome_transform = Standardize(
                    m=train_Y.shape[-1],
                    batch_shape=train_X.shape[:-2],
                )
            elif outcome_transform.upper() in ("NONE", ""):
                outcome_transform = None
            else:
                raise ValueError(f"Unknown outcome_transform: {outcome_transform}")

        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
            self.outcome_transform = outcome_transform  # ★ 属性として保持

        # ===== input_transform の設定 =====
        if isinstance(input_transform, str):  # ★ "DEFAULT" / "NONE" サポート
            if input_transform.upper() == "DEFAULT":
                input_transform = Normalize(d=input_dim)
            elif input_transform.upper() in ("NONE", ""):
                input_transform = None
            else:
                raise ValueError(f"Unknown input_transform: {input_transform}")

        self.input_transform = input_transform
        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)
            _ = self.input_transform(train_X)
            self.input_transform.eval()
            
        # 入力データの検証
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        if train_Y.shape[-1] == 1:
            train_Y = train_Y.ravel()

        train_X = train_X.to(torch.float32)
        train_Y = train_Y.to(torch.float32)

        self.train_inputs = (train_X,)
        self.train_targets = train_Y

        if likelihood is None:
            if num_outputs is None:
                self.likelihood = GaussianLikelihood()  # 単変量の場合
            else:
                self.likelihood = MultitaskGaussianLikelihood(num_tasks=train_Y.shape[-1])  # 多変量の場合

        else:
            self.likelihood = likelihood

        self.deepkernel = DeepKernelMixed(train_X, train_Y, cat_dims, self.likelihood)

        # 出力変換を設定
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

    # ★ BoTorchが内部で呼び出すために必要
    def set_train_data(self, inputs=None, targets=None, strict: bool = True):
        """
        BoTorch の Posterior 計算時に呼び出されるメソッド。
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
        if targets is not None:
            self.train_targets = targets
            
    # def forward(self, inputs):
    #     """
    #     入力データを処理し、多変量正規分布を計算。

    #     Args:
    #         inputs (Tensor): モデルへの入力データ。
    #         batch_mean (bool): バッチ全体の平均を計算するかどうか。デフォルトはTrue。

    #     Returns:
    #         MultivariateNormal: 平均と共分散を持つ多変量正規分布。
    #     """
    #     # GPyTorch / BoTorch から (X,) で来る場合に対応
    #     if isinstance(inputs, tuple):
    #         inputs = inputs[0]

    #     # ★ 重要: 学習時のみ transform_inputs をここで適用
    #     # eval() -> posterior() のときは Model.posterior 側で transform_inputs が呼ばれる
    #     if self.training:
    #         inputs = self.transform_inputs(inputs)
    
    #     return self.deepkernel.forward(inputs)
    def forward(self, inputs, batch_mean: bool = True):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
    
        if self.training:
            # ここは prior（学習用）でOK
            inputs = self.transform_inputs(inputs)
            return self.deepkernel.forward(inputs)
        else:
            # ここが重要：posterior（条件付け）を返す
            # なお eval 時は BoTorch の posterior 側が transform_inputs を呼ぶ前提なら
            # ここで transform しなくて良い
            return self.deepkernel(inputs)

    def predict(self, inputs):
        """
        モデルによる予測を計算。

        Args:
            x (Tensor): 予測対象の入力データ。

        Returns:
            Tuple[Tensor, Tensor]: 予測の平均と分散。
        """
        preds = self.likelihood(self.forward(inputs))
        y_pred = preds.mean
        y_var = preds.variance
        y_pred, y_var = self.outcome_transform.untransform(y_pred, y_var)
        return y_pred, y_var

    @property
    def num_outputs(self) -> int:
        """
        モデルの出力次元数を取得。

        Returns:
            int: 出力次元数。
        """
        return self._num_outputs