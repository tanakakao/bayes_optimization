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
from ..layers import  DeepGPHiddenLayer, DeepMixedGPHiddenLayer, SkipDeepGPHiddenLayer, SkipDeepMixedGPHiddenLayer
import warnings
warnings.simplefilter('ignore')

class DeepGPModel(DeepGP, GPyTorchModel):
    """
    Deep Gaussian Processモデルクラス。

    Args:
        train_X (Tensor): 訓練データの入力。
        train_Y (Tensor): 訓練データの出力。
        train_Yvar (Optional[Tensor]): 観測ノイズの分散。
        likelihood (Optional): モデルの尤度関数。
        input_transform (Union[str, InputTransform, None]):
            入力変換。"DEFAULT" の場合は Normalize(d)。
        outcome_transform (Union[str, OutcomeTransform, None]):
            出力変換。"DEFAULT" の場合は Standardize。
        list_hidden_dims (list): 隠れ層の次元リスト。
    """
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood=None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",   # ★
        outcome_transform: Union[str, OutcomeTransform, None] = "DEFAULT",  # ★
        list_hidden_dims=[10],
        model_type="DEFAULT"
    ):
        super().__init__()

        # 入力次元数と出力次元数を取得
        input_dim = train_X.shape[-1]
        num_outputs = train_Y.shape[-1]

        # 入力データの検証
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

        # 標準化済みの Y で再チェック
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        _, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

        # BoTorch 互換: train_inputs は必ずタプル (X,) にする
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

        # 隠れ層の構築
        self.hidden_layer = torch.nn.Sequential()
        input_dims = input_dim
        for i, dim in enumerate(list_hidden_dims):
            if model_type=="skip":
                self.hidden_layer.add_module(
                    f"hidden{i}",
                    SkipDeepGPHiddenLayer(
                        input_dims=input_dims,
                        output_dims=dim,
                        mean_type="linear",
                    ),
                )
            else:
                self.hidden_layer.add_module(
                    f"hidden{i}",
                    DeepGPHiddenLayer(
                        input_dims=input_dims,
                        output_dims=dim,
                        mean_type="linear",
                    ),
                )              
            input_dims = dim  # 次の層の入力次元を更新

        # 最終層の設定
        if model_type=="skip":
            self.last_layer = SkipDeepGPHiddenLayer(
                input_dims=list_hidden_dims[-1],
                output_dims=None if num_outputs == 1 else num_outputs,
                mean_type="constant",
            )
        else:
            self.last_layer = DeepGPHiddenLayer(
                input_dims=list_hidden_dims[-1],
                output_dims=None if num_outputs == 1 else num_outputs,
                mean_type="constant",
            )
            
        # モデルの出力数と尤度を設定
        self._num_outputs = num_outputs
        if likelihood is None:
            self.likelihood = (
                get_gaussian_likelihood_with_lognormal_prior(
                    batch_shape=self._aug_batch_shape
                )
                if num_outputs == 1
                else MultitaskGaussianLikelihood(num_tasks=num_outputs)
            )
        else:
            self.likelihood = likelihood

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

    def forward(self, inputs, batch_mean: bool = True):
        """
        入力データを処理し、多変量正規分布を計算。
        """
        # GPyTorch / BoTorch から (X,) で来る場合に対応
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        # ★ 重要: 学習時のみ transform_inputs をここで適用
        # eval() -> posterior() のときは Model.posterior 側で transform_inputs が呼ばれる
        if self.training:
            inputs = self.transform_inputs(inputs)

        h = self.hidden_layer(inputs)  # 隠れ層を通過
        output = self.last_layer(h)    # 最終層を通過

        # DeepGP のサンプル平均で 1つの分布にまとめる
        mean_x = output.mean.mean(0)
        covar_x = output.covariance_matrix.mean(0)

        if batch_mean:
            if self._num_outputs > 1:
                return MultitaskMultivariateNormal(mean_x, covar_x)
            else:
                return MultivariateNormal(mean_x, covar_x)
        else:
            return output

    @property
    def num_outputs(self) -> int:
        """モデルの出力次元数を取得。"""
        return self._num_outputs

    @staticmethod
    def get_batch_dimensions(
        train_X: Tensor, train_Y: Tensor
    ) -> tuple[torch.Size, torch.Size]:
        """
        入力データのバッチ形状と、出力次元を含む拡張バッチ形状を取得。
        """
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = input_batch_shape
        num_outputs = train_Y.shape[-1]
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        return input_batch_shape, aug_batch_shape

class DeepMixedGPModel(DeepGP, GPyTorchModel):
    """
    Deep Gaussian Processモデル（混合データ対応）。

    このモデルは、カテゴリデータと連続データの混在した入力を扱う深層ガウス過程モデルを実現します。

    Args:
        train_X (Tensor): 訓練データの入力。
        train_Y (Tensor): 訓練データの出力。
        cat_dims (Sequence[int]): 入力のカテゴリ次元のインデックス。
        train_Yvar (Optional[Tensor]): 観測ノイズの分散。デフォルトはNone。
        outcome_transform (Union[str, Standardize, None]): 出力変換。デフォルトは"DEFAULT"。
        hidden_dim (int): 隠れ層の次元数。デフォルトは10。
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
        hidden_dim=8,
        model_type="DEFAULT"
    ):
        """
        モデルの初期化。

        訓練データに基づいて、隠れ層や出力層を設定します。

        Args:
            train_X (Tensor): 訓練データの特徴量。
            train_Y (Tensor): 訓練データの観測値。
            cat_dims (Sequence[int]): 入力のカテゴリ次元のインデックス。
            train_Yvar (Optional[Tensor]): 観測ノイズの分散。
            likelihood (Optional): モデルの尤度関数。
            outcome_transform (Union[str, Standardize, None]): 出力変換。
            hidden_dim (int): 隠れ層の次元数。
        """
        super().__init__()
        input_dim = train_X.shape[-1]
        num_outputs = train_Y.shape[-1]

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

        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        self.train_inputs = (train_X,)
        self.train_targets = train_Y

        # カテゴリ次元が指定されていない場合にエラーをスロー
        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        # 入力データの次元を正規化して、連続データとカテゴリカルデータを分ける
        self._ignore_X_dims_scaling_check = cat_dims
        _, self._aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        # 入力層の定義（混合データ対応）
        if model_type=="skip":
            self.input_layer = SkipDeepMixedGPHiddenLayer(
                input_dims=input_dim,
                output_dims=hidden_dim,
                ord_dims=ord_dims,
                cat_dims=cat_dims,
                num_inducing=128,
                mean_type="linear",
                input_data=train_X
            )
            # 最終層の定義
            self.last_layer = SkipDeepGPHiddenLayer(
                input_dims=hidden_dim,
                output_dims=None if num_outputs == 1 else num_outputs,
                mean_type="constant",
            )
        else:
            self.input_layer = DeepMixedGPHiddenLayer(
                input_dims=input_dim,
                output_dims=hidden_dim,
                ord_dims=ord_dims,
                cat_dims=cat_dims,
                num_inducing=128,
                mean_type="linear",
                input_data=train_X
            )            
        
            # 最終層の定義
            self.last_layer = DeepGPHiddenLayer(
                input_dims=hidden_dim,
                output_dims=None if num_outputs == 1 else num_outputs,
                mean_type="constant",
            )

        # モデルの出力数と尤度を設定
        self._num_outputs = num_outputs
        if likelihood is None:
            if num_outputs == 1:
                self.likelihood = get_gaussian_likelihood_with_lognormal_prior(
                            batch_shape=self._aug_batch_shape
                        )
            else:
                self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self.likelihood = likelihood

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
    
    def forward(self, inputs, batch_mean: bool = True):
        """
        入力データを処理し、多変量正規分布を計算。
        """
        # GPyTorch / BoTorch から (X,) で来る場合に対応
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        # ★ 重要: 学習時のみ transform_inputs をここで適用
        # eval() -> posterior() のときは Model.posterior 側で transform_inputs が呼ばれる
        if self.training:
            inputs = self.transform_inputs(inputs)

        # 入力層を通過
        h = self.input_layer(inputs)
        # 最終層を通過して出力を取得
        output = self.last_layer(h)

        # 平均と共分散を計算
        mean_x = output.mean.mean(0)
        covar_x = output.covariance_matrix.mean(0)

        # バッチ平均を計算
        if batch_mean:
            if self._num_outputs > 1:
                return MultitaskMultivariateNormal(mean_x, covar_x)
            else:
                return MultivariateNormal(mean_x, covar_x)
        else:
            return output

    @property
    def num_outputs(self) -> int:
        """
        モデルの出力次元数を取得。

        Returns:
            int: 出力次元数。
        """
        return self._num_outputs

    @staticmethod
    def get_batch_dimensions(
        train_X: Tensor, train_Y: Tensor
    ) -> tuple[torch.Size, torch.Size]:
        """
        入力データのバッチ形状と、出力次元を含む拡張バッチ形状を取得。

        Args:
            train_X (Tensor): 訓練データの特徴量 (n x d または batch_shape x n x d)。
            train_Y (Tensor): 訓練データの観測値 (n x m または batch_shape x n x m)。

        Returns:
            Tuple[torch.Size, torch.Size]:
                - 入力バッチ形状。
                - 出力次元を含む拡張バッチ形状 (input_batch_shape x (m))。
        """
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = input_batch_shape
        num_outputs = train_Y.shape[-1]
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        return input_batch_shape, aug_batch_shape