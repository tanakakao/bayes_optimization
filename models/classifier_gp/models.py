import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.models.deep_gps import DeepGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel

from botorch.models.model import Model
from botorch.models.utils import validate_input_scaling
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.posteriors.torch import TorchPosterior
from botorch.posteriors import Posterior, GPyTorchPosterior
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform, unnormalize, normalize, normalize_indices
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.get_sampler import GetSampler
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)

import torch
from torch import cdist, Tensor
from torch.distributions import Bernoulli
from torch.nn import Module
from typing import Optional, Union, Sequence, Tuple, List

from ..deep_gp.layers import DeepGPHiddenLayer, DeepMixedGPHiddenLayer

from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.models.transforms.input import InputTransform  # ★ 追加
from gpytorch.distributions import MultivariateNormal
import torch


class SimpleBernoulliPosterior(GPyTorchPosterior):
    """
    Bernoulli 分布に基づいたサンプリングを行う BoTorch 互換の単純なポスターリオル（後分布）クラス。

    BoTorch の posterior インターフェースに合わせるため、ダミーの MultivariateNormal 分布を構築しています。

    属性:
        mean (torch.Tensor): 平均テンソル（shape: [..., m]）
        variance (torch.Tensor): 分散テンソル（shape: [..., m]）
        distribution (MultivariateNormal): BoTorch の互換性のために用意されたダミー分布
    """

    def __init__(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor
    ):
        """
        クラスの初期化。

        Args:
            mean (torch.Tensor): Bernoulli 分布の平均（成功確率）、shape: [..., m]
            variance (torch.Tensor): 分散テンソル、shape: [..., m]
        """
        self._mean = mean
        self._variance = variance

        # BoTorch の posterior API に対応するため、ダミーの MultivariateNormal を構築
        batch_shape = mean.shape[:-1]
        cov = torch.diag_embed(variance)  # 分散テンソルから対角共分散行列を作成
        self.distribution = MultivariateNormal(mean, covariance_matrix=cov)

    @property
    def mean(self) -> torch.Tensor:
        """
        平均テンソルを返す。

        Returns:
            torch.Tensor: 平均テンソル（shape: [..., m]）
        """
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """
        分散テンソルを返す。

        Returns:
            torch.Tensor: 分散テンソル（shape: [..., m]）
        """
        return self._variance

    @property
    def device(self):
        """
        テンソルが配置されているデバイス（CPU/GPU）を返す。

        Returns:
            torch.device: デバイス情報
        """
        return self._mean.device

    @property
    def dtype(self):
        """
        テンソルのデータ型を返す。

        Returns:
            torch.dtype: データ型
        """
        return self._mean.dtype

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Bernoulli 分布からサンプルを生成する。

        Args:
            sample_shape (torch.Size, optional): サンプル数（デフォルトはスカラー出力）

        Returns:
            torch.Tensor: サンプルされたテンソル（shape: [*sample_shape, *batch_shape, m]）
                          各要素は 0 または 1 の値を取る。
        """
        # 指定されたサンプルサイズに mean を拡張し、Bernoulli サンプリングを実行
        return torch.bernoulli(self._mean.expand(sample_shape + self._mean.shape))

@GetSampler.register(SimpleBernoulliPosterior)
def get_sampler_for_simple_bernoulli(
    posterior: SimpleBernoulliPosterior,
    sample_shape: torch.Size,
    seed=None
):
    # 従来の MCSampler 継承を避ける：BoTorch古い環境でも動作
    class SimpleBernoulliSampler:
        def __call__(
            self,
            posterior: Posterior
        ) -> torch.Tensor:
            return posterior.rsample(sample_shape)

    return SimpleBernoulliSampler()

class GPClassificationModel(ApproximateGP):
    """
    ガウス過程を用いた分類モデル（近似推論ベース）。

    このモデルは Variational Inference を用いており、大規模データに対応可能です。
    平均関数には定数、カーネルには RBF カーネル（スケーリング付き）を使用します。

    属性:
        train_inputs (torch.Tensor): 学習データの入力（特徴量）
        train_targets (torch.Tensor): 学習データの出力（ラベル）
        mean_module (gpytorch.means.Mean): 平均関数モジュール（定数）
        covar_module (gpytorch.kernels.Kernel): 共分散関数モジュール（RBF + スケーリング）
    """

    def __init__(
        self,
        train_X,
        train_Y,
        num_inducing_points: int = 20
    ):
        """
        モデルの初期化を行う。

        Args:
            train_x (torch.Tensor): 入力データ（特徴量）、shape: [N, D]
            train_y (torch.Tensor): 出力データ（二値ラベル）、shape: [N]
            num_inducing_points (int, optional): 誘導点の最大数（デフォルト: 20）
        """
        # 誘導点は、変換済みの train_X から選択する
        num_inducing = min(train_X.shape[0], num_inducing_points)
        input_dims = train_X.shape[-1]
        batch_shape = torch.Size([])
        
        # 誘導点に対して Cholesky 分解型の変分分布を定義
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        
        if train_X.shape[0] > num_inducing:
            rand_indices = torch.randperm(train_X.shape[0], device=train_X.device)[:num_inducing]
            inducing_points = train_X[rand_indices].clone()
        else:
            inducing_points = train_X.clone()

        # 変分戦略
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
        )
        super().__init__(variational_strategy)

        # 平均関数：定数値
        self.mean_module = gpytorch.means.ConstantMean()

        # 共分散関数：RBF カーネル（スケーリング付き）
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

        # 学習用データを内部に保持（後で参照する可能性がある場合）
        self.train_inputs = train_X
        self.train_targets = train_Y

        self._train_inputs_transformed = train_X # _train_inputs_transformed という属性名にする
        self._train_targets = train_Y # 内部で使うため

    def forward(self, x):
        """
        入力 x に対するガウス過程の出力（分布）を計算する。

        Args:
            x (torch.Tensor): 入力テンソル、shape: [N, D]

        Returns:
            gpytorch.distributions.MultivariateNormal: 予測分布（平均と共分散を持つ）
        """
        # 平均と共分散の計算
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        # 多変量正規分布として返す（分類なので SoftmaxLikelihood と組み合わせる想定）
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DeepGPClassificationModel(DeepGP):
    """2値分類向けの Deep Gaussian Process モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        list_hidden_dims: Optional[List[int]] = None,
        num_inducing_points: int = 64,
    ):
        super().__init__()
        if list_hidden_dims is None:
            list_hidden_dims = [8]

        input_dim = train_X.shape[-1]
        input_dims = input_dim
        num_inducing = min(train_X.shape[0], num_inducing_points)

        self.hidden_layers = torch.nn.ModuleList()
        for hidden_dim in list_hidden_dims:
            self.hidden_layers.append(
                DeepGPHiddenLayer(
                    input_dims=input_dims,
                    output_dims=hidden_dim,
                    num_inducing=num_inducing,
                    mean_type="linear",
                )
            )
            input_dims = hidden_dim

        self.last_layer = DeepGPHiddenLayer(
            input_dims=input_dims,
            output_dims=None,
            num_inducing=num_inducing,
            mean_type="constant",
        )

        self.train_inputs = train_X
        self.train_targets = train_Y

    def forward(self, inputs, batch_mean: bool = True):
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        h = inputs
        for layer in self.hidden_layers:
            h = layer(h)

        output = self.last_layer(h)
        if not batch_mean:
            return output

        mean_x = output.mean.mean(0)
        covar_x = output.covariance_matrix.mean(0)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# class ClassifierGP(Model):
#     """
#     分類タスク用のガウス過程モデルラッパークラス。

#     BoTorch の Model インターフェースに準拠しており、後部分布（posterior）の取得も可能。
#     内部に GPClassificationModel（近似GPモデル）と BernoulliLikelihood を保持する。

#     属性:
#         train_inputs (Tensor): 学習データ（特徴量）
#         train_targets (Tensor): 学習データ（ラベル）
#         model (ApproximateGP): 内部で使用するGP分類モデル
#         likelihood (BernoulliLikelihood): ベルヌーイ尤度関数（分類用）
#     """

#     def __init__(
#         self,
#         train_X: Tensor,
#         train_Y: Tensor,
#         model: Optional[ApproximateGP] = None,
#         likelihood: Optional[BernoulliLikelihood] = None,
#     ):
#         """
#         モデルの初期化。

#         Args:
#             train_X (Tensor): 入力特徴量（shape: [N, D]）
#             train_Y (Tensor): ラベル（二値: 0 or 1）（shape: [N]）
#             model (Optional[ApproximateGP], optional): 事前に定義されたGPモデル（省略時は新たに作成）
#             likelihood (Optional[BernoulliLikelihood], optional): ベルヌーイ尤度（省略時は新たに作成）
#         """
#         super().__init__()

#         # 入力データとラベルを保存
#         self.train_inputs = train_X
#         self.train_targets = train_Y

#         # モデルと尤度関数を設定（指定がなければデフォルトで作成）
#         self.model = model if model is not None else GPClassificationModel(train_X, train_Y)
#         self.likelihood = likelihood if likelihood is not None else BernoulliLikelihood()

#         # モデルのデバイスを train_X に合わせて移動（GPU 対応）
#         self.to(train_X)

#     def forward(self, X: Tensor):
#         """
#         入力 X に対する GP モデルの予測を返す。

#         Args:
#             X (Tensor): 入力特徴量（shape: [N, D]）

#         Returns:
#             gpytorch.distributions.MultivariateNormal: GPの予測分布
#         """
#         return self.model(X)

#     def posterior(
#         self,
#         X: Tensor,
#         observation_noise: bool = False,
#         **kwargs
#     ) -> SimpleBernoulliPosterior:
#         """
#         入力 X に対する後部分布（posterior）を返す。

#         Args:
#             X (Tensor): 入力特徴量（shape: [N, D]）
#             observation_noise (bool, optional): ノイズを考慮するか（未使用）

#         Returns:
#             SimpleBernoulliPosterior: Bernoulli 分布に基づく BoTorch 互換 posterior
#         """
#         self.model.eval()
#         self.likelihood.eval()

#         # GPモデルと尤度関数による予測
#         preds = self.likelihood(self.model(X))

#         # 予測平均を成功確率とみなし、Bernoulli 分布の分散も計算
#         p = preds.mean  # shape: [..., m]
#         # var = p * (1 - p)  # Bernoulli分布の分散
#         var = preds.variance
        
#         # SimpleBernoulliPosterior を構築して返す
#         return SimpleBernoulliPosterior(mean=p, variance=var)

#     @property
#     def num_outputs(self) -> int:
#         """
#         出力の次元数を返す（分類なので常に1）。

#         Returns:
#             int: 出力次元数（1）
#         """
#         return 1

#     @property
#     def batch_shape(self) -> torch.Size:
#         """
#         モデルのバッチサイズを返す。

#         Returns:
#             torch.Size: バッチ次元のサイズ
#         """
#         return self.model.batch_shape



class DeepMixedGPClassificationModel(DeepGP):
    """カテゴリ変数を含む2値分類向け Deep Gaussian Process モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        list_hidden_dims: Optional[List[int]] = None,
        num_inducing_points: int = 64,
    ):
        super().__init__()
        if list_hidden_dims is None:
            list_hidden_dims = [8]
        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))
        num_inducing = min(train_X.shape[0], num_inducing_points)

        first_hidden_dim = list_hidden_dims[0]
        self.input_layer = DeepMixedGPHiddenLayer(
            input_dims=d,
            output_dims=first_hidden_dim,
            ord_dims=ord_dims,
            cat_dims=cat_dims,
            num_inducing=num_inducing,
            mean_type="linear",
            input_data=train_X,
        )

        self.hidden_layers = torch.nn.ModuleList()
        in_dim = first_hidden_dim
        for hidden_dim in list_hidden_dims[1:]:
            self.hidden_layers.append(
                DeepGPHiddenLayer(
                    input_dims=in_dim,
                    output_dims=hidden_dim,
                    num_inducing=num_inducing,
                    mean_type="linear",
                )
            )
            in_dim = hidden_dim

        self.last_layer = DeepGPHiddenLayer(
            input_dims=in_dim,
            output_dims=None,
            num_inducing=num_inducing,
            mean_type="constant",
        )

        self.train_inputs = train_X
        self.train_targets = train_Y

    def forward(self, inputs, batch_mean: bool = True):
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        h = self.input_layer(inputs)
        for layer in self.hidden_layers:
            h = layer(h)

        output = self.last_layer(h)
        if not batch_mean:
            return output

        mean_x = output.mean.mean(0)
        covar_x = output.covariance_matrix.mean(0)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class ClassifierGPBinaryFromMulticlass(Model):
    """
    マルチクラス分類ラベルから任意のクラス集合を「1」、その他を「0」として
    2値分類に変換し、BoTorch 互換の GP 分類モデルとして扱うためのラッパークラス。

    属性:
        train_inputs (Tensor): 学習用の入力特徴量
        train_targets (Tensor): 指定クラス vs その他 に変換されたラベル（二値）
        target_class_set (set[int]): 正例として扱うクラスの集合
        model (ApproximateGP): 近似 GP モデル
        likelihood (BernoulliLikelihood): Bernoulli 尤度関数
    """

    def __init__(
        self,
        train_X: Tensor, # Raw train_X
        train_Y: Tensor,
        target_class: Union[int, Sequence[int]],
        model: Optional[GPClassificationModel] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",
        deep_gp: bool = False,
        list_hidden_dims: Optional[List[int]] = None,
        num_inducing_points: int = 20,
    ):
        """
        モデルの初期化。指定クラス vs その他で2値分類用のデータを生成する。

        Args:
            train_X (Tensor): 入力特徴量（shape: [N, D]）
            train_Y (Tensor): マルチクラスラベル（shape: [N]）
            target_class (int or list of int): 正例とみなすクラス or クラスのリスト
            model (Optional[ApproximateGP], optional): 使用する GP モデル（省略時は内部で構築）
            likelihood (Optional[BernoulliLikelihood], optional): Bernoulli尤度（省略時は内部で構築）
        """
        super().__init__()

        self.train_inputs = (train_X,)

        # --- 正例とみなすクラスを集合で保持（1つでも複数でも対応） ---
        if isinstance(target_class, int):
            self.target_class_set = {target_class}
        else:
            self.target_class_set = set(target_class)

        # --- ラベルを 2値化 ---
        # 指定クラスなら 1.0、それ以外なら 0.0 とする
        self.train_targets = torch.tensor([
            float(y.item() in self.target_class_set) for y in train_Y
        ], device=train_Y.device)

        # --- GP モデルと Bernoulli 尤度を構築（または受け取る） ---
        self.input_transform = input_transform # この時点で input_transform は初期化済み
        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)
            # input_transform の学習はここで完結させる
            _ = self.input_transform(train_X) 
            self.input_transform.eval() # 以降は変換のみに使用

        # 生の train_X を保持
        self.train_inputs_raw = (train_X,) 

        # 変換済みの train_X を作成し、内部モデルに渡す
        transformed_train_X = self.input_transform(train_X) if self.input_transform else train_X

        # 内部GPモデルの初期化時に、変換済みのXを渡す
        if model is not None:
            self.model = model
        elif deep_gp:
            self.model = DeepGPClassificationModel(
                transformed_train_X,
                self.train_targets,
                list_hidden_dims=list_hidden_dims,
                num_inducing_points=max(num_inducing_points, 32),
            )
        else:
            self.model = GPClassificationModel(
                transformed_train_X, self.train_targets, num_inducing_points
            )
        self.likelihood = likelihood if likelihood is not None else BernoulliLikelihood()

        # GPU/CPU に合わせてモデルを転送
        self.to(train_X)

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
    
    def forward(self, X: Tensor):
        """
        入力に対する GP モデルの予測を返す。

        Args:
            X (Tensor): 入力特徴量（shape: [N, D]）

        Returns:
            gpytorch.distributions.MultivariateNormal: GP の出力分布
        """
        if isinstance(X, tuple):
            X = X[0]
        transformed_X = self.input_transform(X) if self.input_transform else X
        return self.model(transformed_X)

    def posterior(
        self,
        X: Tensor,
        observation_noise: bool = False,
        **kwargs
    ) -> SimpleBernoulliPosterior:
        """
        後部分布（posterior）を返す。

        Args:
            X (Tensor): 入力特徴量（shape: [N, D]）
            observation_noise (bool): ノイズを含めるか（未使用）

        Returns:
            SimpleBernoulliPosterior: 平均と分散に基づく BoTorch 互換 posterior（ベルヌーイ分布）
        """
        self.eval()
        self.likelihood.eval()
        
        if isinstance(X, tuple):
            X = X[0]
        
        transformed_X = self.input_transform(X) if self.input_transform else X
        latent = self.model(transformed_X)
        preds = self.likelihood(latent)
        p = preds.mean
        var = preds.variance
        return SimpleBernoulliPosterior(mean=p, variance=var)

    @property
    def num_outputs(self) -> int:
        """
        モデル出力の次元数を返す（分類なので常に1）

        Returns:
            int: 出力次元数（1）
        """
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        """
        モデルのバッチサイズを返す。

        Returns:
            torch.Size: バッチ次元のサイズ
        """
        return self.model.batch_shape

class GPClassificationMixedModel(ApproximateGP):
    """
    連続変数とカテゴリ変数が混在するデータに対応したガウス過程分類モデル（近似推論）。

    このクラスは Variational Inference を使用し、大規模データにも対応可能。
    CategoricalKernel と通常の RBF カーネルを組み合わせることで、混合データ構造を表現する。

    属性:
        mean_module (gpytorch.means.Mean): 定数平均関数
        covar_module (gpytorch.kernels.Kernel): カテゴリ・連続両対応の共分散関数
        train_inputs (Tensor): 学習入力データ
        train_targets (Tensor): 学習ラベル（二値）
        cat_dims (List[int]): カテゴリ変数の次元インデックス
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        cat_dims,
        num_inducing_points: int = 20
    ):
        """
        モデルの初期化。

        Args:
            train_X (Tensor): 入力特徴量（shape: [N, D]）
            train_Y (Tensor): ラベル（二値, shape: [N]）
            cat_dims (List[int]): カテゴリ変数の次元インデックス
            num_inducing_points (int): 誘導点の最大数（デフォルト: 20）
        """
        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        # 誘導点は、変換済みの train_X から選択する
        num_inducing = min(train_X.shape[0], num_inducing_points)
        
        # 誘導点に対して Cholesky 分解型の変分分布を定義
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        
        if train_X.shape[0] > num_inducing:
            rand_indices = torch.randperm(train_X.shape[0], device=train_X.device)[:num_inducing]
            inducing_points = train_X[rand_indices].clone()
        else:
            inducing_points = train_X.clone()

        # 変分戦略
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
        )
        super().__init__(variational_strategy)

        d = train_X.shape[-1]
        self.cat_dims = cat_dims

        # カテゴリ次元と連続次元を分離
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        batch_shape = torch.Size([])
        self.mean_module = gpytorch.means.ConstantMean()

        # カーネル構築：カテゴリ変数のみの場合
        if len(ord_dims) == 0:
            self.covar_module = ScaleKernel(
                CategoricalKernel(
                    batch_shape=batch_shape,
                    ard_num_dims=len(cat_dims),
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )
        else:
            # 連続・カテゴリ混合の場合：和カーネル + 積カーネルの合成
            cont_kernel_factory = get_covar_module_with_dim_scaled_prior

            # 和カーネル（連続 + カテゴリ）
            sum_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                + ScaleKernel(
                    CategoricalKernel(
                        batch_shape=batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-6),
                    )
                )
            )

            # 積カーネル（連続 * カテゴリ）
            prod_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                * CategoricalKernel(
                    batch_shape=batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )

            # 合成：sum + product
            self.covar_module = sum_kernel + prod_kernel

        # 学習用データを内部に保持（後で参照する可能性がある場合）
        self.train_inputs = train_X
        self.train_targets = train_Y

        self._train_inputs_transformed = train_X # _train_inputs_transformed という属性名にする
        self._train_targets = train_Y # 内部で使うため

    def forward(self, x: torch.Tensor):
        """
        入力 x に対する平均と共分散を計算し、予測分布を返す。

        Args:
            x (Tensor): 入力特徴量（shape: [N, D]）

        Returns:
            gpytorch.distributions.MultivariateNormal: 予測分布（平均と共分散）
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# class ClassifierMixedGP(Model):
#     """
#     連続変数とカテゴリ変数の混合データに対応した GP 分類モデルのラッパークラス。

#     BoTorch の Model 抽象クラスに準拠しており、獲得関数との連携が容易。
#     内部で GPClassificationMixedModel と BernoulliLikelihood を用いる。

#     属性:
#         train_inputs (Tensor): 学習用の特徴量
#         train_targets (Tensor): 学習用のラベル（二値）
#         model (ApproximateGP): カスタム GP 分類モデル（混合カーネル使用）
#         likelihood (BernoulliLikelihood): Bernoulli 尤度
#     """

#     def __init__(
#         self,
#         train_X: Tensor,
#         train_Y: Tensor,
#         cat_dims,
#         model: Optional[ApproximateGP] = None,
#         likelihood: Optional[BernoulliLikelihood] = None,
#     ):
#         """
#         モデルの初期化。

#         Args:
#             train_X (Tensor): 入力特徴量（shape: [N, D]）
#             train_Y (Tensor): 出力ラベル（二値: 0 or 1、shape: [N]）
#             cat_dims (List[int]): カテゴリ変数の次元番号
#             model (Optional[ApproximateGP], optional): 外部から渡された GP モデル（指定しない場合は内部で構築）
#             likelihood (Optional[BernoulliLikelihood], optional): 尤度（指定しない場合は BernoulliLikelihood を使用）
#         """
#         super().__init__()

#         # 入力とラベルを保持（あとで参照可能にするため）
#         self.train_inputs = train_X
#         self.train_targets = train_Y

#         # モデルと尤度関数の設定（なければ内部で構築）
#         self.model = model if model is not None else GPClassificationMixedModel(train_X, train_Y, cat_dims)
#         self.likelihood = likelihood if likelihood is not None else BernoulliLikelihood()

#         # デバイス（GPU/CPU）を train_X に合わせる
#         self.to(train_X)

#     def forward(self, X: Tensor):
#         """
#         入力 X に対する GP モデルの予測を返す。

#         Args:
#             X (Tensor): 入力特徴量（shape: [N, D]）

#         Returns:
#             gpytorch.distributions.MultivariateNormal: GP モデルによる出力分布
#         """
#         return self.model(X)

#     def posterior(
#         self,
#         X: Tensor,
#         observation_noise: bool = False,
#         **kwargs
#     ) -> SimpleBernoulliPosterior:
#         """
#         入力 X に対する後部分布（posterior）を返す。

#         Args:
#             X (Tensor): 入力特徴量
#             observation_noise (bool): 観測ノイズの有無（未使用）

#         Returns:
#             SimpleBernoulliPosterior: BoTorch 互換の Bernoulli ベース posterior
#         """
#         # 評価モードに切り替え
#         self.model.eval()
#         self.likelihood.eval()

#         # GP モデルと尤度による予測
#         preds = self.likelihood(self.model(X))
#         p = preds.mean                      # 予測確率（成功確率）
#         # var = p * (1 - p)                   # Bernoulli 分布の分散計算
#         var = preds.variance

#         # BoTorch の Posterior インターフェースに準拠するためのラップ
#         return SimpleBernoulliPosterior(mean=p, variance=var)

#     @property
#     def num_outputs(self) -> int:
#         """
#         出力の次元数（常に1）を返す。

#         Returns:
#             int: 出力数（1）
#         """
#         return 1

#     @property
#     def batch_shape(self) -> torch.Size:
#         """
#         モデルのバッチ形状を返す。

#         Returns:
#             torch.Size: バッチ形状（通常は [] または [B]）
#         """
#         return self.model.batch_shape

# class ClassifierMixedGPBinaryFromMulticlass(Model):
#     """
#     マルチクラス分類ラベルを指定クラス vs その他で 2 値化し、
#     連続・カテゴリ変数の混在に対応した GP 分類モデルとして扱うラッパークラス。

#     属性:
#         train_inputs (Tensor): 入力特徴量（連続＋カテゴリ混合）
#         train_targets (Tensor): 二値化された出力ラベル（0 または 1）
#         target_class_set (set[int]): 正例と見なすクラスの集合
#         model (ApproximateGP): GPClassificationMixedModel インスタンス
#         likelihood (BernoulliLikelihood): Bernoulli 尤度関数
#     """

#     def __init__(
#         self,
#         train_X: Tensor,
#         train_Y: Tensor,
#         target_class: Union[int, Sequence[int]],
#         cat_dims,
#         model: Optional[ApproximateGP] = None,
#         likelihood: Optional[BernoulliLikelihood] = None,
#         input_transform: Union[str, InputTransform, None] = "DEFAULT",
#         num_inducing_points: int = 20,
#     ):
#         """
#         モデルの初期化。target_class を 1、それ以外を 0 に変換する。

#         Args:
#             train_X (Tensor): 入力特徴量（shape: [N, D]）
#             train_Y (Tensor): マルチクラスラベル（shape: [N]）
#             target_class (int or list[int]): 正例（1）とみなすクラス
#             cat_dims (list[int]): カテゴリ変数の次元番号
#             model (Optional[ApproximateGP], optional): 外部から渡す GP モデル
#             likelihood (Optional[BernoulliLikelihood], optional): Bernoulli 尤度
#         """
#         super().__init__()

#         # 入力保持
#         self.train_inputs = train_X

#         # --- クラス集合を構築（単一 or 複数） ---
#         if isinstance(target_class, int):
#             self.target_class_set = {target_class}
#         else:
#             self.target_class_set = set(target_class)

#         # --- ラベルを 2値化（target_class_set に含まれるものだけ 1.0） ---
#         self.train_targets = torch.tensor(
#             [float(y.item() in self.target_class_set) for y in train_Y],
#             device=train_Y.device
#         )

#         # --- GP モデルと Bernoulli 尤度を構築（または受け取る） ---
#         self.input_transform = input_transform # この時点で input_transform は初期化済み
#         if self.input_transform is not None and hasattr(self.input_transform, "to"):
#             self.input_transform = self.input_transform.to(train_X)
#             # input_transform の学習はここで完結させる
#             _ = self.input_transform(train_X) 
#             self.input_transform.eval() # 以降は変換のみに使用

#         # 生の train_X を保持
#         self.train_inputs_raw = (train_X,) 

#         # 変換済みの train_X を作成し、内部モデルに渡す
#         transformed_train_X = self.input_transform(train_X) if self.input_transform else train_X
        
#         # --- モデルと尤度の初期化（外部提供なければ自前で構築） ---
#         self.model = model if model is not None else GPClassificationMixedModel(
#             transformed_train_X, self.train_targets, cat_dims
#         )
#         self.likelihood = likelihood if likelihood is not None else BernoulliLikelihood()

#         # GPU/CPU 対応
#         self.to(train_X)

#     # ★ BoTorchが内部で呼び出すために必要
#     def set_train_data(self, inputs=None, targets=None, strict: bool = True):
#         """
#         BoTorch の Posterior 計算時に呼び出されるメソッド。
#         """
#         if inputs is not None:
#             if torch.is_tensor(inputs):
#                 inputs = (inputs,)
#             self.train_inputs = inputs
#         if targets is not None:
#             self.train_targets = targets
    
#     def forward(self, X: Tensor):
#         """
#         入力に対する GP モデルの予測を返す。

#         Args:
#             X (Tensor): 入力特徴量（shape: [N, D]）

#         Returns:
#             gpytorch.distributions.MultivariateNormal: GP の出力分布
#         """
#         if isinstance(X, tuple):
#             X = X[0]
#         transformed_X = self.input_transform(X) if self.input_transform else X
#         return self.model(transformed_X)

#     def posterior(
#         self,
#         X: Tensor,
#         observation_noise: bool = False,
#         **kwargs
#     ) -> SimpleBernoulliPosterior:
#         """
#         後部分布（posterior）を返す。

#         Args:
#             X (Tensor): 入力特徴量（shape: [N, D]）
#             observation_noise (bool): ノイズを含めるか（未使用）

#         Returns:
#             SimpleBernoulliPosterior: 平均と分散に基づく BoTorch 互換 posterior（ベルヌーイ分布）
#         """
#         self.eval()
#         self.likelihood.eval()
        
#         if isinstance(X, tuple):
#             X = X[0]
        
#         transformed_X = self.input_transform(X) if self.input_transform else X
#         latent = self.model(transformed_X)
#         preds = self.likelihood(latent)
#         p = preds.mean
#         var = preds.variance
#         return SimpleBernoulliPosterior(mean=p, variance=var)

#     @property
#     def num_outputs(self) -> int:
#         """
#         モデルの出力次元（分類なので常に 1）

#         Returns:
#             int: 出力次元（1）
#         """
#         return 1

#     @property
#     def batch_shape(self) -> torch.Size:
#         """
#         モデルのバッチ形状を返す。

#         Returns:
#             torch.Size: バッチ形状
#         """
#         return self.model.batch_shape

class ClassifierMixedGPBinaryFromMulticlass(Model):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        target_class: Union[int, Sequence[int]],
        cat_dims,
        model: Optional[ApproximateGP] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform=None,  # Union[str, InputTransform, None] にしていてもOK
        deep_gp: bool = False,
        list_hidden_dims: Optional[List[int]] = None,
        num_inducing_points: int = 20,
    ):
        super().__init__()

        # --- train_inputs は BoTorch 的に (Tensor,) の tuple が必須 ---
        self.train_inputs_raw = (train_X,)          # raw を保持したいなら
        self.train_inputs = (train_X,)              # ★これが必須（Tensor単体はNG）

        # --- クラス集合 ---
        if isinstance(target_class, int):
            self.target_class_set = {target_class}
        else:
            self.target_class_set = set(target_class)

        # --- 2値化 targets（dtype/device を揃える） ---
        self.train_targets = (train_Y.detach().clone())
        # train_Y が (N,) の想定：target class set に入っていれば 1.0
        self.train_targets = torch.tensor(
            [float(y.item() in self.target_class_set) for y in train_Y],
            device=train_Y.device,
            dtype=train_X.dtype,  # float
        )

        # --- input_transform の設定 ---
        # ここはあなたの既存方針に合わせる。
        # 重要：BoTorch の Model.eval() が input_transform を触るので、None or InputTransform を入れる。
        if isinstance(input_transform, str):
            # 安全側：文字列が来たら transform 無しとして扱う（必要なら外で具体オブジェクトを渡す）
            self.input_transform = None
        else:
            self.input_transform = input_transform

        # input_transform を「学習」させたい場合は train モードで一度通して eval にする
        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)
            _ = self.input_transform(train_X)  # 学習（Normalize 等の統計確定）
            self.input_transform.eval()

        # 内部モデルに渡す train_X は transform 後
        transformed_train_X = self.input_transform(train_X) if self.input_transform else train_X

        if model is not None:
            self.model = model
        elif deep_gp:
            self.model = DeepMixedGPClassificationModel(
                transformed_train_X,
                self.train_targets,
                cat_dims=cat_dims,
                list_hidden_dims=list_hidden_dims,
                num_inducing_points=max(num_inducing_points, 32),
            )
        else:
            self.model = GPClassificationMixedModel(
                transformed_train_X, self.train_targets, cat_dims, num_inducing_points
            )
        self.likelihood = likelihood if likelihood is not None else BernoulliLikelihood()

        self.to(device=train_X.device, dtype=train_X.dtype)

    # BoTorch が内部で呼ぶ
    def set_train_data(self, inputs=None, targets=None, strict: bool = True):
        if inputs is not None:
            # ★常に tuple で持つ
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
            self.train_inputs_raw = inputs  # raw を追従させたいなら
        if targets is not None:
            self.train_targets = targets

    # def forward(self, X: Tensor):
    #     # forward は gpytorch の分布を返す用途。
    #     # ここも tuple/shape に頑健にしておく。
    #     if isinstance(X, tuple):
    #         X = X[0]

    #     transformed_X = self.input_transform(X) if self.input_transform else X

    #     # Mixed GP が 3D を受けない場合に備えて flatten（安全策）
    #     orig = transformed_X.shape[:-1]  # (...,)
    #     Xf = transformed_X.reshape(-1, transformed_X.shape[-1])  # (-1, d)
    #     out = self.model(Xf)
    #     return out  # 分布の shape は内部モデル側に依存

    # def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs):
    #     self.eval()
    #     self.likelihood.eval()

    #     if isinstance(X, tuple):
    #         X = X[0]

    #     transformed_X = self.input_transform(X) if self.input_transform else X

    #     # ★ここが重要：q>1 / batch でも壊れないよう flatten → reshape
    #     orig = transformed_X.shape[:-1]                 # batch_shape x q など
    #     Xf = transformed_X.reshape(-1, transformed_X.shape[-1])  # (-1, d)

    #     latent = self.model(Xf)
    #     preds = self.likelihood(latent)                # Bernoulli 的な分布を想定

    #     p = preds.mean.reshape(*orig).unsqueeze(-1)     # (..., 1)
    #     var = preds.variance.reshape(*orig).unsqueeze(-1)

    #     return SimpleBernoulliPosterior(mean=p, variance=var)
    def forward(self, X: Tensor):
        """
        入力に対する GP モデルの予測を返す。

        Args:
            X (Tensor): 入力特徴量（shape: [N, D]）

        Returns:
            gpytorch.distributions.MultivariateNormal: GP の出力分布
        """
        if isinstance(X, tuple):
            X = X[0]
        transformed_X = self.input_transform(X) if self.input_transform else X
        return self.model(transformed_X)

    def posterior(
        self,
        X: Tensor,
        observation_noise: bool = False,
        **kwargs
    ) -> SimpleBernoulliPosterior:
        """
        後部分布（posterior）を返す。

        Args:
            X (Tensor): 入力特徴量（shape: [N, D]）
            observation_noise (bool): ノイズを含めるか（未使用）

        Returns:
            SimpleBernoulliPosterior: 平均と分散に基づく BoTorch 互換 posterior（ベルヌーイ分布）
        """
        self.eval()
        self.likelihood.eval()
        
        if isinstance(X, tuple):
            X = X[0]
        
        transformed_X = self.input_transform(X) if self.input_transform else X
        latent = self.model(transformed_X)
        preds = self.likelihood(latent)
        p = preds.mean
        var = preds.variance
        return SimpleBernoulliPosterior(mean=p, variance=var)

    @property
    def num_outputs(self) -> int:
        return 1

    # @property
    # def batch_shape(self) -> torch.Size:
    #     """
    #     モデルのバッチサイズを返す。

    #     Returns:
    #         torch.Size: バッチ次元のサイズ
    #     """
    #     return self.model.batch_shape
    @property
    def batch_shape(self) -> torch.Size:
        # ★この分類ラッパーはバッチモデルではないので常に空
        return torch.Size([])

# class GPClassificationModelMC(ApproximateGP):
#     """
#     クラスごとに独立な GP を持つ多クラス分類用の変分 GP モデル。

#     Args:
#         train_x (Tensor): 入力特徴量（shape: [N, D]）
#         train_y (Tensor): 出力ラベル（整数, shape: [N]）
#         num_classes (int): クラス数
#     """

#     def __init__(
#         self,
#         train_X,
#         train_Y,
#         num_classes
#     ):
#         num_inducing = min(train_X.shape[0], 20)

#         # クラスごとに独立なバッチを定義
#         variational_distribution = CholeskyVariationalDistribution(
#             num_inducing_points=num_inducing,
#             batch_shape=torch.Size([num_classes])
#         )

#         inducing_points = train_X[:num_inducing].clone()

#         variational_strategy = VariationalStrategy(
#             self,
#             inducing_points=inducing_points,
#             variational_distribution=variational_distribution,
#             learn_inducing_locations=True,
#         )

#         super().__init__(variational_strategy)

#         # 各クラスごとに平均・共分散モジュールをバッチで構成
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_classes])),
#             batch_shape=torch.Size([num_classes]),
#         )

#         self.train_inputs = train_X
#         self.train_targets = train_Y

#     def forward(self, x):
#         """
#         入力に対する予測分布（各クラス独立 GP）を返す。

#         Args:
#             x (Tensor): 入力特徴量（shape: [N, D] または [B, q, D]）

#         Returns:
#             MultivariateNormal: 出力分布（クラスごとの GP 予測）
#         """
#         if x.dim() == 3:
#             B, q, D = x.shape
#             x_flat = x.view(B * q, D)
#             mean = self.mean_module(x_flat)
#             covar_lazy = self.covar_module(x_flat)
#             return gpytorch.distributions.MultivariateNormal(mean, covar_lazy)
#         else:
#             mean = self.mean_module(x)
#             covar = self.covar_module(x)
#             return gpytorch.distributions.MultivariateNormal(mean, covar)


# class ClassifierGP_MC(Model):
#     """
#     BoTorch 用の多クラス GP 分類ラッパー。

#     Args:
#         train_X (Tensor): 入力特徴量
#         train_Y (Tensor): ラベル（0〜num_classes-1）
#         num_classes (int): クラス数
#         model (Optional[ApproximateGP]): 外部から渡すモデル
#         likelihood (Optional[SoftmaxLikelihood]): ソフトマックス尤度
#     """

#     def __init__(
#         self,
#         train_X: Tensor,
#         train_Y: Tensor,
#         num_classes: int,
#         model: Optional[ApproximateGP] = None,
#         likelihood: Optional[SoftmaxLikelihood] = None,
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.train_inputs = train_X
#         self.train_targets = train_Y.long()

#         self.model = model if model is not None else GPClassificationModelMC(
#             train_X, train_Y.long(), num_classes
#         )
#         self.likelihood = likelihood if likelihood is not None else SoftmaxLikelihood(
#             num_classes=num_classes,
#             mixing_weights=False,
#         )
#         self.to(train_X)

#     def forward(self, X: Tensor):
#         """GP モデルによる前向き推論"""
#         print("X", X.shape)
#         return self.model(X)

#     def posterior(
#         self,
#         X: Tensor,
#         observation_noise: bool = False,
#         **kwargs
#     ) -> Posterior:
#         """
#         入力に対する BoTorch Posterior（多クラス確率）を返す。

#         Args:
#             X (Tensor): 入力特徴量
#             observation_noise (bool): ノイズ考慮（未使用）

#         Returns:
#             SimpleMulticlassPosterior: クラス確率と分散を含む Posterior
#         """
#         self.model.eval()
#         self.likelihood.eval()

#         output = self.model(X)
#         preds = self.likelihood(output)

#         probs = preds.probs                # shape: (N, num_classes)
#         # var = probs * (1 - probs)          # 各クラスごとの分散
#         var = preds.variance

#         return SimpleMulticlassPosterior(probs=probs, variance=var)

#     @property
#     def num_outputs(self) -> int:
#         """クラス数（出力次元）"""
#         return self.num_classes

def fit_classifier_mll(mll):
    """
    ガウス過程分類モデル（近似推論）の学習ループを実行する関数。

    与えられた `mll`（変分 ELBOなどの損失）を用いて、
    モデルと尤度のパラメータを最適化する。

    Args:
        mll: gpytorch の MarginalLogLikelihood インスタンス（例：VariationalELBO）
             `mll.model.train_inputs` と `mll.model.train_targets` を内部で使用する。

    Note:
        - 学習回数は 300 エポックに固定されています。
        - 学習率は固定（lr=0.01）で Adam オプティマイザを使用します。
        - DeepGP の場合は ELBO 用に `batch_mean=False` の出力を使用します。
    """
    # モデルと尤度を学習モードに設定
    mll.model.train()
    mll.likelihood.train()

    # Adam オプティマイザを初期化（学習率: 0.01）
    optimizer = torch.optim.Adam(mll.model.parameters(), lr=0.01)

    y_tensor = mll.model.train_targets
    x_tensor = mll.model.train_inputs[0] if isinstance(mll.model.train_inputs, tuple) else mll.model.train_inputs

    # 300 エポック学習
    for i in range(300):
        optimizer.zero_grad()  # 勾配を初期化

        # DeepGP は ELBO にサンプル次元を保った出力を渡す
        if isinstance(mll.model, DeepGP):
            output = mll.model.forward(x_tensor, batch_mean=False)
        else:
            output = mll.model(x_tensor)

        # 変分 ELBO を最大化（損失関数はマイナスを取る）
        if (y_tensor.ndim > 1) and (y_tensor.shape[-1] == 1):
            loss = -mll(output, y_tensor.view(-1))
        else:
            loss = -mll(output, y_tensor)

        # 勾配計算と更新
        loss.backward()
        optimizer.step()
