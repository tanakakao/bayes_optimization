import torch
from torch import Tensor
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.utils.transforms import normalize_indices
# from botorch.utils.transforms import Normalize
from gpytorch.utils.grid import ScaleToBounds
from ..utils import LargeFeatureExtractor  # `your_module` は適切なモジュールに置き換えてください


class DeepKernel(ExactGP):
    """
    DeepKernelModelは、深層学習を組み合わせたガウス過程モデルです。
    特徴抽出器（Deep Feature Extractor）を使用して入力データを変換し、その後ガウス過程回帰に基づいて予測を行います。

    Attributes:
        mean_module (gpytorch.modules.mean.Mean): 平均モジュール
        covar_module (gpytorch.kernels.Kernel): 共分散モジュール
        feature_extractor (LargeFeatureExtractor): 特徴抽出器
        scale_to_bounds (ScaleToBounds): 入力データを[-1, 1]の範囲にスケーリング
    """

    def __init__(self, train_x, train_y, likelihood):
        """
        DeepKernelModelのコンストラクタ。

        Args:
            train_x (Tensor): 学習データの特徴量
            train_y (Tensor): 学習データのターゲット値
            likelihood (gpytorch.likelihoods.Likelihood): 尤度関数
        """
        super(DeepKernel, self).__init__(train_x, train_y, likelihood)
        
        # 入力データの次元数を取得
        input_dim = train_x.size(-1)
        num_outputs = train_y.shape[-1] if (len(train_y.shape)>1)&(train_y.shape[-1]!=1) else None
        self.num_outputs = train_y.shape[-1] if (len(train_y.shape)>1)&(train_y.shape[-1]!=1) else 1
        batch_shape = torch.Size([] if num_outputs is None else [num_outputs])

        # 定数平均関数を設定
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        if num_outputs is None:
            self.mean_module = ConstantMean(batch_shape=batch_shape)
            self.covar_module = ScaleKernel(
                    RBFKernel(
                        batch_shape=batch_shape,
                         ard_num_dims=train_x.size(1)
                    ),
                    batch_shape=batch_shape)
        else:
            self.mean_module = MultitaskMean(
                ConstantMean(), num_tasks=train_y.shape[-1]
            )
            self.covar_module = MultitaskKernel(
                        ScaleKernel(RBFKernel(ard_num_dims=train_x.size(1))),
                num_tasks=train_y.shape[-1])

        # 特徴抽出器を初期化 (データの次元数を使用)
        self.feature_extractor = LargeFeatureExtractor(
            input_dim=input_dim,
            output_dim=input_dim
        )

        # 入力データを[-1, 1]の範囲にスケーリングするためのモジュール
        self.scale_to_bounds = ScaleToBounds(-1., 1.)

    def forward(self, inputs):
        """
        モデルの順伝播を定義します。入力データを特徴抽出器を通して処理し、ガウス過程による予測を行います。

        Args:
            x (Tensor): 入力データ

        Returns:
            MultivariateNormal: ガウス過程の予測結果（平均と共分散）
        """
        # データを特徴抽出器で変換
        projected_x = self.feature_extractor(inputs)

        # スケーリングを適用
        projected_x = self.scale_to_bounds(projected_x)

        # 変換されたデータに基づいて平均と共分散を計算
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        # 平均と共分散からガウス過程の分布を作成
        if self.num_outputs == 1:
            return MultivariateNormal(mean_x, covar_x)
        else:
            return MultitaskMultivariateNormal(mean_x, covar_x)

class DeepKernelMixed(BatchedMultiOutputGPyTorchModel, ExactGP):
    """
    DeepKernelModelは、深層学習を組み合わせたガウス過程モデルです。
    特徴抽出器（Deep Feature Extractor）を使用して入力データを変換し、その後ガウス過程回帰に基づいて予測を行います。

    Attributes:
        mean_module (gpytorch.modules.mean.Mean): 平均モジュール
        covar_module (gpytorch.kernels.Kernel): 共分散モジュール
        feature_extractor (LargeFeatureExtractor): 特徴抽出器
        scale_to_bounds (ScaleToBounds): 入力データを[-1, 1]の範囲にスケーリング
    """

    def __init__(self, train_x, train_y, cat_dims, likelihood):
        """
        DeepKernelModelのコンストラクタ。

        Args:
            train_x (Tensor): 学習データの特徴量
            train_y (Tensor): 学習データのターゲット値
            likelihood (gpytorch.likelihoods.Likelihood): 尤度関数
        """
        super(DeepKernelMixed, self).__init__(train_x, train_y, likelihood)

        # 入力データの次元数を取得
        data_dim = train_x.size(-1)
        self._num_outputs = train_y.shape[-1] if (len(train_y.shape)>1)&(train_y.shape[-1]!=1) else 1

        # カテゴリ次元が指定されていない場合にエラーをスロー
        if len(cat_dims) == 0:
            raise ValueError(
                "カテゴリ次元を指定する必要があります (cat_dims)。"
            )

        # 次元の正規化と分割
        self._ignore_X_dims_scaling_check = cat_dims
        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_x, train_Y=train_y)
        # aug_batch_shape = torch.Size([1])
        aug_batch_shape = train_x.shape[:-2]

        d = train_x.shape[-1]
        self.cat_dims = cat_dims
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.feature_extractor = LargeFeatureExtractor(
            input_dim=len(ord_dims),
            output_dim=len(ord_dims)
        )

        # 平均関数の選択
        if self._num_outputs==1:
            self.mean_module = ConstantMean(batch_shape=aug_batch_shape)
        else:
            self.mean_module = MultitaskMean(
                ConstantMean(), num_tasks=train_y.shape[-1]
            )

        # カーネル関数の構築（混合データに対応）
        if len(ord_dims) == 0:
            # 連続データがなく、カテゴリカルデータのみの場合
            self.covar_module = ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )
        else:
            # 連続データとカテゴリカルデータの両方が存在する場合
            cont_kernel_factory = get_covar_module_with_dim_scaled_prior

            # 和カーネル
            sum_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                + ScaleKernel(
                    CategoricalKernel(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-6),
                    )
                )
            )

            # 積カーネル
            prod_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                * CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )

        # 和カーネルと積カーネルを合成
        if self._num_outputs==1:
            self.covar_module = sum_kernel + prod_kernel
        else:
            self.covar_module = MultitaskKernel(
                sum_kernel + prod_kernel,
                num_tasks=train_y.shape[-1]
            )

        # 入力データを[-1, 1]の範囲にスケーリングするためのモジュール
        self.scale_to_bounds = ScaleToBounds(-1., 1.)

    def forward(self, x):
        """
        モデルの順伝播を定義します。入力データを特徴抽出器を通して処理し、ガウス過程による予測を行います。

        Args:
            x (Tensor): 入力データ

        Returns:
            MultivariateNormal: ガウス過程の予測結果（平均と共分散）
        """
        # 抜き出した列を取得
        extracted_columns = x[..., self.cat_dims]
        
        # 残りの列を取得
        remaining_columns = x[..., [i for i in range(x.size(-1)) if i not in self.cat_dims]]

        # データを特徴抽出器で変換
        projected_x = self.feature_extractor(remaining_columns)

        # スケーリングを適用
        projected_x = self.scale_to_bounds(projected_x)

        # 抜き出した列を戻す処理
        restored_tensor = remaining_columns.clone()  # 残りのテンソルをコピーして初期化
        for idx, col_idx in enumerate(self.cat_dims):
            restored_tensor = torch.cat((projected_x[..., :col_idx],
                                         extracted_columns[..., idx:idx+1],
                                         projected_x[..., col_idx:]), dim=len(x.shape)-1)
        
        # 変換されたデータに基づいて平均と共分散を計算
        mean_x = self.mean_module(restored_tensor)
        covar_x = self.covar_module(restored_tensor)
        # 平均と共分散からガウス過程の分布を作成
        # return  gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        if self._num_outputs==1:
            return MultivariateNormal(mean_x, covar_x)
        else:
            return MultitaskMultivariateNormal(mean_x, covar_x)