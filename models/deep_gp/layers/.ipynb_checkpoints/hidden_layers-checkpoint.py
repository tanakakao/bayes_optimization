import numpy as np
import torch
from torch import Tensor
import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import GreaterThan
from botorch.utils.transforms import normalize_indices
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
# from botorch.utils.transforms import Normalize
from gpytorch.utils.grid import ScaleToBounds
from ..utils import LargeFeatureExtractor  # `your_module` は適切なモジュールに置き換えてください


class DeepGPHiddenLayer(DeepGPLayer):
    """
    Deep Gaussian Processの隠れ層を表現するクラス。

    Args:
        input_dims (int): 入力次元数。
        output_dims (Optional[int]): 出力次元数。スカラ出力の場合はNone。
        num_inducing (int): 誘導点の数。デフォルトは128。
        mean_type (str): 平均関数の種類（'constant'または'linear'）。デフォルトは'constant'。
    """
    def __init__(
        self,
        input_dims,
        output_dims=None,
        num_inducing=128,
        mean_type='constant',
        inducing_points=None  # ★ 追加: 外部から初期値を受け取る
    ):
        """
        初期化メソッド。

        誘導点、変分分布、変分戦略、平均関数、カーネルを設定します。

        Args:
            input_dims (int): 入力次元数。
            output_dims (Optional[int]): 出力次元数。
            num_inducing (int): 誘導点の数。
            mean_type (str): 平均関数の種類。
        """
        # ★ 修正: 外部から誘導点が渡された場合はそれを使用、なければ randn で初期化
        if inducing_points is None:
            # デフォルト: 標準正規分布 N(0, 1)
            # Normalize([0,1])を使う場合は範囲がズレるため、外部から渡すのが推奨
            inducing_init = torch.randn(
                num_inducing, input_dims
            ) if output_dims is None else torch.randn(
                output_dims, num_inducing, input_dims
            )
        else:
            # 渡されたデータを使用（入力データのサブセットなど）
            inducing_init = inducing_points
        
        # バッチ形状の設定
        batch_shape = torch.Size([] if output_dims is None else [output_dims])

        # 変分分布と変分戦略の設定
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self, inducing_init, variational_distribution, learn_inducing_locations=True
        )

        # 親クラスの初期化
        super().__init__(variational_strategy, input_dims, output_dims)

        # 平均関数の設定
        self.mean_module = ConstantMean() if mean_type == 'constant' else LinearMean(input_dims)
        
        # RBFカーネル（Maternカーネル）の設定
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

    def forward(self, x):
        """
        フォワードパスで多変量正規分布を計算する。

        Args:
            x (Tensor): 入力テンソル。

        Returns:
            MultivariateNormal: 平均と共分散を持つ多変量正規分布。
        """
        mean_x = self.mean_module(x)  # 平均の計算
        covar_x = self.covar_module(x)  # 共分散の計算
        return MultivariateNormal(mean_x, covar_x)

class DeepMixedGPHiddenLayer(DeepGPLayer):
    """
    混合型データ（カテゴリカルデータと連続データ）に対応したDeep Gaussian Processの隠れ層クラス。

    Args:
        input_dims (int): 入力次元数。
        output_dims (Optional[int]): 出力次元数。スカラー出力の場合はNone。
        ord_dims (Sequence[int]): 連続データの次元インデックス。
        cat_dims (Sequence[int]): カテゴリカルデータの次元インデックス。
        aug_batch_shape (torch.Size): カーネルのバッチ形状。
        num_inducing (int): 誘導点の数。デフォルトは128。
        mean_type (str): 平均関数の種類（'constant'または'linear'）。デフォルトは'constant'。
    """
    def __init__(
        self,
        input_dims,
        output_dims,
        ord_dims,
        cat_dims,
        num_inducing=128,
        mean_type='constant',
        input_data=None,
        inducing_points=None  # ★ 追加: 外部から初期値を受け取る
    ):
        """
        初期化メソッド。

        隠れ層の誘導点、変分分布、カーネル関数を設定します。

        Args:
            input_dims (int): 入力の次元数。
            output_dims (Optional[int]): 出力の次元数。
            ord_dims (Sequence[int]): 連続データの次元インデックス。
            cat_dims (Sequence[int]): カテゴリカルデータの次元インデックス。
            num_inducing (int): 誘導点の数。
            mean_type (str): 平均関数の種類。
        """        
        # ★ 修正: 外部から誘導点が渡された場合はそれを使用、なければ randn で初期化
        if inducing_points is None:
            # デフォルト: 標準正規分布 N(0, 1)
            # Normalize([0,1])を使う場合は範囲がズレるため、外部から渡すのが推奨
            inducing_init = torch.randn(
                num_inducing, input_dims
            ) if output_dims is None else torch.randn(
                output_dims, num_inducing, input_dims
            )
        else:
            # 渡されたデータを使用（入力データのサブセットなど）
            inducing_init = inducing_points
        
        for i in range(len(cat_dims)):
            if output_dims is None:
                inducing_init[:,cat_dims_ind[i]] = torch.tensor(
                    np.random.choice(np.unique(input_data[:,cat_dims[i]]), (num_inducing))
                )
            else:
                inducing_init[:,:,cat_dims[i]] = torch.tensor(
                    np.random.choice(np.unique(input_data[:,cat_dims[i]]), (output_dims, num_inducing))
                )
        batch_shape = torch.Size([] if output_dims is None else [output_dims])

        # 変分分布の初期化
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )

        # 変分戦略の設定
        variational_strategy = VariationalStrategy(
            self,
            inducing_init,
            variational_distribution,
            learn_inducing_locations=True,
        )

        # 親クラスの初期化
        super().__init__(variational_strategy, input_dims, output_dims)
        
        # 平均関数の選択
        self.mean_module = (
            ConstantMean() if mean_type == 'constant' else LinearMean(input_dims)
        )

        # カーネル関数の構築（混合データに対応）
        if len(ord_dims) == 0:
            # 連続データがなく、カテゴリカルデータのみの場合
            self.covar_module = ScaleKernel(
                CategoricalKernel(
                    batch_shape=batch_shape,
                    ard_num_dims=len(cat_dims),
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )
        else:
            # 連続データとカテゴリカルデータの両方が存在する場合
            cont_kernel_factory = get_covar_module_with_dim_scaled_prior

            # 和カーネル（連続データ + カテゴリカルデータ）
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

            # 積カーネル（連続データ * カテゴリカルデータ）
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

            # 和カーネルと積カーネルを合成
            self.covar_module = sum_kernel + prod_kernel

    def forward(self, x):
        """
        入力データに対して平均関数と共分散関数を適用し、多変量正規分布を計算します。

        Args:
            x (Tensor): 入力テンソル。

        Returns:
            MultivariateNormal: 平均と共分散を持つ多変量正規分布。
        """
        mean_x = self.mean_module(x)  # 平均の計算
        covar_x = self.covar_module(x)  # 共分散の計算
        return MultivariateNormal(mean_x, covar_x)

class DeepKernelDeepGPHiddenLayer(DeepGPHiddenLayer):
    """
    Deep Gaussian Processの隠れ層を表現するクラス。

    Args:
        input_dims (int): 入力次元数。
        output_dims (Optional[int]): 出力次元数。スカラ出力の場合はNone。
        num_inducing (int): 誘導点の数。デフォルトは128。
        mean_type (str): 平均関数の種類（'constant'または'linear'）。デフォルトは'constant'。
    """
    def __init__(
        self,
        input_dims,
        output_dims=None,
        num_inducing=128,
        mean_type='constant',
        inducing_points=None  # ★ 追加: 外部から初期値を受け取る
    ):
        super().__init__(input_dims, output_dims, num_inducing, mean_type, inducing_points)

        self.feature_extractor = LargeFeatureExtractor(
                input_dim=input_dims,
                output_dim=input_dims
        )
        self.scale_to_bounds = ScaleToBounds(-1., 1.)

    def forward(self, x):
        """
        フォワードパスで多変量正規分布を計算する。

        Args:
            x (Tensor): 入力テンソル。

        Returns:
            MultivariateNormal: 平均と共分散を持つ多変量正規分布。
        """
         # データを特徴抽出器で変換
        projected_x = self.feature_extractor(x)

        # スケーリングを適用
        projected_x = self.scale_to_bounds(projected_x)
        
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

class DeepKernelDeepMixedGPHiddenLayer(DeepMixedGPHiddenLayer):
    """
    混合型データ（カテゴリカルデータと連続データ）に対応したDeep Gaussian Processの隠れ層クラス。

    Args:
        input_dims (int): 入力次元数。
        output_dims (Optional[int]): 出力次元数。スカラー出力の場合はNone。
        ord_dims (Sequence[int]): 連続データの次元インデックス。
        cat_dims (Sequence[int]): カテゴリカルデータの次元インデックス。
        aug_batch_shape (torch.Size): カーネルのバッチ形状。
        num_inducing (int): 誘導点の数。デフォルトは128。
        mean_type (str): 平均関数の種類（'constant'または'linear'）。デフォルトは'constant'。
    """
    def __init__(
        self,
        input_dims,
        output_dims,
        ord_dims,
        cat_dims,
        num_inducing=128,
        mean_type='constant',
        input_data=None,
        inducing_points=None  # ★ 追加: 外部から初期値を受け取る
    ):
        super().__init__(input_dims,output_dims,ord_dims,cat_dims,num_inducing,mean_type,input_data,inducing_points)

        self.feature_extractor = LargeFeatureExtractor(
            input_dim=len(ord_dims),
            output_dim=len(ord_dims)
        )
        self.scale_to_bounds = ScaleToBounds(-1., 1.)

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        
    def forward(self, x):
        """
        入力データに対して平均関数と共分散関数を適用し、多変量正規分布を計算します。

        Args:
            x (Tensor): 入力テンソル。

        Returns:
            MultivariateNormal: 平均と共分散を持つ多変量正規分布。
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
    
        mean_x = self.mean_module(restored_tensor)
        covar_x = self.covar_module(restored_tensor)
        return MultivariateNormal(mean_x, covar_x)

# =========================================================================================
# Advance
# =========================================================================================

class SkipDeepGPHiddenLayer(DeepGPHiddenLayer):
    """
    Input Reinjection（スキップ接続）に対応したDeep GP層。
    forward時に `original_input` を受け取り、現在の入力と結合します。
    """
    def forward(self, x, original_input=None):
        # x: 前の層からの出力 (Batch x N x Hidden_Dim)
        # original_input: 元の入力データ (Batch x N x Input_Dim)
        
        if original_input is not None:
            # サンプル次元(num_samples)などの整合性を取るため、original_inputを拡張
            if original_input.dim() < x.dim():
                original_input = original_input.expand(*x.shape[:-1], original_input.shape[-1])
            
            # 特徴量次元(dim=-1)で結合
            combined_x = torch.cat([x, original_input], dim=-1)
            mean_x = self.mean_module(combined_x)
            covar_x = self.covar_module(combined_x)
        else:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            
        return MultivariateNormal(mean_x, covar_x)

class SkipDeepMixedGPHiddenLayer(DeepMixedGPHiddenLayer):
    """
    Input Reinjection（スキップ接続）に対応した混合データ用 Deep GP層。
    
    forward時に `original_input` (カテゴリカル変数を含む) を受け取り、
    現在の層への入力 (連続値) と結合して処理します。
    """
    def forward(self, x, original_input=None):
        # x: 前の層からの出力 (Batch x N x Hidden_Dim) [すべて連続値]
        # original_input: 元の入力データ (Batch x N x Input_Dim) [連続 + カテゴリ]
        
        if original_input is not None:
            # 1. 形状の整合性を取る (サンプル次元などの拡張)
            if original_input.dim() < x.dim():
                # xが (num_samples x Batch x N x D) で originalが (Batch x N x D) の場合などを想定
                original_input = original_input.expand(*x.shape[:-1], original_input.shape[-1])
            
            # 2. 結合 (Concatenate)
            # 結合後の形状: (..., Hidden_Dim + Input_Dim)
            combined_x = torch.cat([x, original_input], dim=-1)
            
            # 3. カーネルと平均関数の適用
            # 注意: __init__ で設定された cat_dims は、この combined_x のインデックスを指している必要がある
            mean_x = self.mean_module(combined_x)
            covar_x = self.covar_module(combined_x)
        else:
            # original_inputがない場合は通常の処理 (ただし、次元数が合わないとエラーになる可能性あり)
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            
        return MultivariateNormal(mean_x, covar_x)

class SkipDeepKernelDeepGPHiddenLayer(DeepKernelDeepGPHiddenLayer):
    """
    DKL (Deep Kernel Learning) と Input Reinjection (Skip接続) を組み合わせた層。
    連続値データ用。
    """
    def forward(self, x, original_input=None):
        # 1. 結合処理 (Skip Connection)
        if original_input is not None:
            # 次元の整合性を取る (サンプル次元などの拡張)
            if original_input.dim() < x.dim():
                original_input = original_input.expand(*x.shape[:-1], original_input.shape[-1])
            
            # 特徴抽出の前に結合する
            # これにより、NNは「前層の特徴」と「元の入力」の両方を見て変換を学習できる
            x = torch.cat([x, original_input], dim=-1)

        # 2. 特徴抽出 (Deep Kernel Learning)
        # 親クラスで定義された feature_extractor は、結合後の次元数で初期化されている必要がある
        projected_x = self.feature_extractor(x)

        # 3. スケーリング
        projected_x = self.scale_to_bounds(projected_x)
        
        # 4. GPの平均・分散計算
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        
        return MultivariateNormal(mean_x, covar_x)

class SkipDeepKernelDeepMixedGPHiddenLayer(DeepKernelDeepMixedGPHiddenLayer):
    """
    DKL (Deep Kernel Learning) と Input Reinjection (Skip接続) を組み合わせた層。
    混合データ（カテゴリ＋連続）用。
    """
    def forward(self, x, original_input=None):
        # 1. 結合処理 (Skip Connection)
        if original_input is not None:
            if original_input.dim() < x.dim():
                original_input = original_input.expand(*x.shape[:-1], original_input.shape[-1])
            x = torch.cat([x, original_input], dim=-1)

        # --- ここからは結合後の x に対して処理を行う ---

        # 2. データの分離
        # x の中からカテゴリ変数の列を抽出
        x_cat = x[..., self.cat_dims]
        
        # x の中から連続変数の列を抽出
        # (self.ord_dims が __init__ で保存されている前提で使用します。
        #  保存されていない場合はリスト内包表記で動的に計算してください)
        x_ord = x[..., self.ord_dims]

        # 3. 連続変数の特徴抽出とスケーリング
        projected_x_ord = self.feature_extractor(x_ord)
        projected_x_ord = self.scale_to_bounds(projected_x_ord)

        # 4. データの再構築 (Reconstruction)
        # 特徴変換された連続変数と、元のカテゴリ変数を、正しい次元順序に戻す
        
        # 結果を格納する空のテンソルを作成 (形状は x と同じ)
        x_reconstructed = torch.empty_like(x)
        
        # インデックスを指定して値を埋め込む
        # これにより、cat_dims が末尾に追加された場合や飛び飛びの場合でも正しく配置されます
        x_reconstructed[..., self.ord_dims] = projected_x_ord
        x_reconstructed[..., self.cat_dims] = x_cat
    
        # 5. GPの平均・分散計算
        mean_x = self.mean_module(x_reconstructed)
        covar_x = self.covar_module(x_reconstructed)
        
        return MultivariateNormal(mean_x, covar_x)