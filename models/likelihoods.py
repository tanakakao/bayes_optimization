import torch
from torch import Tensor
from typing import Tuple, Optional

from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.priors.torch_priors import LogNormalPrior
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL


def get_batch_dimensions(
    train_X: Tensor,
    train_Y: Tensor,
) -> Tuple[torch.Size, torch.Size]:
    """
    入力データのバッチ形状と、出力次元を含む拡張バッチ形状を取得。

    Args:
        train_X: 訓練データの特徴量 (n x d または batch_shape x n x d)。
        train_Y: 訓練データの観測値 (n x m または batch_shape x n x m)。

    Returns:
        Tuple[torch.Size, torch.Size]:
            - input_batch_shape: 入力バッチ形状。
            - aug_batch_shape: 出力次元を含む拡張バッチ形状 (input_batch_shape x (m))。
    """
    input_batch_shape = train_X.shape[:-2]
    aug_batch_shape = input_batch_shape
    num_outputs = train_Y.shape[-1]
    if num_outputs > 1:
        aug_batch_shape = input_batch_shape + torch.Size([num_outputs])
    return input_batch_shape, aug_batch_shape


def singletasklikelihood(
    train_X: Tensor,
    train_Y: Tensor,
    deep: bool = False,
    alpha: float = MIN_INFERRED_NOISE_LEVEL,
) -> GaussianLikelihood:
    """
    単一タスクの GaussianLikelihood を構築するユーティリティ。

    Args:
        train_X: 学習入力データ（形状から batch shape 推定に利用）。
        train_Y: 学習出力データ（同上）。
        deep: Deep GP モデルなどで noise_prior を使わない場合に True。
        alpha: ノイズ下限（noise_constraint の下限値）。

    Returns:
        GaussianLikelihood: GPyTorch 互換の GaussianLikelihood インスタンス。
    """
    # train_X, train_Y から拡張されたバッチサイズを取得
    _, aug_batch_shape = get_batch_dimensions(train_X, train_Y)

    if not deep:
        # 通常の SingleTask モデル用：ノイズ事前分布を設定
        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        return GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=aug_batch_shape,
            noise_constraint=GreaterThan(
                lower_bound=alpha,
                transform=None,
                initial_value=noise_prior.mode,
            ),
        )
    else:
        # Deep GP 用など、noise_prior を使わない設定
        return GaussianLikelihood(
            batch_shape=aug_batch_shape,
            noise_constraint=GreaterThan(lower_bound=alpha),
        )


def multitasklikelihood(
    train_X: Tensor,
    train_Y: Tensor,
    deep: bool = False,
    rank: Optional[int] = None,
    alpha: float = MIN_INFERRED_NOISE_LEVEL,
) -> MultitaskGaussianLikelihood:
    """
    マルチタスク（多出力）Gaussian Likelihood を構築する関数。

    Args:
        train_X: 入力データ（batch shape の推定に使用）。
        train_Y: 出力データ（最終次元がタスク数）。
        deep: Deep GP 用の設定。True の場合は noise_prior を使用しない。
        rank: タスク間のノイズ共分散のランク。未指定の場合はタスク数と同じに設定。
        alpha: ノイズレベルの最小値（noise_constraint の下限値）。

    Returns:
        MultitaskGaussianLikelihood: 多出力 GaussianLikelihood インスタンス。
    """
    # batch_shape: GP のバッチ処理に対応するための shape
    batch_shape, _ = get_batch_dimensions(train_X, train_Y)

    # 出力の最終次元がタスク数
    num_tasks = train_Y.shape[-1]

    # ランクが指定されていなければ num_tasks に合わせる
    if rank is None:
        rank = num_tasks

    if not deep:
        # 通常の GP 用の prior + constraint
        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        return MultitaskGaussianLikelihood(
            num_tasks=num_tasks,
            batch_shape=batch_shape,
            rank=rank,
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                lower_bound=alpha,
                transform=None,
                initial_value=noise_prior.mode,
            ),
        )
    else:
        # Deep GP 用（prior を外すが、batch_shape / rank は維持）
        return MultitaskGaussianLikelihood(
            num_tasks=num_tasks,
            batch_shape=batch_shape,
            rank=rank,
            noise_constraint=GreaterThan(lower_bound=alpha),
        )