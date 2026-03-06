import torch
from gpytorch.mlls import MarginalLogLikelihood

def fit_deepgp_mll(mll, lr: float = 0.01, epoch: int = 50):
    """
    DeepGPモデルの ELBO(Marginal Log Likelihood) を最適化して学習する。

    Args:
        mll: DeepApproximateMLL(VariationalELBO(...)) など。
    """
    model = mll.model

    # モデルと尤度を訓練モードに設定
    model.train()
    model.likelihood.train()

    # Adamオプティマイザの設定
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ★ 修正ポイント 1: train_targets は Tensor なので、そのまま使う
    y_tensor = model.train_targets  # shape: [n] or [n, m]

    # ★ 修正ポイント 2: train_inputs はタプルなので 0 番目を取り出す
    x_tensor = model.train_inputs[0] if isinstance(model.train_inputs, tuple) else model.train_inputs

    # 訓練ループ
    for i in range(epoch):
        optimizer.zero_grad()

        # DeepGP の出力を計算
        # batch_mean=False なので、最後のサンプル次元を平均せずにそのまま ELBO に渡す想定
        output = model.forward(x_tensor, batch_mean=False)

        # 損失を計算 (出力次元数に応じて処理を分岐)
        if (y_tensor.ndim > 1) and (y_tensor.shape[-1] > 1):
            # 多変量の場合: y の shape は [n, m] のままで OK
            loss = -mll(output, y_tensor)
        else:
            # 単変量の場合: shape [n] にして渡す
            loss = -mll(output, y_tensor.view(-1))

        loss.backward()   # 逆伝播で勾配を計算
        optimizer.step()  # パラメータの更新

        # 学習の様子を軽く確認したければ:
        # if (i + 1) % 10 == 0:
        #     print(f"epoch {i+1}/{epoch}, loss={loss.item():.4f}")

def fit_deepkernel_mll(mll, lr=0.01, epoch=50):
    """
    モデルの周辺対数尤度 (Marginal Log Likelihood, MLL) を最適化して、DeepGPモデルを訓練します。

    Args:
        mll (gpytorch.mlls.MarginalLogLikelihood): 訓練対象の周辺対数尤度オブジェクト。
    """
    model = mll.model
    
    # モデルと尤度を訓練モードに設定
    model.train()
    model.likelihood.train()

    # Adamオプティマイザの設定（学習率: 0.01）
    optimizer = torch.optim.Adam([
        {'params': model.deepkernel.feature_extractor.parameters()},
        {'params': model.deepkernel.covar_module.parameters()},
        {'params': model.deepkernel.mean_module.parameters()},
        {'params': model.deepkernel.likelihood.parameters()},
    ], lr=lr)

    y_tensor = model.train_targets
    x_tensor = model.train_inputs[0] if isinstance(model.train_inputs, tuple) else model.train_inputs
    
    # 訓練ループ
    for i in range(epoch):
        optimizer.zero_grad()  # 勾配の初期化
        output = model.forward(x_tensor)  # モデルの出力を計算
        # 損失を計算 (出力次元数に応じて処理を分岐)
        if (len(y_tensor.shape)>1)&(y_tensor.shape[-1]>1):
            loss = -mll(output, y_tensor)  # 多変量の場合
        else:
            loss = -mll(output, y_tensor.view(-1))  # 単変量の場合

        loss.backward()  # 逆伝播で勾配を計算
        optimizer.step()  # パラメータの更新