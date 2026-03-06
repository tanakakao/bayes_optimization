import torch
from torch import Tensor
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple, Any, Dict, Sequence
from sklearn.model_selection import KFold

from bayes_optimization.models.data_preprocessing import (
    prepare_categorical_info, normalize_bounds, preprocess_X, search_bounds, num_and_cat_idx,
    round_to_grid_multi_constraints, impute_iterative_mice, make_multitask_data, filter_cat_features_list,
    make_permutation, apply_permutation
)
from bayes_optimization.models.model_fitting import fit_model
from bayes_optimization.models.metrics import reg_scores, cls_scores
from bayes_optimization.models.acquisition_factory import get_acqf
from bayes_optimization.models.candidate_generator import get_candidate

from contextlib import contextmanager
import torch.nn as nn

import warnings
warnings.simplefilter('ignore')

class BOCandidateModel:
    """ベイズ最適化の候補点探索モデル。

    学習・予測・候補生成に責務を限定し、評価（CV）は外部で行う前提。

    Attributes:
        categorical_cols (List[str]): カテゴリ特徴量の列名。
        target_cols (List[str]): 目的変数の列名。
        cat_targets_cols (List[str]): 2値化（しきい値化）対象の目的変数列名。
        feature_cols (List[str]): 特徴量の列名（カテゴリ/数値混在）。
        numeric_cols (List[str]): 数値特徴量の列名。
        numeric_idx (List[int]): 数値特徴量の列インデックス。
        categorical_idx (List[int]): カテゴリ特徴量の列インデックス。
        cat_targets_idx (List[int]): 2値化対象の目的列インデックス。
        # has_categorical (bool): カテゴリ列が存在するか。
        X (pd.DataFrame|None): 学習前処理前の説明変数（インデックスリセット済み）。
        Y (pd.DataFrame|None): 学習前処理前の目的変数（インデックスリセット済み）。
        train_X (Tensor|None): 学習用X（前処理後）。
        train_Y (Tensor|None): 学習用Y。
        bounds_norm (Tensor|None): 正規化空間の bounds（shape: (2, n_features)）。
        fixed_features_list (Optional[List[Dict[int, float]]]): カテゴリ用固定特徴の基本リスト。
        labels (Optional[Dict[str, Dict[Any, int]]]): カテゴリ列のラベル→インデックス辞書。
        model (Any): BoTorch 等の学習済みモデル。
        multi_model_type (Optional[str]): 多目的時のモデル構成。
        multi_task_type (Optional[str]): Multi-task の構成。
        dtype (torch.dtype): Deep 系は float32、それ以外は double。
        # 候補生成関連
        acq_method (Optional[str]): 使用する獲得関数名（'EI', 'EHI', 'KG' など）。
        n_cand (int): 生成候補数。
        extra_rate (float): 探索空間の外挿率。
        candidates (Optional[pd.DataFrame]): 逆変換済み候補点＋予測統計の表。
        # 制約 / しきい値
        y_constraints_cols (List[str]): 出力側制約の対象列名。
        y_constraints_idx (List[int]): 出力側制約の対象列インデックス。
        y_ops (List[str]): 出力側制約の演算子（'=', '>', '<'）。
        y_thresholds (List[float]): 出力側制約の閾値。
        y_weights (List[float]): 多目的の重み。
        y_directions (List[str]): 'min' / 'max' の方向。
        h_lse (Optional[List[float]]): LSE 系で用いるしきい値（互換用）。
        x_steps (List[float]): 数値列の丸め刻み。
        constraint_cols (List[List[str]]): 入力側制約の列名グループ。
        constraint_idx (List[List[int]]): 入力側制約の列インデックス群。
        constraint_coefs (List[List[float]]): 各制約の係数。
        constraint_values (List[float]): 各制約の値。
        constraint_ops (List[str]): 各制約の演算子（'=', '>', '<'）。
        bounds (Optional[Tensor]): 候補生成時に用いる（正規化空間の）境界。
        acqf (Any): 構築された獲得関数オブジェクト。
        cv_results (Any): 交差検証結果（外部で取得したもの）。
    """

    def __init__(self) -> None:
        """インスタンスを初期化。

        ほとんどの属性は `fit` 呼び出し後に確定します。
        """
        # 入力・列情報
        self.target_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.cat_targets_cols: List[str] = []
        self.feature_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.numeric_idx: List[int] = []
        self.categorical_idx: List[int] = []
        self.cat_targets_idx: List[int] = []
        self.perm: List[int] = None
        # self.has_categorical: bool = False

        self.task_col: Optional[str] = None
        self.task_target_item: Optional[Any] = None

        # 学習後に埋まる属性
        self.train_X: Optional[Tensor] = None
        self.train_Y: Optional[Tensor] = None
        self.X: Optional[pd.DataFrame] = None
        self.Y: Optional[pd.DataFrame] = None
        self.Y_proc: Optional[pd.DataFrame] = None
        self.bounds_norm: Optional[Tensor] = None
        self.cat_features_list: Optional[List[Dict[int, float]]] = None
        self.labels: Optional[Dict[str, Dict[Any, int]]] = None
        self.categorical_features: Dict[int, Sequence[float]] = None
        self.model: Any = None
        self.multi_model_type: Optional[str] = None
        self.multi_task_type: Optional[str] = None
        self.dtype: torch.dtype = torch.double  # 既定は double（Deep 系で float32 に切替）

        # オプション（fit で上書き）
        self.impute: bool = False
        self.robust: bool = False
        self.perturbation: bool = False
        self.heteroscedastic: bool = False
        self.is_high_dim: bool = False
        self.deep_gp: bool = False
        self.deep_kernel: bool = False
        self.alpha: float = 1e-1
        self.lr: float = 1e-2
        self.epoch: int = 300
        self.cat_target_items: Dict[str, Sequence[Any]] = {}

        # 候補生成関連
        self.acq_method: Optional[str] = None
        self.y_constraints_cols: List[str] = []
        self.y_constraints_idx: List[int] = []
        self.y_ops: List[str] = []
        self.y_thresholds: List[float] = []
        self.y_weights: List[float] = []
        self.y_directions: List[str] = []
        self.x_steps: List[float] = []
        self.n_cand: int = 1
        self.h_lse: Optional[List[float]] = None

        self.constraint_cols: List[List[str]] = []
        self.constraint_idx: List[List[int]] = []
        self.constraint_coefs: List[List[float]] = []
        self.constraint_values: List[float] = []
        self.constraint_ops: List[str] = []

        self.extra_rate: float = 0.0
        self.bounds: Optional[Tensor] = None
        self.acqf: Any = None
        self.candidates_raw: Optional[Tensor] = None
        self.candidates: Optional[pd.DataFrame] = None
        self.fixed_features: Optional[Dict[str, float]] = None
        self.cat_fixed_features: Optional[Dict[str, float]] = None
        self.fixed_features_list: Optional[Union[Dict[int, float], List[Dict[int, float]]]] = None

        self.cv_results: Any = None
        self.score_df: Dict[str, pd.DataFrame] = {}

    # ============================
    # 学習
    # ============================
    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        cat_targets_cols: Optional[List[str]] = None,
        df_base: pd.DataFrame = None,
        task_col: str = None,
        task_target_item: Any = None,
        bounds_norm: Optional[Tensor] = None,
        multi_model_type: Optional[str] = None,
        multi_task_type: Optional[str] = None,
        cat_target_items: Optional[Dict[str, Sequence[Any]]] = None,
        impute: bool = False,
        robust: bool = False,
        perturbation: bool = False,
        heteroscedastic: bool = False,
        is_high_dim: bool = False,
        hd_model: str=None, # saas, pca, rembo
        n_components: int = 2, 
        deep_gp: bool = False,
        deep_kernel: bool = False,
        alpha: float = 1e-1,
        lr: float = 1e-2,
        epoch: int = 300,
    ) -> None:
        """モデル学習。

        Args:
            df: 入力データ。
            feature_cols: 特徴量の列名。
            target_cols: 目的変数の列名。
            categorical_cols: カテゴリ特徴量の列名。
            cat_targets_cols: 2値化対象の目的変数列名。
            df_base: multi-task 用のベースデータ（ある場合）。
            task_col: タスクを識別する列名。
            task_target_item: 対象タスクの値。
            bounds_norm: 正規化後の境界（未指定時は自動推定）。
            multi_model_type: 多目的時のモデル構成（'single_task_multi_output' 等）。
            multi_task_type: multi-task モデル構成。
            cat_target_items: 2値化対象の {列名: 正例とみなす値リスト}。
            impute: True なら欠損補完（IterativeImputer）。
            robust: ロバスト化フラグ。
            perturbation: 入力摂動を考慮したロバスト化フラグ。
            heteroscedastic: 不均一分散フラグ。
            is_high_dim: 高次元フラグ。
            deep_gp: Deep Gaussian Process を使用。
            deep_kernel: Deep Kernel Learning を使用。
            alpha: 正則化係数（Deep Kernel）。
            lr: 学習率。
            epoch: 学習エポック数。
        """
        # --- 前提チェック ---
        if df is None or len(df) == 0:
            raise ValueError("学習データが空です。df を確認してください。")
        if not feature_cols or not target_cols:
            raise ValueError("feature_cols と target_cols は空にできません。")
        for c in feature_cols + target_cols:
            if c not in df.columns:
                raise KeyError(f"列がデータに存在しません: {c}")

        if (cat_targets_cols is not None and cat_targets_cols != []) and cat_target_items is None:
            raise ValueError("cat_targets_cols を設定した場合は cat_target_items に{'target': ['a','b',...]}形式で予測対象カテゴリを指定してください。")
        if robust and len(target_cols) > 1 and multi_model_type!="model_list":
            raise ValueError("robust=True での多目的モデルは multi_model_type='model_list' のみ使えます。")
        
        self.target_cols = list(target_cols)
        feature_cols = list(feature_cols)
        self.categorical_cols = list(categorical_cols or [])
        self.cat_targets_cols = list(cat_targets_cols or [])

        # 特徴量列を数値/カテゴリに分割
        f_idx = num_and_cat_idx(feature_cols, self.categorical_cols)
        self.feature_cols, self.numeric_cols, self.numeric_idx, self.categorical_idx = f_idx

        self.perm = make_permutation(feature_cols, self.feature_cols)

        self.cat_targets_idx = [target_cols.index(c) for c in self.cat_targets_cols]
        # indices_list = num_and_cat_idx(target_cols, self.cat_targets_cols)
        # _, _, _, self.cat_targets_idx = indices_list

        # self.has_categorical = len(self.categorical_cols) > 0
        self.cat_target_items = dict(cat_target_items or {})

        self.task_col = task_col
        self.task_target_item = task_target_item

        # multi-task のためのデータ整形
        if df_base is not None:
            df = make_multitask_data(
                df_base,
                df,
                self.feature_cols,
                self.target_cols,
                self.task_col,
                self.task_target_item,
            )
            self.feature_cols = self.feature_cols + ["task"]

        elif task_col:
            df["task"] = (df[task_col] == task_target_item).astype(int)
            self.feature_cols = self.feature_cols + ["task"]

        # 欠損処理
        _df_all = self._handle_missing_values(df, impute)

        self.X = _df_all[self.feature_cols].reset_index(drop=True)
        self.Y = _df_all[self.target_cols].reset_index(drop=True)

        # 多目的時のモデル構成
        self._configure_model_types(df_base, multi_model_type, multi_task_type)

        # 学習パラメータ
        self.impute = False if impute is None else impute
        self.robust = False if robust is None else robust
        self.perturbation = False if perturbation is None else perturbation
        self.heteroscedastic = False if heteroscedastic is None else heteroscedastic
        self.is_high_dim = False if is_high_dim is None else is_high_dim
        self.hd_model = "saas" if (hd_model is None and is_high_dim) else hd_model
        self.n_components = 2 if n_components is None else n_components
        self.deep_gp = False if deep_gp is None else deep_gp
        self.deep_kernel = False if deep_kernel is None else deep_kernel
        self.alpha = 1e-1 if alpha is None else alpha
        self.lr = 1e-2 if lr is None else lr
        self.epoch = 300 if epoch is None else epoch
        self.dtype = torch.float32 if (deep_gp or deep_kernel) else torch.double

        # 前処理（カテゴリ変換など）
        self.cat_features_list, self.labels, self.categorical_features = prepare_categorical_info(
            X=self.X,
            categorical_cols=self.categorical_cols,
        )
        self.train_X = self._preprocess_X(self.X)

        # 正規化後の境界
        if bounds_norm is not None:
            bounds_norm = bounds_norm[:,self.perm]
        else:
            bounds_norm = None
        self._build_bounds(bounds_norm.to(self.dtype) if bounds_norm is not None else None)

        # # multi-task のときの task 列の処理
        # if self.multi_task_type is not None:
        #     self.train_X = torch.concat(
        #         [self.train_X, torch.tensor(self.X[["task"]].values, dtype=self.dtype)], dim=-1
        #     )
        #     self.bounds_norm = torch.concat(
        #         [self.bounds_norm, torch.tensor([[0], [1]], dtype=self.dtype)], dim=-1
        #     )

        # 目的の 2値化
        self.Y_proc = self.Y.copy()
        for tgt in self.cat_targets_cols:
            pos_set = set(self.cat_target_items.get(tgt, [self.Y[tgt].values[0]]))
            self.Y_proc[tgt] = self.Y_proc[tgt].apply(lambda v: float(v in pos_set))
        self.train_Y = torch.tensor(self.Y_proc.to_numpy(), dtype=self.dtype)

        # モデル学習
        self.model = self._fit_model(self.train_X, self.train_Y)

    def _fit_model(self, X: Tensor, Y: Tensor):
        return fit_model(
            X,
            Y,
            self.bounds_norm,
            self.multi_model_type,
            self.multi_task_type,
            self.categorical_idx,
            self.cat_targets_idx,
            self.robust,
            self.perturbation,
            self.heteroscedastic,
            self.is_high_dim,
            self.hd_model,
            self.n_components,
            self.deep_gp,
            self.deep_kernel,
            self.alpha,
            self.lr,
            self.epoch,
        )

    def _handle_missing_values(self, df: pd.DataFrame, impute: bool) -> pd.DataFrame:
        """特徴量・目的変数の欠損処理を行う。

        impute=True のとき:
            - 目的変数が欠損している行だけを除外
            - 特徴量の欠損は IterativeImputer で補完

        impute=False のとき:
            - 特徴量・目的変数ともに欠損行を除外
        """
        feat_cols = self.feature_cols
        all_cols = feat_cols + self.target_cols

        if impute:
            # 目的変数だけは欠損不可
            _df_all = impute_iterative_mice(
                df[all_cols],
                cat_cols=self.categorical_cols + self.cat_targets_cols,
            )
        else:
            # 全列で欠損行を除外
            _df_all = df[all_cols].dropna(subset=all_cols)

        if len(_df_all) == 0:
            raise ValueError("欠損処理後のデータが空です。列設定や欠損状況を確認してください。")

        return _df_all

    def _configure_model_types(self, df_base, multi_model_type, multi_task_type):
        if self.Y.shape[1] > 1:
            self.multi_model_type = "single_task_multi_output" if multi_model_type is None else multi_model_type
            self.multi_model_type = (
                "model_list" if len(self.cat_targets_cols) > 0 else self.multi_model_type
            )
        else:
            self.multi_model_type = None

        self.multi_task_type = multi_task_type
        if df_base is not None and self.multi_model_type is None:
            self.multi_task_type = "multi_task"
            self.multi_model_type = "model_list"
        elif df_base is None and multi_task_type is None:
            self.multi_task_type = None       
        
    def _preprocess_X(self, X: Union[pd.DataFrame, np.ndarray, Tensor]) -> Tensor:
        """
        X を前処理して Tensor 化（正規化・ラベル変換・固定値反映など）する。
        - DataFrame: feature_cols のみを抽出して preprocess_X に渡す
        - ndarray/Tensor: そのまま preprocess_X に渡す（列順は feature_cols に一致している前提）
        """
        if isinstance(X, pd.DataFrame):
            X_in = X[self.feature_cols]
        else:
            X_in = X
        return preprocess_X(
            X_in,
            self.fixed_features_list,
            self.labels,
            dtype=self.dtype,
        )

    def _build_bounds(self, bounds_norm):
        self.bounds_norm = normalize_bounds(
            self.train_X,
            # num_idx=self.numeric_idx,
            bounds=bounds_norm,
            dtype=self.dtype,
        )    

    # ============================
    # 候補生成
    # ============================
    def candidate(
        self,
        acq_method: str,
        n_cand: int = 1,
        bounds: Optional[Tensor] = None,
        bounds_not_neg: Optional[List[bool]] = None,
        extra_rate: Optional[float] = None,
        x_steps: Optional[Sequence[float]] = None,
        fixed_features: Optional[Dict[str, float]] = None,
        cat_fixed_features: Optional[Dict[str, List[float]]] = None,
        x_constraint_cols: Optional[List[List[str]]] = None,
        x_constraint_coefs: Optional[List[List[float]]] = None,
        x_constraint_values: Optional[List[float]] = None,
        x_constraint_ops: Optional[List[str]] = None,
        y_constraints_cols: Optional[List[str]] = None,
        y_ops: Optional[List[Optional[str]]] = None,
        y_thresholds: Optional[List[float]] = None,
        y_weights: Optional[List[float]] = None,
        y_directions: Optional[List[str]] = None,
        evo_method: Optional[str] = None,
        risk_type: Optional[str] = None,
        k_sparse_spec: Optional[Tuple[List[str], int]] = None,
    ) -> pd.DataFrame:
        """ベイズ最適化で候補点を生成。

        Args:
            acq_method: 獲得関数名（例: 'EI', 'UCB', 'KG', 'EHI'）。
            n_cand: 生成する候補数。
            bounds: 探索境界（正規化空間）。未指定なら訓練域から自動。
            bounds_not_neg: 各変数の非負制約フラグ。
            extra_rate: 探索境界の外挿率（None=0.0）。
            x_steps: 数値特徴量の丸め刻み（len == #numeric を推奨）。
            fixed_features: 固定したい特徴量 {列名: 値}。
            x_constraint_cols: 入力側制約の列名グループ（[[colA, colB], ...]）。
            x_constraint_coefs: 各グループの係数（[[1,1], ...]）。
            x_constraint_values: 各グループの閾値（[1.0, ...]）。
            x_constraint_ops: 各グループの演算子（['=', '>', '<', ...]）。
            y_constraints_cols: 出力側制約の列名。
            y_ops: 出力側制約の演算子（None を含み得る）。
            y_thresholds: 出力側制約の閾値。
            y_weights: 多目的の重み。
            y_directions: 'min' / 'max'。
        """
        if self.model is None or self.train_X is None or self.bounds_norm is None:
            raise RuntimeError("モデルが未学習です。先に .fit() を実行してください。")

        # 保存（再利用用）
        self.acq_method = acq_method
        self.n_cand = int(n_cand)
        self.extra_rate = 0.0 if extra_rate is None else float(extra_rate)

        # 出力側制約
        self.y_constraints_cols = list(y_constraints_cols or [])
        self.y_constraints_idx = [self.target_cols.index(c) for c in self.y_constraints_cols]
        self.y_ops = list(y_ops or [])
        self.y_thresholds = list(y_thresholds or [])
        self.y_weights = list(y_weights or [])
        self.y_directions = list(y_directions or [])
        if len(self.y_directions)<len(self.target_cols):
            self.y_directions = self.y_directions + ["max"] * (len(self.target_cols) - len(self.y_directions))
        self.h_lse = self.y_thresholds
        if not (
            len(self.y_constraints_idx) == len(self.y_ops) == len(self.y_thresholds)
            or len(self.y_constraints_idx) == 0
        ):
            raise ValueError("y側制約の長さが一致していません。")

        # 入力側制約
        if x_steps:
            self.x_steps = apply_permutation(x_steps, self.perm)
        else:
            self.x_steps = []
        self.constraint_cols = list(x_constraint_cols or [])
        self.constraint_idx = [
            [self.feature_cols.index(c) for c in grp] for grp in self.constraint_cols
        ]
        self.constraint_values = list(x_constraint_values or [])
        self.constraint_ops = list(x_constraint_ops or [])
        if not (
            len(self.constraint_idx) == len(self.constraint_values) == len(self.constraint_ops)
            or len(self.constraint_idx) == 0
        ):
            raise ValueError("x側制約の長さが一致していません。")

        # 係数が未指定なら 1.0 で埋める
        if x_constraint_coefs and len(x_constraint_coefs) == len(self.constraint_idx):
            self.constraint_coefs = x_constraint_coefs
        else:
            self.constraint_coefs = [[1.0] * len(g) for g in self.constraint_idx]

        # 固定値（列名→（現在は raw スケールの Tensor））
        _fixed_features = dict(fixed_features or {})
        self.cat_fixed_features = dict(cat_fixed_features or {})
        self.fixed_features = _fixed_features
        if _fixed_features:
            fixed_norm: Dict[int, Tensor] = {}
            for k, v in _fixed_features.items():
                idx = self.feature_cols.index(k)
                # fixed_norm[idx] = torch.tensor([float(v)], dtype=self.dtype)
                fixed_norm[idx] = float(v)
            self.fixed_features_list = (
                fixed_norm
                if self.cat_features_list is None
                else [{**d, **fixed_norm} for d in self.cat_features_list]
            )
            self.fixed_features_list = filter_cat_features_list(
                self.feature_cols,
                self.labels,
                self.fixed_features_list,
                self.cat_fixed_features
            )
        else:
            self.fixed_features_list = filter_cat_features_list(self.feature_cols, self.labels, self.cat_features_list, self.cat_fixed_features)
            # self.fixed_features_list = self.cat_features_list
            
        self.evo_method=evo_method
        if k_sparse_spec is not None:
            k_sparse_spec = ([self.feature_cols.index(k) for k in k_sparse_spec[0]], k_sparse_spec[1])
        self.k_sparse_spec=k_sparse_spec
        self.risk_type = risk_type
        
        # 探索境界
        # if _bounds_not_neg and len(_bounds_not_neg) != len(self.feature_cols):
        #     raise ValueError("bounds_not_neg の長さが feature_cols と一致していません。")
        if bounds_not_neg is not None:
            bounds_not_neg = bounds_not_neg[:,self.perm]
        else:
            bounds_not_neg = None
        _bounds_not_neg = list(bounds_not_neg or [])

        if bounds is not None:
            bounds = bounds[:,self.perm]
        else:
            bounds = None
        self.bounds = self._get_bounds(bounds, _bounds_not_neg).to(self.dtype)

        # if self.k_sparse_spec is not None:
        #     if not all(self.bounds[0, self.k_sparse_spec[0]]<=1e-8):
        #         raise ValueError("選択数制約を付ける項目のboundsには0を含ませてください。探索下限が0以上のものがあります。")
        #     if not all(self.bounds[1, self.k_sparse_spec[0]]>=1e-8):
        #         raise ValueError("選択数制約を付ける項目のboundsには0を含ませてください。探索上限が0以下のものがあります。")
        
        # 獲得関数と候補生成
        self.acqf = self._build_acqf()
        candidates = self._generate_candidates()

        if self.multi_task_type is not None:
            candidates = self.acqf._construct_X_full(candidates)

        candidates = candidates.detach().numpy().copy()
        
        if len(candidates.shape)==1:
            candidates = candidates[np.newaxis]
        
        self.candidates_raw = candidates

        # 元スケールに戻す
        candidates = self._tensor_to_dataframe(candidates)

        # 刻み幅で丸め（数値列のみ）
        steps_t = self._resolve_numeric_steps_tensor()
        if steps_t is not None and len(self.numeric_cols) > 0:
            bounds_clip = self.bounds[:, self.numeric_idx]
            base = bounds_clip[0]
            candidates[self.numeric_cols] = round_to_grid_multi_constraints(
                candidates[self.numeric_cols],
                steps_t,
                base,
                bounds_clip,
                self.constraint_idx,
                self.constraint_coefs,
                self.constraint_values,
                self.constraint_ops,
            )
            self.candidates_raw[:, self.numeric_idx] = candidates[self.numeric_cols].values

        # 予測（平均・標準偏差）
        pred_mean, pred_std = self.predict(candidates.copy())
        pred_mean.columns = [f"{c}_mean" for c in pred_mean.columns]
        pred_std.columns = [f"{c}_std" for c in pred_std.columns]

        self.candidates = pd.concat([candidates, pred_mean, pred_std], axis=1)
        return self.candidates

    def _get_bounds(self, bounds: Optional[Tensor], bounds_not_neg: List[bool]):
        _bounds = search_bounds(
            train_X=self.train_X,
            bounds=bounds,
            bounds_not_neg=bounds_not_neg,
            extra_rate=self.extra_rate,
        )
        # Deepモデルの場合は float32 に統一
        if self.deep_gp or self.deep_kernel:
            # _bounds = torch.tensor(_bounds, dtype=torch.float32)
            _bounds = _bounds.to(torch.float32)
        if self.multi_task_type is not None:
            _bounds = _bounds[:, :-1]
        return _bounds

    def _build_acqf(self):
        return get_acqf(
            model=self.model,
            train_X=self.train_X,
            train_Y=self.train_Y,
            bounds=self.bounds,
            acq_method=self.acq_method,
            y_constraints_idx=self.y_constraints_idx,
            y_ops=self.y_ops,
            y_thresholds=self.y_thresholds,
            y_weights=self.y_weights,
            y_directions=self.y_directions,
            h_lse=self.h_lse,
            risk_type=self.risk_type,
            n_cand=self.n_cand,
            dtype=self.dtype,
        )

    def _generate_candidates(self):
        return get_candidate(
            acqf=self.acqf,
            constraint_idx=self.constraint_idx,
            constraint_coefs=self.constraint_coefs,
            constraint_values=self.constraint_values,
            constraint_ops=self.constraint_ops,
            categorical_features=self.categorical_features,
            fixed_features=self.fixed_features,
            fixed_features_list=self.fixed_features_list,
            bounds=self.bounds,
            n_cand=self.n_cand,
            sequential=(self.n_cand > 1) and (self.acq_method != "KG"),
            evo_method=self.evo_method,
            k_sparse_spec=self.k_sparse_spec,
            dtype=self.dtype,
        )

    def _tensor_to_dataframe(
        self,
        X: Union[np.ndarray, Tensor],
    ) -> pd.DataFrame:
        """正規化空間の Tensor / ndarray を DataFrame に変換し、カテゴリ列を復元。"""
        if isinstance(X, Tensor):
            X = X.detach().cpu().numpy()
            if len(X.shape)==1:
                X = X[np.newaxis]
            results = pd.DataFrame(X, columns=self.feature_cols)
        else:
            results = pd.DataFrame(X, columns=self.feature_cols)
        

        for idx, cat_col in enumerate(self.categorical_cols):
            # labels[cat_col]: {label -> index} を仮定してインデックス→ラベルに戻す
            mapping = self.labels[cat_col]  # type: ignore[index]
            inv = {v: k for k, v in mapping.items()}
            results[cat_col] = results[cat_col].astype(int).map(inv)
        return results

    def _resolve_numeric_steps_tensor(self) -> Optional[Tensor]:
        """
        self.x_steps を numeric_cols に対応する Tensor へ変換。
        受け付ける形式:
          - len(x_steps) == len(numeric_cols)  （推奨）
          - len(x_steps) == len(feature_cols)  （旧仕様互換）
        """
        steps = list(getattr(self, "x_steps", []) or [])
        if len(steps) == 0 or len(self.numeric_cols) == 0:
            return None
    
        if len(steps) == len(self.numeric_cols):
            steps_numeric = steps
        elif len(steps) == len(self.feature_cols):
            steps_numeric = [steps[i] for i in self.numeric_idx]
        else:
            raise ValueError(
                f"x_steps の長さが不正です: len(x_steps)={len(steps)}. "
                f"len(numeric_cols)={len(self.numeric_cols)} または len(feature_cols)={len(self.feature_cols)} を指定してください。"
            )
    
        return torch.tensor(steps_numeric, dtype=self.dtype)
    
    # ============================
    # 予測・評価
    # ============================
    def predict(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray, Tensor]] = None,
        use_type: str = "candidate",
        model: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """予測（平均・標準偏差）

        Args:
            X: 入力特徴量。None の場合、`use_type` に応じて候補 or 学習データを使用。
            use_type: 'candidate' or 'train' または None（X 明示指定時）。
            model: 予測で使うモデル（未指定は self.model）。
        Returns:
            (pred_mean_df, pred_std_df)
        """
        if self.model is None:
            raise RuntimeError("モデルが未学習です。先に .fit() を実行してください。")

        model_used = self.model if model is None else model
        scaled_X = self._prepare_scaled_X(X=X, use_type=use_type, require_acqf=False)
        
        with self._disable_perturbation():
            with torch.no_grad():
                if self.heteroscedastic:
                    posterior = model_used.posterior(scaled_X, observation_noise=True)
                else:
                    posterior = model_used.posterior(scaled_X)
        if self.is_high_dim and self.hd_model=="saas":
            pred_mean = posterior.mixture_mean.detach().cpu().numpy()
            pred_std = posterior.mixture_variance.sqrt().detach().cpu().numpy()
        else:
            pred_mean = posterior.mean.detach().cpu().numpy()
            pred_std = posterior.variance.sqrt().detach().cpu().numpy()

        if self.cat_targets_idx:
            # 分類目的については std^2（variance）を返す仕様を踏襲
            if len(self.target_cols)==1:
                pred_std = pred_std ** 2
            elif len(self.cat_targets_idx)>0:
                pred_std[:, self.cat_targets_idx] = pred_std[:, self.cat_targets_idx] ** 2

        
        pred_mean_df = pd.DataFrame(pred_mean, columns=self.target_cols)
        pred_std_df = pd.DataFrame(pred_std, columns=self.target_cols)
        return pred_mean_df, pred_std_df

    def _prepare_scaled_X(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray, Tensor]],
        use_type: Optional[str],
        *,
        require_acqf: bool = False,
    ) -> Tensor:
        """
        予測/獲得関数評価の入力 X を、モデルに入れられる Tensor に整形する。
        - X is None の場合は use_type に応じて train/candidate を参照
        - multi-task の場合、DataFrame なら task_col から task を付与（なければ 1 を仮定）
        - Tensor/ndarray で task 次元が欠ける場合は末尾に task=1 を付与
        """
        if X is None:
            if use_type == "candidate":
                if self.candidates is None:
                    raise RuntimeError("候補が存在しません。先に .candidate() を実行するか X を指定してください。")
                X_src: Union[pd.DataFrame, np.ndarray, Tensor] = self.candidates[self.feature_cols]
                scaled_X = self._preprocess_X(X_src)
            elif use_type == "train":
                if self.train_X is None:
                    raise RuntimeError("train_X が未設定です。")
                scaled_X = self.train_X
            else:
                raise ValueError("X が None の場合、use_type は 'candidate' または 'train' を指定してください。")
        else:
            if isinstance(X, pd.DataFrame):
                X_df = X.copy()
                if self.multi_task_type is not None and "task" not in X_df.columns:
                    if self.task_col is not None and self.task_col in X_df.columns:
                        X_df["task"] = (X_df[self.task_col] == self.task_target_item).astype(int)
                    else:
                        # task_col が無い場合は target task を仮定（実務上の利便性優先）
                        X_df["task"] = 1
                scaled_X = self._preprocess_X(X_df)
    
            elif isinstance(X, Tensor):
                scaled_X = X.to(dtype=self.dtype)
    
            elif isinstance(X, np.ndarray):
                scaled_X = self._preprocess_X(X)
    
            else:
                raise TypeError("X は Tensor / DataFrame / ndarray のいずれかである必要があります。")
    
        # multi-task で task 次元が欠けている（= feature_cols より 1 少ない）場合は task=1 を付与
        if self.multi_task_type is not None and scaled_X.size(-1) == len(self.feature_cols) - 1:
            ones = torch.ones((*scaled_X.shape[:-1], 1), dtype=scaled_X.dtype, device=scaled_X.device)
            scaled_X = torch.cat([scaled_X, ones], dim=-1)
    
        # それでも shape が合わない場合のみ、acqf 側の補助にフォールバック
        if self.multi_task_type is not None and scaled_X.size(-1) != len(self.feature_cols):
            if require_acqf:
                if self.acqf is None:
                    raise RuntimeError("multi-task 入力補完に acqf が必要ですが未構築です。先に .candidate() を実行してください。")
                scaled_X = self.acqf._construct_X_full(scaled_X)
    
        return scaled_X
    
    @contextmanager
    def _disable_perturbation(self, model: Optional[Any] = None):
        """
        InputPerturbation を一時的に無効化して posterior を取得する。
        model 引数があればそれを対象にする（predict(model=...) と整合）。
        """
        model_used = self.model if model is None else model
    
        original_perturb = None
        try:
            if hasattr(model_used, "input_transform") and model_used.input_transform is not None:
                if hasattr(model_used.input_transform, "perturb"):
                    original_perturb = model_used.input_transform.perturb
                    model_used.input_transform.perturb = nn.Identity()
            yield
        finally:
            if original_perturb is not None and hasattr(model_used, "input_transform") and model_used.input_transform is not None:
                model_used.input_transform.perturb = original_perturb

    def score(
        self,
        X: Optional[Union[np.ndarray, Tensor]] = None,
        y: Optional[pd.DataFrame] = None,
        model: Optional[Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        """回帰/分類スコア（RMSE, MAE, MAPE, R2 / accuracy, precision, recall, F1 等）。

        Args:
            X: 入力特徴量。None の場合は学習データを使用。
            y: 正解データ（DataFrame 推奨）。None の場合は学習データ。
            model: 評価に用いるモデル。
        Returns:
            target -> score_df の辞書。
        """
        if X is None:
            pred_mean_df, _ = self.predict(use_type="train", model=model)
            y_tensor = self.train_Y
        else:
            if y is None:
                raise ValueError("X を指定する場合は y も指定してください。")
            pred_mean_df, _ = self.predict(X=X, use_type=None, model=model)
            y_proc = y.copy()
            for tgt in self.cat_targets_cols:
                pos_set = set(self.cat_target_items.get(tgt, []))
                y_proc[tgt] = y_proc[tgt].apply(lambda v: float(v in pos_set))
            y_tensor = torch.tensor(y_proc.to_numpy(), dtype=self.dtype)

        scores: Dict[str, pd.DataFrame] = {}
        for i, target in enumerate(self.target_cols):
            pred = pred_mean_df[target].values
            true = y_tensor[:, i].cpu().numpy()
            if target in self.cat_targets_cols:
                pred = (pred > 0.5).astype(float)
                score_df = cls_scores(true, pred, target)
            else:
                score_df = reg_scores(true, pred, target)
            scores[target] = score_df

        self.score_df = scores  # type: ignore[attr-defined]
        return self.score_df

    # ============================
    # 獲得関数
    # ============================
    def acquisition_function(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray, Tensor]] = None,
        use_type: str = "candidate",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """獲得関数

        Args:
            X: 入力特徴量。None の場合、`use_type` に応じて候補 or 学習データを使用。
            use_type: 'candidate' or 'train' または None（X 明示指定時）。
            model: 予測で使うモデル（未指定は self.model）。
        Returns:
            (pred_mean_df, pred_std_df)
        """
        if self.acqf is None:
            raise RuntimeError("獲得関数が定義されていません。先に .candidate() を実行してください。")

        scaled_X = self._prepare_scaled_X(X=X, use_type=use_type, require_acqf=False)

        with torch.no_grad():
            acqf = self.acqf(scaled_X.unsqueeze(1))
        return acqf.numpy()
    
    # ============================
    # CKPT 入出力
    # ============================
    def load_cv_results(self, cv_result: Any) -> None:
        self.cv_results = cv_result

    def to_checkpoint(self, include_data: bool = True) -> Dict[str, Any]:
        """学習済みモデルを“完全再現”できるチェックポイント dict を返す。"""
        if self.model is None or self.train_X is None or self.train_Y is None:
            raise ValueError("モデルが未学習です。export には .fit() 後のインスタンスが必要です。")

        list_hidden_dims = getattr(self.model, "list_hidden_dims", None)
        hidden_dim = getattr(self.model, "hidden_dim", None)

        hparams: Dict[str, Any] = {
            "multi_model_type": getattr(self, "multi_model_type", None),
            "multi_task_type": getattr(self, "multi_task_type", None),
            "task_target_item": getattr(self, "task_target_item", None),
            "impute": getattr(self, "impute", False),
            "robust": getattr(self, "robust", False),
            "perturbation": getattr(self, "perturbation", False),
            "heteroscedastic": getattr(self, "heteroscedastic", False),
            "is_high_dim": getattr(self, "is_high_dim", False),
            "hd_model": getattr(self, "hd_model", None),
            "n_components": getattr(self, "n_components", 2),
            "deep_gp": getattr(self, "deep_gp", False),
            "deep_kernel": getattr(self, "deep_kernel", False),
            "alpha": getattr(self, "alpha", 1e-1),
            "lr": getattr(self, "lr", 1e-2),
            "epoch": getattr(self, "epoch", 300),
            "cat_target_items": dict(getattr(self, "cat_target_items", {}) or {}),
            "fixed_features_list": getattr(self, "fixed_features_list", None),
            "task_col": getattr(self, "task_col", None),
        }
        if list_hidden_dims is not None:
            try:
                hparams["list_hidden_dims"] = list(list_hidden_dims)
            except Exception:
                pass
        if hidden_dim is not None:
            try:
                hparams["hidden_dim"] = int(hidden_dim)
            except Exception:
                pass

        meta = {
            "feature_cols": list(self.feature_cols or []),
            "target_cols": list(self.target_cols or []),
            "numeric_cols": list(getattr(self, "numeric_cols", []) or []),
            "categorical_cols": list(self.categorical_cols or []),
            "cat_targets_cols": list(self.cat_targets_cols or []),
            "dtype": "float32"
            if getattr(self, "dtype", torch.double) == torch.float32
            else "float64",
            "task_col": getattr(self, "task_col", None),
        }
        indices = {
            "numeric_idx": list(getattr(self, "numeric_idx", []) or []),
            "categorical_idx": list(getattr(self, "categorical_idx", []) or []),
            "cat_targets_idx": list(getattr(self, "cat_targets_idx", []) or []),
        }

        bounds_norm_cpu = (
            self.bounds_norm.detach().cpu()
            if getattr(self, "bounds_norm", None) is not None
            else None
        )

        train_pack = {
            "X": (self.X if include_data else None),
            "Y": (self.Y if include_data else None),
            "Y_proc": (self.Y_proc if include_data else None),
            "train_X": self.train_X.detach().cpu(),
            "train_Y": self.train_Y.detach().cpu(),
        }

        candparams: Dict[str, Any] = {
            "acq_method": getattr(self, "acq_method", None),
            "n_cand": getattr(self, "n_cand", 1),
            "extra_rate": getattr(self, "extra_rate", 0.0),
            "evo_method": getattr(self, "evo_method", None),
            "k_sparse_spec": getattr(self, "k_sparse_spec", None),
            "risk_type": getattr(self, "risk_type", None),
            "y_constraints_cols": getattr(self, "y_constraints_cols", []),
            "y_constraints_idx": getattr(self, "y_constraints_idx", []),
            "y_ops": getattr(self, "y_ops", []),
            "y_thresholds": getattr(self, "y_thresholds", []),
            "y_weights": getattr(self, "y_weights", []),
            "y_directions": getattr(self, "y_directions", []),
            "h_lse": getattr(self, "h_lse", []),
            "x_steps": getattr(self, "x_steps", []),
            "constraint_cols": getattr(self, "constraint_cols", []),
            "constraint_idx": getattr(self, "constraint_idx", []),
            "constraint_values": getattr(self, "constraint_values", []),
            "constraint_ops": getattr(self, "constraint_ops", []),
            "constraint_coefs": getattr(self, "constraint_coefs", []),
            "fixed_features": getattr(self, "fixed_features", None),
            "cat_fixed_features": getattr(self, "cat_fixed_features", None),
            "bounds": getattr(self, "bounds", None),
            "candidates": getattr(self, "candidates", None),
            "candidates_raw": getattr(self, "candidates_raw", None),
            "evo_method": getattr(self, "evo_method", None),
            "risk_type": getattr(self, "risk_type", None),
            "k_sparse_spec": getattr(self, "k_sparse_spec", None),
        }

        ckpt: Dict[str, Any] = {
            "meta": meta,
            "ckpt_version": 1,
            "cat_features_list": getattr(self, "cat_features_list", None),
            "categorical_features": getattr(self, "categorical_features", None),
            "indices": indices,
            "labels": getattr(self, "labels", None),
            "bounds_norm": bounds_norm_cpu,
            "train": train_pack,
            "hparams": hparams,
            "state_dict": self.model.state_dict(),
            "cand": candparams,
            "cv_results": getattr(self, "cv_results", None),
        }
        return ckpt

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]) -> "BOCandidateModel":
        """チェックポイントからモデルを復元して返す。"""
        m = cls()
        m._load_from_checkpoint(ckpt)
        return m

    # 既存互換: 外部から呼ぶ必要があれば残す（非推奨）
    def _load_from_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        meta = ckpt["meta"]
        indices = ckpt.get("indices", {})
        self.labels = ckpt.get("labels", None)
        self.bounds_norm = ckpt.get("bounds_norm")

        self.cat_features_list = ckpt.get("cat_features_list", None)
        self.categorical_features = ckpt.get("categorical_features", None)

        train = ckpt.get("train", {})
        hparams = ckpt.get("hparams", {})
        state = ckpt["state_dict"]
        cand = ckpt["cand"]
        cv_results = ckpt.get("cv_results")

        self.feature_cols = meta["feature_cols"]
        self.target_cols = meta["target_cols"]
        self.numeric_cols = meta["numeric_cols"]
        self.categorical_cols = meta["categorical_cols"]
        self.cat_targets_cols = meta["cat_targets_cols"]
        self.dtype = (
            torch.double if meta.get("dtype", "float64") == "float64" else torch.float32
        )

        self.numeric_idx = indices.get("numeric_idx", [])
        self.categorical_idx = indices.get("categorical_idx", [])
        self.cat_targets_idx = indices.get("cat_targets_idx", [])

        self.X = None if train.get("X") is None else pd.DataFrame(train["X"])  # type: ignore[arg-type]
        self.Y = None if train.get("Y") is None else pd.DataFrame(train["Y"])  # type: ignore[arg-type]
        self.Y_proc = (
            None if train.get("Y_proc") is None else pd.DataFrame(train["Y_proc"])
        )  # type: ignore[arg-type]
        self.train_X = train.get("train_X")
        self.train_Y = train.get("train_Y")

        if self.train_X is None or self.train_Y is None:
            raise ValueError("checkpoint に train_X/train_Y が含まれていないため復元できません。to_checkpoint(include_data=True) で保存してください。")

        self.multi_model_type = hparams.get("multi_model_type")
        self.multi_task_type = hparams.get("multi_task_type")
        self.impute = hparams.get("impute", False)
        self.robust = hparams.get("robust", False)
        self.perturbation = hparams.get("perturbation", False)
        self.heteroscedastic = hparams.get("heteroscedastic", False)
        self.is_high_dim = hparams.get("is_high_dim", False)
        self.hd_model = hparams.get("hd_model", None)
        self.n_components = hparams.get("n_components", 2)
        self.deep_gp = hparams.get("deep_gp", False)
        self.deep_kernel = hparams.get("deep_kernel", False)
        self.alpha = hparams.get("alpha", 1e-1)
        self.lr = hparams.get("lr", 1e-2)
        self.epoch = hparams.get("epoch", 300)
        self.cat_target_items = hparams.get("cat_target_items", {})
        self.fixed_features_list = hparams.get("fixed_features_list", None)
        # task_col は hparams 優先、なければ meta から復元
        self.task_col = hparams.get("task_col", meta.get("task_col", None))
        self.task_target_item = hparams.get("task_target_item", None)

        self.has_categorical = len(self.categorical_cols) > 0

        # 器だけ構築して state_dict を流し込む
        self.model = fit_model(
            self.train_X,
            self.train_Y,
            self.bounds_norm,
            self.multi_model_type,
            self.multi_task_type,
            self.categorical_idx,
            self.cat_targets_idx,
            self.robust,
            self.perturbation,
            self.heteroscedastic,
            self.is_high_dim,
            self.deep_gp,
            self.deep_kernel,
            self.alpha,
            self.lr,
            self.epoch,
            build_only=True,
        )
        self.model.load_state_dict(state, strict=False)

        # 候補生成関連
        self.acq_method = cand.get("acq_method")
        self.n_cand = cand.get("n_cand", 1)
        self.extra_rate = cand.get("extra_rate", 0.0)

        self.evo_method = cand.get("evo_method", None)
        self.k_sparse_spec = cand.get("k_sparse_spec", None)
        self.risk_type = cand.get("risk_type", None)

        self.y_constraints_cols = cand.get("y_constraints_cols", [])
        self.y_constraints_idx = cand.get("y_constraints_idx", [])
        self.y_ops = cand.get("y_ops", [])
        self.y_thresholds = cand.get("y_thresholds", [])
        self.y_weights = cand.get("y_weights", [])
        self.y_directions = cand.get("y_directions", [])
        self.h_lse = cand.get("h_lse", [])

        self.x_steps = cand.get("x_steps", [])
        self.constraint_cols = cand.get("constraint_cols", [])
        self.constraint_idx = cand.get("constraint_idx", [])
        self.constraint_values = cand.get("constraint_values", [])
        self.constraint_ops = cand.get("constraint_ops", [])
        self.constraint_coefs = cand.get("constraint_coefs", [])

        self.fixed_features = cand.get("fixed_features")
        self.cat_fixed_features = cand.get("cat_fixed_features")
        self.bounds = cand.get("bounds")
        self.candidates = cand.get("candidates")
        self.candidates_raw = cand.get("candidates_raw")

        # cand 情報がある場合のみ acqf を復元（fit 直後の ckpt でも load できるように）
        self.acqf = None
        if self.acq_method is not None and self.bounds is not None:
            self.acqf = self._build_acqf()

        self.cv_results = cv_results


def _mean_score(
    score_cv: Dict[str, pd.DataFrame],
    target: str,
    cv_type: str = "train",
) -> pd.DataFrame:
    _score = score_cv[target].groupby(level=0).mean()
    _score.index = [[cv_type] * len(_score), _score.index.tolist()]
    return _score


def _concat(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    return (
        pd.concat(df_list)
        .groupby("index")
        .mean()
        .reset_index(drop=False)
        .sort_values("index")
        .set_index("index")
    )


def cross_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    cat_targets_cols: Optional[List[str]] = None,
    df_base: pd.DataFrame = None,
    task_col: str = None,
    task_target_item: Any = None,
    bounds_norm: Optional[Tensor] = None,
    multi_model_type: Optional[str] = None,
    multi_task_type: Optional[str] = None,
    cat_target_items: Optional[Dict[str, Sequence[Any]]] = None,
    impute: bool = False,
    robust: bool = False,
    perturbation: bool = False,
    heteroscedastic: bool = False,
    is_high_dim: bool = False,
    deep_gp: bool = False,
    deep_kernel: bool = False,
    alpha: float = 1e-1,
    lr: float = 1e-2,
    epoch: int = 300,
    n_splits: int = 5,
    random_state: int = 0,
):
    """KFold Cross-Validation を実施（非破壊・毎 fold 新規学習）

    Args:
        df: 全データ
        feature_cols: 特徴量列
        target_cols: 目的列
        categorical_cols: カテゴリ特徴量列
        cat_targets_cols: 2値化対象の目的列
        df_base: multi-task 用のベースデータ
        task_col: タスク識別用列
        task_target_item: タスク対象値
        bounds_norm: 正規化境界（共有するなら指定）
        multi_model_type: 多目的時のモデル構成
        multi_task_type: multi-task モデル構成
        cat_target_items: 2値化 {列名: 正例値リスト}
        impute: 欠損補完フラグ
        is_high_dim: 高次元フラグ
        deep_gp: Deep GP を使う
        deep_kernel: DKL を使う
        alpha: DKL 正則化
        lr: 学習率
        epoch: エポック数
        n_splits: KFold 分割数
        random_state: 乱数シード
    """
    score_train: Dict[str, pd.DataFrame] = {}
    score_test: Dict[str, pd.DataFrame] = {}
    mean_trains: List[pd.DataFrame] = []
    std_trains: List[pd.DataFrame] = []
    mean_tests: List[pd.DataFrame] = []
    std_tests: List[pd.DataFrame] = []
    cv_scores: Dict[str, pd.DataFrame] = {}

    feature_cols = list(feature_cols)
    target_cols = list(target_cols)
    categorical_cols = list(categorical_cols or [])
    cat_targets_cols = list(cat_targets_cols or [])
    cat_target_items = cat_target_items or {}

    # fit に渡す特徴量と、CV 用に保持する列（task_col があれば追加で保持）
    cols_for_fit = list(feature_cols)
    if task_col is not None and task_col not in cols_for_fit:
        cols_for_fit.append(task_col)

    df_use = df[cols_for_fit + target_cols].copy()
    if impute:
        # 目的変数が欠損している行だけ除外（特徴量の欠損は fold 内で補完）
        df_use = df_use.dropna(subset=target_cols)
    else:
        df_use = df_use.dropna(subset=cols_for_fit + target_cols)

    CV = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    for train_index, test_index in CV.split(df_use):
        df_train = df_use.iloc[train_index].copy()
        df_test = df_use.iloc[test_index].copy()

        model = BOCandidateModel()
        model.fit(
            df=df_train,
            feature_cols=feature_cols,
            target_cols=target_cols,
            categorical_cols=categorical_cols,
            cat_targets_cols=cat_targets_cols,
            df_base=df_base,
            task_col=task_col,
            task_target_item=task_target_item,
            bounds_norm=bounds_norm,
            multi_model_type=multi_model_type,
            multi_task_type=multi_task_type,
            cat_target_items=cat_target_items,
            impute=impute,
            robust=robust,
            perturbation=perturbation,
            heteroscedastic=heteroscedastic,
            is_high_dim=is_high_dim,
            deep_gp=deep_gp,
            deep_kernel=deep_kernel,
            alpha=alpha,
            lr=lr,
            epoch=epoch,
        )

        # スコア計算
        _score_train = model.score()
        _score_test = model.score(df_test[feature_cols], df_test[target_cols])

        # 予測値保存（index を fold 全体で識別できるように保持）
        _mean_train, _std_train = model.predict(df_train[feature_cols])
        _mean_test, _std_test = model.predict(df_test[feature_cols])
        _mean_train["index"] = train_index
        _std_train["index"] = train_index
        _mean_test["index"] = test_index
        _std_test["index"] = test_index

        for target in target_cols:
            score_train[target] = (
                pd.concat([score_train[target], _score_train[target]])
                if target in score_train
                else _score_train[target]
            )
            score_test[target] = (
                pd.concat([score_test[target], _score_test[target]])
                if target in score_test
                else _score_test[target]
            )

        mean_trains.append(_mean_train)
        std_trains.append(_std_train)
        mean_tests.append(_mean_test)
        std_tests.append(_std_test)

    # fold ごとのスコアを整理
    for target in target_cols:
        score_train[target] = _mean_score(score_train, target, cv_type="train")
        score_test[target] = _mean_score(score_test, target, cv_type="test")
        cv_scores[target] = pd.concat(
            [score_train[target], score_test[target]],
            axis=0,
        )

    return {
        "cv_score": cv_scores,
        "mean_train_cv": _concat(mean_trains),
        "std_train_cv": _concat(std_trains),
        "mean_test_cv": _concat(mean_tests),
        "std_test_cv": _concat(std_tests),
    }