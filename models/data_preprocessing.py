import torch
from torch import Tensor
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union, Any, Dict, Sequence

from botorch.utils.transforms import normalize
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from copy import deepcopy
from collections import defaultdict
import itertools

def make_permutation(
    src: Sequence[str],
    dst: Sequence[str],
    *,
    strict: bool = True,
) -> List[int]:
    """
    src を dst の順に並べ替えるためのインデックス置換 perm を作る。

    perm = [src.index(dst[0]), src.index(dst[1]), ...] なので、
    new = [x[i] for i in perm] が dst 順になる。

    strict=True:
        - dst に src に無い要素があればエラー
        - src に dst に無い要素があってもエラー（=完全一致を要求）
    strict=False:
        - dst に src に無い要素があればエラー
        - src の余り要素は無視（落ちる）
    """
    pos = {k: i for i, k in enumerate(src)}
    missing = [k for k in dst if k not in pos]
    if missing:
        raise KeyError(f"dst has unknown keys: {missing}")

    perm = [pos[k] for k in dst]

    if strict:
        extra = [k for k in src if k not in set(dst)]
        if extra:
            raise ValueError(f"src has extra keys not in dst: {extra}")
    return perm

def apply_permutation(seq: Sequence, perm: Sequence[int]) -> List:
    return [seq[i] for i in perm]
    
def num_and_cat_idx(
    feature_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    特徴量列とカテゴリカル列から、数値列とカテゴリカル列のインデックスを抽出する関数。

    Args:
        feature_cols (List[str]): 全特徴量列のリスト。
        categorical_cols (List[str]): カテゴリカル列のリスト。

    Returns:
        Tuple[List[str], List[str], List[int], List[int]]:
            - feature_cols (List[str]): 特徴量列のリスト（数値列＋カテゴリカル列の順）。
            - numeric_cols (List[str]): 数値列のリスト。
            - numeric_idx (List[int]): 数値列のインデックスリスト。
            - categorical_idx (List[int]): カテゴリカル列のインデックスリスト。
    """
    # 数値列 = 特徴量列からカテゴリカル列を除いたもの
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    # 特徴量列を「数値列＋カテゴリカル列」の順に並べ替える
    feature_cols_reordered = numeric_cols + categorical_cols

    # 各列のインデックスを取得
    numeric_idx = [feature_cols_reordered.index(col) for col in numeric_cols]
    categorical_idx = [feature_cols_reordered.index(col) for col in categorical_cols]

    return feature_cols_reordered, numeric_cols, numeric_idx, categorical_idx

def impute_iterative_mice(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    max_iter: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    IterativeImputer(MICE) による欠損補完ユーティリティ。

    - カテゴリ列は OrdinalEncoder で一時的に数値化
    - 補完後に四捨五入＆カテゴリ数でクリップ → 逆変換で元に戻す
    - 現状は 1 回分の imputed DataFrame を返す（多重代入が必要なら外側でループ）

    Args:
        df: 入力 DataFrame。
        cat_cols: カテゴリ列名。None の場合は object/category 型から自動検出。
        max_iter: IterativeImputer の最大反復回数。
        random_state: 乱数シード。

    Returns:
        欠損補完後の DataFrame。
    """
    df_in = df.copy()

    # カテゴリ列の自動検出
    if cat_cols is None:
        cat_cols = list(df_in.select_dtypes(include=["object", "category"]).columns)

    num_cols = [c for c in df_in.columns if c not in cat_cols]

    enc = OrdinalEncoder()
    if cat_cols:
        df_in[cat_cols] = enc.fit_transform(df_in[cat_cols])

    # Imputer 用に float に寄せる
    df_in = df_in.astype(float)

    imp = IterativeImputer(
        max_iter=max_iter,
        sample_posterior=True,  # 多重代入したければ random_state を変えて何回か回す
        random_state=random_state,
    )
    arr_imp = imp.fit_transform(df_in)
    df_imp = pd.DataFrame(arr_imp, columns=df_in.columns, index=df.index)

    # カテゴリ列を安全に戻す（丸め→クリップ→逆変換）
    if cat_cols:
        for i, col in enumerate(cat_cols):
            n_cat = len(enc.categories_[i])
            codes = np.rint(df_imp[col].values).astype(int)
            codes = np.clip(codes, 0, n_cat - 1)
            df_imp[col] = codes

        df_imp[cat_cols] = enc.inverse_transform(df_imp[cat_cols].values)
        for col in cat_cols:
            df_imp[col] = df_imp[col].astype("object")

    return df_imp

def prepare_categorical_info(
    X: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[Optional[List[Dict[int, int]]], Dict[str, Dict[Any, int]]]:
    """
    カテゴリ列を整数ラベル化し、BoTorch 用の fixed_features_list と
    ラベル→インデックスの辞書を作成する。

    Args:
        X: 入力 DataFrame。
        categorical_cols: カテゴリ列名リスト。

    Returns:
        fixed_features_list:
            - 各カテゴリ値の全組み合わせ [{col_idx: val, ...}, ...]
            - カテゴリ列が無い場合は None
        labels:
            - {col_name: {label -> index}} の辞書
    """
    if categorical_cols is None or len(categorical_cols) == 0:
        return None, {}, None

    fixed_features: List[List[Dict[int, int]]] = []
    labels: Dict[str, Dict[Any, int]] = {}
    categorical_features: Dict[int, Sequence[float]] = {}

    col_to_idx = {c: i for i, c in enumerate(X.columns)}

    for col in categorical_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(X[col])
        codes = np.unique(encoded)

        # BoTorch の fixed_features 用: {列インデックス: 整数コード}
        fixed_features.append([{col_to_idx[col]: int(code)} for code in codes])

        # ラベル -> インデックス の辞書
        label2idx: Dict[Any, int] = {
            label: int(code) for label, code in zip(le.classes_.tolist(), codes)
        }
        labels[col] = label2idx
        categorical_features[col_to_idx[col]] = list(np.unique(encoded))

    # 全組み合わせを生成
    fixed_features_list: Optional[List[Dict[int, int]]]
    if fixed_features:
        fixed_features_list = fixed_features[0]
        for feats in fixed_features[1:]:
            product = itertools.product(fixed_features_list, feats)
            fixed_features_list = [{**d1, **d2} for d1, d2 in product]
    else:
        fixed_features_list = None

    return fixed_features_list, labels, categorical_features

def preprocess_X(
    X: Union[pd.DataFrame, np.ndarray],
    fixed_features_list: Optional[List[Dict[int, int]]],
    cat2idx_list: Dict[str, Dict[Any, int]],
    dtype: torch.dtype,
) -> Tensor:
    """
    DataFrame / ndarray を Tensor に変換しつつ、カテゴリ列を
    {label -> index} の辞書に基づいて数値化する。

    Args:
        X: 特徴量 (DataFrame または ndarray)。
        fixed_features_list: 固定特徴リスト（本関数内では未使用だが、インターフェース互換のため受け取る）。
        cat2idx_list: {列名: {label -> index}}。
        dtype: 出力 Tensor の dtype。

    Returns:
        Tensor: shape (n, d)
    """
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    elif isinstance(X, np.ndarray):
        df = pd.DataFrame(X)
    else:
        raise TypeError("X は DataFrame または ndarray である必要があります。")

    # カテゴリ列を index に置き換え
    for col, mapping in cat2idx_list.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(float)

    return torch.tensor(df.values, dtype=dtype)

def normalize_bounds(
    X: Tensor,
    # num_idx: Optional[List[int]] = None,
    bounds = None,
    dtype: torch.dtype = torch.double
) -> Tensor:
    """
    train_X から [min, max] 境界を作成し、必要に応じて指定 bounds で
    数値次元のみ上書きする。
    """
    if bounds is None:
        bounds = torch.cat(
            [
                X.min(dim=0, keepdim=True)[0],
                X.max(dim=0, keepdim=True)[0],
            ],
            dim=0,
        ).to(dtype)
    return bounds

def search_bounds(
    train_X: Tensor,
    bounds: Optional[Tensor] = None,
    bounds_not_neg: Optional[List[bool]] = None,
    extra_rate: float = 0.1,
) -> Tensor:
    """
    検索境界を作成します（train_X はすでに正規化空間を想定）。

    Args:
        train_X: 訓練データの特徴量 (q, d)。
        bounds: 既存の境界 (2, d)。指定時は extra_rate を足し引き。
        bounds_not_neg: 各次元が非負制約かどうかのフラグ。
        extra_rate: 下限/上限に足し引きする幅。

    Returns:
        Tensor: shape (2, d) の境界。
    """
    d = train_X.shape[1]
    
    bounds_not_neg = list(bounds_not_neg or [])
    if bounds_not_neg and len(_bounds_not_neg) != d:
        raise ValueError(
            f"bounds_not_neg({len(bounds_not_neg)}) と train_X の次元({d}) が一致しません。"
        )
    
    if bounds is not None:
        lower = bounds[0:1, :] - extra_rate
        upper = bounds[1:2, :] + extra_rate
    else:
        lower = train_X.min(dim=0, keepdim=True)[0] - extra_rate
        upper = train_X.max(dim=0, keepdim=True)[0] + extra_rate

    out = torch.cat([lower, upper], dim=0)
    
    if bounds_not_neg is not None:
        for i, flag in enumerate(bounds_not_neg):
            if flag:
                # 正規化空間なので 0.0 で OK
                out[0, i] = torch.clamp(out[0, i], min=0.0)

    return out

def round_to_grid_multi_constraints(
    X: Union[pd.DataFrame, np.ndarray],
    step: Tensor,
    base: Tensor,
    bounds: Tensor,
    constraint_indices: List[List[int]],
    constraint_coeffs: List[List[float]],
    constraint_targets: List[float],
    constraint_ops: List[str],
    max_iter: int = 5,
    tol: float = 1e-5,
) -> Tensor:
    """
    丸め＋複数の線形制約を満たすよう調整（制約への関与が少ない変数を優先）。

    Args:
        X: shape (q, d) の候補点 (DataFrame / ndarray)。
        step: shape (d,) の丸め刻み。
        base: shape (d,) の基準点。
        bounds: shape (2, d) の境界。
        constraint_indices: 各制約の変数インデックス群。
        constraint_coeffs: 各制約の係数。
        constraint_targets: 各制約の目標値。
        constraint_ops: 各制約の演算子（現状 '='/'==' のみ調整対象）。
    """
    if isinstance(X, pd.DataFrame):
        arr = X.values
    elif isinstance(X, np.ndarray):
        arr = X
    else:
        raise TypeError("X は DataFrame か ndarray である必要があります。")

    x_tensor = torch.as_tensor(arr, dtype=base.dtype)
    q, d = x_tensor.shape

    step = step.to(dtype=base.dtype)
    base = base.to(dtype=base.dtype)
    bounds = bounds.to(dtype=base.dtype)

    rounded = torch.round((x_tensor - base) / step) * step + base
    rounded = torch.clamp(rounded, bounds[0], bounds[1])

    # 各変数が何個の制約に関与しているかを計算
    constraint_counts = torch.zeros(d, dtype=torch.long)
    for idxs in constraint_indices:
        for i in idxs:
            constraint_counts[i] += 1

    for b in range(q):  # 各候補点ごと
        x = rounded[b].clone()

        for _ in range(max_iter):  # 調整ループ
            all_satisfied = True
            for idxs, coeffs, target, op in zip(
                constraint_indices, constraint_coeffs, constraint_targets, constraint_ops
            ):
                # 現状は equality のみを調整対象とする
                if op not in ("=", "=="):
                    continue

                idxs_tensor = torch.tensor(idxs, dtype=torch.long)
                coeffs_tensor = torch.tensor(coeffs, dtype=base.dtype)

                current_val = torch.sum(coeffs_tensor * x[idxs_tensor])
                diff = target - current_val.item()

                if abs(diff) < tol:
                    continue  # この制約は満たされている

                all_satisfied = False

                # 調整候補変数をスコアリング（関与数が少なく、ステップ×係数が小さいもの）
                scores = torch.stack(
                    [
                        constraint_counts[i].float() * torch.abs(coeff * step[i])
                        for i, coeff in zip(idxs, coeffs)
                    ]
                )
                best_idx_in_idxs = torch.argmin(scores)
                adjust_idx = idxs[best_idx_in_idxs]
                coeff = coeffs[best_idx_in_idxs]

                # 必要な差分を該当変数に適用（ステップ幅で丸めて）
                delta = diff / (coeff + 1e-9)
                delta = torch.round(delta / step[adjust_idx]) * step[adjust_idx]

                x[adjust_idx] += delta
                x[adjust_idx] = torch.clamp(
                    x[adjust_idx], bounds[0, adjust_idx], bounds[1, adjust_idx]
                )

            if all_satisfied:
                break  # 全制約を満たしたので終了

        rounded[b] = x

    return rounded


def _add_index(
    df: pd.DataFrame,
    task_idx: int = 0,
) -> pd.DataFrame:
    """単一タスクインデックス列 'task' を追加して返す（元 df はコピー）。"""
    df_out = df.copy()
    df_out["task"] = task_idx
    return df_out


def _add_index_from_target(
    df: pd.DataFrame,
    task_col: str,
    task_target_item: Any,
) -> pd.DataFrame:
    """
    task_col == task_target_item を 1、それ以外を 0 とする 'task' 列を追加。
    元の df はコピーされる。
    """
    df_out = df.copy()
    df_out["task"] = (df_out[task_col] == task_target_item).astype(int)
    return df_out


def make_multitask_data(
    df_base: pd.DataFrame,
    df_main: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    task_col: Optional[str] = None,
    task_target_item: Optional[Any] = None,
) -> pd.DataFrame:
    """
    multi-task GP 用に df_base と df_main を結合し、最後に 'task' 列を追加する。

    - task_col / task_target_item が指定されていないか、列が存在しない場合:
        df_base -> task=0, df_main -> task=1
    - 指定されている場合:
        両方の DataFrame で task_col==task_target_item を 1、それ以外を 0 とする。
        このとき元の task_col は削除し、'task' のみを特徴量として使う想定。
    """
    if (
        task_col is None
        or task_target_item is None
        or task_col not in df_base.columns
        or task_col not in df_main.columns
    ):
        df0 = _add_index(df_base[feature_cols + target_cols], task_idx=0)
        df1 = _add_index(df_main[feature_cols + target_cols], task_idx=1)
        df_all = pd.concat([df0, df1], ignore_index=True)
    else:
        cols = list(dict.fromkeys(feature_cols + target_cols + [task_col]))
        df0 = _add_index_from_target(df_base[cols], task_col, task_target_item)
        df1 = _add_index_from_target(df_main[cols], task_col, task_target_item)
        # task_col は features/targets には含めず、'task' のみ利用
        df0 = df0.drop(columns=[task_col])
        df1 = df1.drop(columns=[task_col])
        df_all = pd.concat([df0, df1], ignore_index=True)

    return df_all

def filter_cat_features_list(
    feature_cols,
    labels,
    cat_features_list: Any,
    cat_fixed_features: Dict[str, List[str]]
):
    """
    model.cat_features_list (list[dict[int, int|float]]) を cat_fixed_features に従って絞り込む。

    cat_fixed_features 例:
        {"cat_feature": ["a", "b"], "cat_feature2": ["x"]}

    前提:
        - model.feature_cols: List[str]
        - model.labels: Dict[str, Dict[str, int]]  # カテゴリ名 -> エンコード値
        - model.cat_features_list: List[Dict[int, int|float]]  # index -> value の組合せ列挙
    """
    # 何も指定されていないなら、そのまま
    if not cat_fixed_features:
        return cat_features_list

    # feature名 -> (特徴量index -> 許容エンコード値set) に変換
    fixed_idx_vals: Dict[int, set] = {}
    for feat_name, allowed_names in cat_fixed_features.items():
        if feat_name not in feature_cols:
            raise ValueError(f"cat_fixed_features に未知の特徴量名: {feat_name}")

        k = feature_cols.index(feat_name)

        if feat_name not in labels:
            raise ValueError(f"model.labels に {feat_name} がありません（カテゴリ特徴の定義不足）")

        try:
            allowed_encoded = {labels[feat_name][v] for v in allowed_names}
        except KeyError as e:
            raise ValueError(f"{feat_name} のカテゴリ値が不正です: {e}") from e

        # ここが空だと全滅するので、明示的に弾く（好みで警告でもOK）
        if not allowed_encoded:
            raise ValueError(f"{feat_name} の許容カテゴリが空です: {allowed_names}")

        fixed_idx_vals[k] = allowed_encoded

    # 絞り込み
    filtered = [
        d for d in cat_features_list
        if all(d.get(k) in allowed for k, allowed in fixed_idx_vals.items())
    ]
    return filtered