from typing import Callable, Dict, List, Optional, Sequence, Tuple, Literal
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

# ================================
# k-sparse transform
# ================================
def k_exact_sparse_transform_factory(
    comp_idx: Sequence[int],
    k: int,
    score: Literal["abs", "value"] = "abs",
    min_active: float = 0.0,  # “使う”を正にしたいなら >0（例: 1e-6）
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    comp_idx の中から必ず k 個を選び（サポートサイズ=k）、
    それ以外を 0 にする transform。sum-to-one はしない。

    X: (..., q, d)
    """
    comp_idx = list(comp_idx)

    def transform(X: torch.Tensor) -> torch.Tensor:
        if len(comp_idx) == 0:
            return X
        if k <= 0:
            # 厳密に0個を選ぶ：全部0
            Xn = X.clone()
            idx_t = torch.tensor(comp_idx, device=X.device, dtype=torch.long)
            Xn[..., :, idx_t] = 0.0
            return Xn

        device = X.device
        dtype = X.dtype
        idx_t = torch.tensor(comp_idx, device=device, dtype=torch.long)

        group = X[..., :, idx_t]  # (..., q, m)
        m = group.size(-1)
        if m < k:
            raise ValueError(f"exact-k requires m>=k, but m={m}, k={k}")

        s = group.abs() if score == "abs" else group
        topk_idx = s.topk(k, dim=-1).indices  # (..., q, k)

        mask = torch.zeros_like(group, dtype=torch.bool).scatter(-1, topk_idx, True)
        out = torch.where(mask, group, torch.zeros_like(group))

        if min_active > 0:
            # “使う”を正に寄せる（ただし等式和制約がある場合は後述の最終補正推奨）
            out = torch.where(mask, torch.clamp(out, min=min_active), out)

        X_new = X.clone()
        X_new[..., :, idx_t] = out
        return X_new

    return transform

# ================================
# 和=rhs の簡易 repair
# ================================
def sample_k_without_replacement(
    scores: torch.Tensor,  # (..., d)
    k: int,
    *,
    tau: float = 0.2,      # 温度: 小さいほど top-k に近い
    eps: float = 0.05,     # 一様混合: 低スコアにもチャンスを残す
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    scores から softmax 分布を作り、k 個の index を非復元抽出で返す。
    戻り値: (..., k) long
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    d = scores.size(-1)
    k_eff = min(k, d)

    # 数値安定化
    s = scores / max(tau, 1e-12)
    s = s - s.max(dim=-1, keepdim=True).values
    p = torch.softmax(s, dim=-1)

    if eps > 0:
        p = (1.0 - eps) * p + eps * (1.0 / d)

    # 非復元抽出（各行ごと）
    flat_p = p.reshape(-1, d)
    idx = torch.multinomial(flat_p, num_samples=k_eff, replacement=False, generator=generator)
    return idx.reshape(scores.shape[:-1] + (k_eff,))
    
def diversify_within_q(
    X: torch.Tensor,
    repair: Callable[[torch.Tensor], torch.Tensor],
    *,
    bounds: Optional[torch.Tensor] = None,
    tol: Optional[float] = None,
    step: Optional[float] = None,
    mode: Literal["deterministic", "random"] = "deterministic",
    frozen_idx: Sequence[int] = (),
    comp_idx: Sequence[int] = (),
    active_eps: float = 0.0,
    max_tries: int = 3,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    q-batch (..., q, d) の中で、同一点/近すぎる点を微小に散らしてから再 repair する。

    想定:
      - repair は任意の制約投影/丸め等の post-processing（あなたの make_k_sparse... の repair 等）
      - bounds を渡すと、tol/step のデフォルトがスケール依存で安定しやすい

    Args:
        X: (..., q, d) または (..., d)
        repair: Callable[[Tensor], Tensor]
        bounds: shape (2, d) あるいは (2, 1, d) 等（最後が d であればよい）
        tol: 近重複判定距離（未指定なら bounds のスケールから決定）
        step: 散らし量（未指定なら bounds のスケールから決定）
        mode:
            - "deterministic": 勾配最適化に載せても挙動が安定（推奨）
            - "random": 重複を強力に崩すが、最適化が非決定的になる（必要時のみ）
        frozen_idx: 固定したい次元（fixed_features 等）
        comp_idx: 任意。アクティブ次元推定の“優先候補”として使う
        active_eps: active 判定（abs(x) > active_eps）閾値
        max_tries: 散らし→repair を繰り返す回数
        generator: mode="random" で使う torch.Generator

    Returns:
        repair + diversification を適用した X（元の shape）
    """
    # まず一回 repair（これで feasibility を確保）
    Xr = repair(X)

    # (..., d) なら散らす意味がほぼ無いので終了
    if Xr.ndim < 2 or Xr.shape[-2] <= 1:
        return Xr

    d = Xr.shape[-1]
    q = Xr.shape[-2]
    orig_shape = Xr.shape
    Xb = Xr.reshape(-1, q, d)  # (B, q, d)
    B = Xb.shape[0]

    # bounds からスケールを決める（未指定の場合の安定策）
    if bounds is not None:
        lb = bounds[0].to(device=X.device, dtype=X.dtype).reshape(-1)[-d:]
        ub = bounds[1].to(device=X.device, dtype=X.dtype).reshape(-1)[-d:]
        span = (ub - lb).abs().clamp_min(torch.finfo(X.dtype).eps)
        # 距離スケール（平均レンジ）
        scale = span.mean().item()
        if tol is None:
            tol = 1e-12 * max(scale, 1.0)
        if step is None:
            step = 1e-6 * max(scale, 1.0)
    else:
        if tol is None:
            tol = 1e-12
        if step is None:
            step = 1e-6
        lb = ub = None

    # 散らしに使える次元候補（frozen は避ける）
    frozen = set(int(i) for i in frozen_idx)
    allowed_dims = [j for j in range(d) if j not in frozen]
    if len(allowed_dims) == 0:
        # 全次元固定ならどうにもならない
        return Xr

    # comp_idx が与えられている場合、まず comp_idx 内で active な次元を優先的に使う
    comp_list = [int(i) for i in comp_idx] if comp_idx else []

    eye = torch.eye(q, device=X.device, dtype=X.dtype).unsqueeze(0)  # (1,q,q)

    for t in range(max_tries):
        # 距離行列 (B,q,q)
        D = torch.cdist(Xb, Xb)
        D = D + eye * 1e9  # 自己距離を除外
        close = D < tol

        # i 行の「過去 j<i に close がある」= duplicate
        close_lower = torch.tril(close, diagonal=-1)
        dup = close_lower.any(dim=-1)  # (B, q)

        if not dup.any():
            out = Xb.reshape(orig_shape)
            return out  # 既に OK

        # duplicate な点だけ、微小に散らす
        # できるだけ 1 次元方向にだけ動かして “サポート破壊” を最小化
        delta = torch.zeros_like(Xb)

        for i in range(1, q):  # i=0 は基準点にしやすいので基本動かさない
            bi = dup[:, i]  # (B,)
            if not bi.any():
                continue

            # この点で “動かしやすい次元” を選ぶ
            # 1) comp_idx の中で active な次元（abs>active_eps）を優先
            dim_choice = None
            if comp_list:
                # (B, len(comp_list))
                vals = Xb[bi, i][:, comp_list].abs()
                active = vals > active_eps
                if active.any():
                    # 各サンプルで最初に active な次元
                    first_active = active.float().argmax(dim=-1)  # (nb,)
                    chosen = torch.tensor(comp_list, device=X.device)[first_active]  # (nb,)
                    # ただし frozen は避ける
                    # frozen が含まれる場合は後で allowed_dims にフォールバック
                    if all(int(c) not in frozen for c in chosen.tolist()):
                        dim_choice = chosen  # (nb,)

            if dim_choice is None:
                # 2) allowed_dims を i で回す（決定的）
                dim = allowed_dims[i % len(allowed_dims)]
                dim_choice = torch.full((int(bi.sum().item()),), dim, device=X.device, dtype=torch.long)

            nb = int(bi.sum().item())
            if mode == "random":
                g = generator
                if g is None:
                    g = torch.Generator(device=X.device)
                    g.manual_seed(0)
                step_vec = step * torch.randn((nb,), device=X.device, dtype=X.dtype, generator=g)
            else:
                # deterministic: i と try に応じて符号と大きさを変える
                sign = -1.0 if (i % 2 == 0) else 1.0
                step_vec = torch.full((nb,), sign * step * (1.0 + 0.25 * t), device=X.device, dtype=X.dtype)

            # delta[bi, i, dim_choice] += step_vec
            rows = torch.nonzero(bi, as_tuple=False).squeeze(-1)
            delta[rows, i, dim_choice] = step_vec

        Xb = Xb + delta

        # bounds clamp（あれば）
        if bounds is not None:
            Xb = torch.max(torch.min(Xb, ub), lb)

        # 再 repair（サポート選択・投影・fixed などを再適用）
        Xb = repair(Xb.reshape(orig_shape)).reshape(-1, q, d)

    return Xb.reshape(orig_shape)
    
def make_k_sparse_linear_constraints_repair(
    bounds: torch.Tensor,
    comp_idx: Sequence[int],
    k: int,
    score: Literal["abs", "value"] = "abs",
    equality_constraints: Optional[List[Tuple[List[int], List[float], float]]] = None,
    inequality_constraints: Optional[List[Tuple[List[int], List[float], float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    max_iters: int = 12,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    1) comp_idx の中で top-k を選び、それ以外を 0 に固定（<=k）
    2) そのサポートを絶対に崩さない（使わない成分は更新しない）形で
       等式/不等式を反復投影して repair
    3) bounds 内に clamp、fixed_features も維持

    想定 X: (..., q, d) でも (..., d) でも可
    """
    comp_idx = list(comp_idx)
    equality_constraints = equality_constraints or []
    inequality_constraints = inequality_constraints or []
    fixed_features = fixed_features or {}

    d = bounds.size(-1)

    def repair(X: torch.Tensor) -> torch.Tensor:
        orig_shape = X.shape
        if orig_shape[-1] != d:
            raise ValueError(f"Last dim {orig_shape[-1]} != d={d}")

        device, dtype = X.device, X.dtype
        lower = bounds[0].to(device=device, dtype=dtype)
        upper = bounds[1].to(device=device, dtype=dtype)

        # (N, d) にフラット化（q があっても独立点として扱う）
        Xf = X.reshape(-1, d)

        # ---- (A) まず <=k のサポートを決めて固定 ----
        if len(comp_idx) > 0 and k > 0:
            idx_t = torch.tensor(comp_idx, device=device, dtype=torch.long)
            group = Xf[:, idx_t]               # (N, m)
            m = group.size(-1)
            k_eff = min(k, m)

            s = group.abs() if score == "abs" else group
            # s: (N, g_size) みたいな “グループ内スコア”
            idx = sample_k_without_replacement(
                scores=s,
                k=k_eff,
                tau=0.2,
                eps=0.05,
                generator=None,
            )  # (N, k)
            
            mask_g = torch.zeros_like(s, dtype=torch.bool)
            mask_g.scatter_(dim=-1, index=idx, value=True)
            
            group_sparse = torch.where(mask_g, group, torch.zeros_like(group))
            Xf[:, idx_t] = group_sparse
        else:
            idx_t = None
            mask_g = None  # unused

        # fixed_features を強制
        if fixed_features:
            for j, v in fixed_features.items():
                Xf[:, j] = torch.as_tensor(v, device=device, dtype=dtype)

        # ---- (B) サポート固定のまま、反復投影（近似） ----
        # 係数ベクトル a を「更新可能次元」だけ残した a_allowed にするのが肝
        # 更新不可: fixed_features と comp_idx の inactive 成分
        fixed_idx = torch.tensor(list(fixed_features.keys()), device=device, dtype=torch.long) if fixed_features else None

        for _ in range(max_iters):
            # (1) bounds clamp
            Xf = torch.max(torch.min(Xf, upper), lower)

            # (2) fixed_features 再強制
            if fixed_features:
                for j, v in fixed_features.items():
                    Xf[:, j] = torch.as_tensor(v, device=device, dtype=dtype)

            # (3) comp_idx の inactive を 0 に戻す（サポート固定）
            if idx_t is not None:
                # Xf[:, idx_t] はすでに sparse だが、念のため維持
                cur = Xf[:, idx_t]
                cur = torch.where(mask_g, cur, torch.zeros_like(cur))
                Xf[:, idx_t] = cur

            # (4) equality: a^T x = b を「許可次元のみ」で1本ずつ投影
            for idxs, coeffs, rhs in equality_constraints:
                if not isinstance(idxs, torch.Tensor):
                    idxs = torch.tensor(idxs, device=device)
                if not isinstance(coeffs, torch.Tensor):
                    coeffs = torch.tensor(coeffs, device=device)
                a = torch.zeros(d, device=device, dtype=dtype)
                # a[idxs] = torch.tensor(coeffs, device=device, dtype=dtype)
                a[idxs] = coeffs

                a_allowed = a.expand(Xf.size(0), -1).clone()

                # fixed は更新しない
                if fixed_idx is not None and fixed_idx.numel() > 0:
                    a_allowed[:, fixed_idx] = 0.0

                # comp_idx inactive は更新しない
                if idx_t is not None:
                    # comp_idx 部分だけ mask を掛ける
                    a_allowed[:, idx_t] = a_allowed[:, idx_t] * mask_g.to(dtype)

                # もし許可方向がゼロならスキップ（このサポートでは満たせない）
                norm2 = (a_allowed * a_allowed).sum(dim=1).clamp_min(1e-12)
                has_dir = norm2 > 1e-11
                if not has_dir.any():
                    continue

                # residual = b - a^T x（a^T x は元の a で評価）
                resid = rhs - (Xf * a).sum(dim=1)  # (N,)
                step = (resid / norm2).unsqueeze(1) * a_allowed
                Xf = Xf + step

            # (5) inequality: a^T x <= b を破っているものだけ halfspace 投影
            for idxs, coeffs, rhs in inequality_constraints:
                if not isinstance(idxs, torch.Tensor):
                    idxs = torch.tensor(idxs, device=device)
                a = torch.zeros(d, device=device, dtype=dtype)
                a[idxs] = torch.tensor(coeffs, device=device, dtype=dtype)

                lhs = (Xf * a).sum(dim=1)  # (N,)
                viol = lhs - rhs
                bad = viol > 0
                if not bad.any():
                    continue

                a_allowed = a.expand(Xf.size(0), -1).clone()
                if fixed_idx is not None and fixed_idx.numel() > 0:
                    a_allowed[:, fixed_idx] = 0.0
                if idx_t is not None:
                    a_allowed[:, idx_t] = a_allowed[:, idx_t] * mask_g.to(dtype)

                norm2 = (a_allowed * a_allowed).sum(dim=1).clamp_min(1e-12)
                # 許可方向が無い個体は直せないのでスキップ
                can = bad & (norm2 > 1e-11)
                if not can.any():
                    continue

                t = (viol[can] / norm2[can]).unsqueeze(1)
                Xf[can] = Xf[can] - t * a_allowed[can]

        # 最終 clamp + 固定
        Xf = torch.max(torch.min(Xf, upper), lower)
        if fixed_features:
            for j, v in fixed_features.items():
                Xf[:, j] = torch.as_tensor(v, device=device, dtype=dtype)
        if idx_t is not None:
            Xf[:, idx_t] = torch.where(mask_g, Xf[:, idx_t], torch.zeros_like(Xf[:, idx_t]))

        return Xf.reshape(orig_shape)

    return repair

# ================================
# 共通ヘルパー: X_pending セット
# ================================

def _set_X_pending_on_acqf(
    acq_function: AcquisitionFunction,
    X_pending: Optional[torch.Tensor],
) -> None:
    """acq_function に X_pending をセット（あれば）"""
    if hasattr(acq_function, "set_X_pending"):
        acq_function.set_X_pending(X_pending)
    elif hasattr(acq_function, "X_pending"):
        acq_function.X_pending = X_pending


# ================================
# 共通ヘルパー: decode & evaluate_population
# ================================

def _build_decode_and_evaluate(
    acq_function: AcquisitionFunction,
    bounds: torch.Tensor,
    q: int,
    fixed_features: Optional[Dict[int, float]],
    inequality_constraints: Optional[List[Tuple[List[int], List[float], float]]],
    equality_constraints: Optional[List[Tuple[List[int], List[float], float]]],
    candidate_transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
    post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]],
    penalty_factor: float,
):
    """
    GA / PSO / SA / CMA-ES 共通の decode, evaluate_population を構築。
    """

    device = bounds.device
    dtype = bounds.dtype

    d_total = bounds.size(-1)
    fixed_features = fixed_features or {}
    free_idx = [i for i in range(d_total) if i not in fixed_features]
    d_free = len(free_idx)
    if d_free == 0:
        raise ValueError("All features are fixed; nothing to optimize.")

    def decode(z: torch.Tensor) -> torch.Tensor:
        """
        z: (..., q, d_free) in [0,1]
        -> X: (..., q, d_total) in bounds
        """
        z_clamped = z.clamp(0.0, 1.0)

        lower = bounds[0, free_idx].to(device=device, dtype=dtype)
        upper = bounds[1, free_idx].to(device=device, dtype=dtype)
        width = upper - lower

        # ブロードキャストで線形写像
        x_free = lower + z_clamped * width  # (..., q, d_free)

        # full X を組み立て
        *batch_shape, q_local, _ = z_clamped.shape
        X = torch.empty(*batch_shape, q_local, d_total, device=device, dtype=dtype)
        X[..., :, free_idx] = x_free
        for i, v in fixed_features.items():
            X[..., :, i] = float(v)

        if candidate_transform is not None:
            X = candidate_transform(X)

        return X

    def _linear_constraints_penalty(X: torch.Tensor) -> torch.Tensor:
        """
        線形制約 (BoTorch 形式) をペナルティに変換。
        X: (N, q, d) 想定。返り値: (N,)
        """
        if inequality_constraints is None and equality_constraints is None:
            return torch.zeros(X.shape[0], device=device, dtype=dtype)

        penalty = torch.zeros(X.shape[0], device=device, dtype=dtype)

        # 不等式: A x <= b
        if inequality_constraints is not None:
            for idxs, coeffs, rhs in inequality_constraints:
                idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
                coeffs_t = torch.tensor(coeffs, device=device, dtype=dtype)
                # (..., q, len(idxs))
                vals = (X[..., :, idx_t] * coeffs_t).sum(dim=-1)  # (N, q)
                lhs = vals.max(dim=-1).values  # (N,)
                viol = torch.clamp(lhs - rhs, min=0.0)
                penalty = penalty + viol

        # 等式: A x = b
        if equality_constraints is not None:
            for idxs, coeffs, rhs in equality_constraints:
                if not isinstance(idxs, torch.Tensor):
                    idx_t = torch.tensor(idxs, device=device).to(torch.long)
                else:
                    idx_t = idxs.to(torch.long)
                if not isinstance(coeffs, torch.Tensor):
                    coeffs_t = torch.tensor(coeffs, device=device).to(torch.long)
                else:
                    coeffs_t = coeffs.to(torch.long)
                vals = (X[..., :, idx_t] * coeffs_t).sum(dim=-1)  # (N, q)
    
                # 修正前:
                # lhs = vals.mean(dim=-1)  # (N,)
                # viol = torch.abs(lhs - rhs)
    
                # 修正後: q 次元で最大違反（全点で満たすを要求）
                viol = (vals - rhs).abs().max(dim=-1).values  # (N,)
    
                penalty = penalty + viol

        return penalty_factor * penalty

    def evaluate_population(pop: torch.Tensor) -> torch.Tensor:
        """
        pop: (N, q, d_free) in [0,1]
        -> fitness: (N,)  （acq - penalty）
        """
        # decode -> 必要なら post_processing -> acq
        X = decode(pop)  # (N, q, d_total)
        if post_processing_func is not None:
            X_proc = post_processing_func(X)
        else:
            X_proc = X

        with torch.no_grad():
            vals = acq_function(X_proc)

        # 形状を (N,) に揃える
        if vals.ndim == 0:
            vals = vals.reshape(1).repeat(pop.shape[0])
        elif vals.ndim == 1:
            vals = vals
        else:
            # 1 次元目以降を平均してつぶす（特殊な acqf 向けの保険）
            dims = tuple(range(1, vals.ndim))
            vals = vals.mean(dim=dims)

        penalty = _linear_constraints_penalty(X_proc)
        fitness = vals - penalty
        fitness[torch.isnan(fitness)] = -float("inf")
        return fitness  # (N,)

    return free_idx, decode, evaluate_population


# ================================
# GA backend
# ================================

def _optimize_acqf_ga_core(
    evaluate_population: Callable[[torch.Tensor], torch.Tensor],
    q: int,
    d_free: int,
    bounds: torch.Tensor,
    batch_initial_conditions: Optional[torch.Tensor],
    options: Dict,
):
    device = bounds.device
    dtype = bounds.dtype

    pop_size_opt = int(options.get("pop_size", 64))
    num_generations = int(options.get("num_generations", 100))
    elite_frac = float(options.get("elite_frac", 0.1))
    mutation_prob = float(options.get("mutation_prob", 0.1))
    mutation_std = float(options.get("mutation_std", 0.1))

    # 初期集団
    if batch_initial_conditions is not None:
        pop = batch_initial_conditions.to(device=device, dtype=dtype)
        if pop.ndim != 3 or pop.shape[-1] != d_free or pop.shape[-2] != q:
            raise ValueError(f"batch_initial_conditions must have shape (N, q={q}, d_free={d_free}).")
        pop_size = pop.shape[0]
    else:
        pop_size = pop_size_opt
        pop = torch.rand(pop_size, q, d_free, device=device, dtype=dtype)

    n_elite = max(1, int(pop_size * elite_frac))

    best_z = None
    best_val = None  # tensor scalar

    for _ in range(num_generations):
        fitness = evaluate_population(pop)  # (pop_size,)

        gen_best_val, gen_best_idx = torch.max(fitness, dim=0)
        if best_val is None or gen_best_val > best_val:
            best_val = gen_best_val.detach().clone()
            best_z = pop[gen_best_idx].detach().clone()  # (q, d_free)

        elite_idx = torch.topk(fitness, n_elite).indices
        elite = pop[elite_idx]

        # トーナメント選択
        def tournament_select(k: int, tour_size: int = 3) -> torch.Tensor:
            idx = torch.randint(0, pop_size, (k, tour_size), device=device)
            cand_fit = fitness[idx]
            best_idx_in_tour = cand_fit.argmax(dim=1)
            selected_idx = idx[torch.arange(k, device=device), best_idx_in_tour]
            return pop[selected_idx]

        n_offspring = pop_size - n_elite
        parents = tournament_select(n_offspring * 2)
        parents = parents.view(n_offspring, 2, q, d_free)

        # 一様交叉
        mask = torch.rand(n_offspring, q, d_free, device=device, dtype=dtype) < 0.5
        offspring = torch.where(mask.unsqueeze(1), parents[:, 0:1], parents[:, 1:2]).squeeze(1)

        # 突然変異
        mut_mask = torch.rand_like(offspring) < mutation_prob
        noise = torch.randn_like(offspring) * mutation_std
        offspring = offspring + mut_mask * noise
        offspring = offspring.clamp(0.0, 1.0)

        pop = torch.cat([elite, offspring], dim=0)

    if best_z is None:
        # 万が一 fitness が全部 -inf で初期化されなかった場合は適当に返す
        best_z = pop[0].detach().clone()
        best_val = evaluate_population(pop[:1])[0].detach().clone()

    return best_z, best_val.view(1)


# ================================
# PSO backend
# ================================

def _optimize_acqf_pso_core(
    evaluate_population: Callable[[torch.Tensor], torch.Tensor],
    q: int,
    d_free: int,
    bounds: torch.Tensor,
    batch_initial_conditions: Optional[torch.Tensor],
    options: Dict,
):
    device = bounds.device
    dtype = bounds.dtype

    swarm_size_opt = int(options.get("swarm_size", 64))
    num_iterations = int(options.get("num_iterations", 100))
    w = float(options.get("inertia", 0.7))
    c1 = float(options.get("c1", 1.5))
    c2 = float(options.get("c2", 1.5))

    # 初期位置・速度
    if batch_initial_conditions is not None:
        pos = batch_initial_conditions.to(device=device, dtype=dtype)
        if pos.ndim != 3 or pos.shape[-1] != d_free or pos.shape[-2] != q:
            raise ValueError(f"batch_initial_conditions must have shape (N, q={q}, d_free={d_free}).")
        swarm_size = pos.shape[0]
    else:
        swarm_size = swarm_size_opt
        pos = torch.rand(swarm_size, q, d_free, device=device, dtype=dtype)

    vel = torch.zeros_like(pos)

    fitness = evaluate_population(pos)  # (swarm_size,)
    pbest_pos = pos.clone()
    pbest_fit = fitness.clone()

    gbest_fit, gbest_idx = torch.max(fitness, dim=0)
    gbest_pos = pos[gbest_idx].clone()  # (q, d_free)

    for _ in range(num_iterations):
        r1 = torch.rand_like(pos)
        r2 = torch.rand_like(pos)

        vel = (
            w * vel
            + c1 * r1 * (pbest_pos - pos)
            + c2 * r2 * (gbest_pos.unsqueeze(0) - pos)
        )
        pos = pos + vel
        pos = pos.clamp(0.0, 1.0)

        fitness = evaluate_population(pos)

        better_mask = fitness > pbest_fit
        pbest_pos[better_mask] = pos[better_mask]
        pbest_fit[better_mask] = fitness[better_mask]

        cur_best_fit, cur_best_idx = torch.max(fitness, dim=0)
        if cur_best_fit > gbest_fit:
            gbest_fit = cur_best_fit
            gbest_pos = pos[cur_best_idx].clone()

    best_z = gbest_pos.detach().clone()
    best_val = gbest_fit.detach().clone().view(1)
    return best_z, best_val


# ================================
# SA backend (Simulated Annealing)
# ================================

def _optimize_acqf_sa_core(
    evaluate_population: Callable[[torch.Tensor], torch.Tensor],
    q: int,
    d_free: int,
    bounds: torch.Tensor,
    batch_initial_conditions: Optional[torch.Tensor],
    options: Dict,
):
    device = bounds.device
    dtype = bounds.dtype

    n_steps = int(options.get("sa_steps", 500))
    init_temp = float(options.get("sa_init_temp", 1.0))
    final_temp = float(options.get("sa_final_temp", 1e-2))
    step_size = float(options.get("sa_step_size", 0.1))
    use_log_schedule = bool(options.get("sa_log_schedule", False))

    # 初期点 z_cur
    if batch_initial_conditions is not None:
        z0 = batch_initial_conditions.to(device=device, dtype=dtype)
        if z0.ndim == 2:
            z0 = z0.unsqueeze(0)  # (1, q, d_free)
        if z0.ndim != 3 or z0.shape[-1] != d_free or z0.shape[-2] != q:
            raise ValueError(f"For SA, batch_initial_conditions must have shape (1, q={q}, d_free={d_free}) or (q, d_free).")
        if z0.shape[0] != 1:
            z0 = z0[:1]
        z_cur = z0.clone()
    else:
        z_cur = torch.rand(1, q, d_free, device=device, dtype=dtype)

    f_cur = evaluate_population(z_cur)[0]
    z_best = z_cur.clone()
    f_best = f_cur.clone()

    def temperature(step: int) -> float:
        if n_steps <= 1:
            return final_temp
        if use_log_schedule:
            t0 = torch.log(torch.tensor(init_temp))
            t1 = torch.log(torch.tensor(final_temp))
            alpha = step / (n_steps - 1)
            return float(torch.exp(t0 + alpha * (t1 - t0)))
        else:
            ratio = (final_temp / init_temp) ** (step / (n_steps - 1))
            return init_temp * ratio

    for step in range(n_steps):
        T = temperature(step)
        noise = torch.randn_like(z_cur) * step_size
        z_new = (z_cur + noise).clamp(0.0, 1.0)
        f_new = evaluate_population(z_new)[0]
        delta = f_new - f_cur  # maximize

        if delta >= 0:
            accept = True
        else:
            prob = torch.exp(delta / max(T, 1e-12))
            accept = bool(torch.rand(1, device=device) < prob)

        if accept:
            z_cur = z_new
            f_cur = f_new

        if f_new > f_best:
            z_best = z_new
            f_best = f_new

    best_z = z_best[0].detach().clone()  # (q, d_free)
    best_val = f_best.detach().clone().view(1)
    return best_z, best_val


# ================================
# CMA-ES backend (q=1 前提)
# ================================

def _optimize_acqf_cmaes_core(
    evaluate_population: Callable[[torch.Tensor], torch.Tensor],
    q: int,
    d_free: int,
    bounds: torch.Tensor,
    batch_initial_conditions: Optional[torch.Tensor],
    options: Dict,
):
    """
    CMA-ES backend。シンプルにするため q=1 のみ対応。
    q>1 で CMA-ES を使いたい場合は、sequential=True で1点ずつ呼ぶのを推奨。
    """
    if q != 1:
        raise NotImplementedError("CMA-ES backend currently assumes q=1. Use sequential=True to pick points one-by-one.")

    try:
        import cma
    except ImportError as e:
        raise ImportError("CMA-ES backend requires 'cma' package. Install via `pip install cma`.") from e

    device = bounds.device
    dtype = bounds.dtype

    sigma0 = float(options.get("sigma0", 0.3))
    maxiter = int(options.get("maxiter", 200))

    # 初期点 x0 (in [0,1]^d_free)
    if batch_initial_conditions is not None:
        z0 = batch_initial_conditions.to(device=device, dtype=dtype)
        if z0.ndim == 3 and z0.shape[-2] == 1 and z0.shape[-1] == d_free:
            x0_tensor = z0[0, 0]
        elif z0.ndim == 2 and z0.shape[-1] == d_free:
            x0_tensor = z0[0]
        else:
            raise ValueError(f"For CMA-ES, batch_initial_conditions must have shape (1, 1, d_free={d_free}) or (1, d_free={d_free}).")
        x0 = x0_tensor.tolist()
    else:
        x0 = [0.5] * d_free

    def _objective(x_flat):
        z = torch.tensor(x_flat, dtype=dtype, device=device).view(1, 1, d_free)
        fit = evaluate_population(z)[0].item()
        return -fit  # CMA-ES は min 化

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "bounds": [0.0, 1.0],
            "maxiter": maxiter,
            "verb_disp": 0,
        },
    )

    while not es.stop():
        xs = es.ask()
        fs = [_objective(x) for x in xs]
        es.tell(xs, fs)

    x_best = es.best.x
    z_best = torch.tensor(x_best, dtype=dtype, device=device).view(1, 1, d_free)
    best_z = z_best[0, 0].detach().clone()  # (1, d_free)
    best_val = evaluate_population(z_best)[0:1].detach().clone()
    return best_z, best_val

def candidate_transform_mixed_factory(
    categorical_features: Dict[int, Sequence[float]],
    bounds: torch.Tensor,
    base_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    連続 + カテゴリ変数の混合入力を扱うための candidate_transform を生成する。

    BoTorch の optimize_acqf_mixed における「カテゴリ側の丸め」に相当する処理を、
    evolutionary optimizer 用の transform として実装したもの。

    Args:
        categorical_features:
            {dim_index: [v0, v1, ...]} 形式の辞書。
            dim_index:
                X の列 index（0-based）。この次元をカテゴリ変数として扱う。
            [v0, v1, ...]:
                その次元で取り得る値の集合。int でも float でもよい。
                例: {3: [0, 1, 2]} なら「3列目は {0,1,2} のどれか」。
        bounds:
            optimize_acqf_evo に渡す bounds と同じ Tensor (2, d)。
            各カテゴリ次元については [lower, upper] のスケールを
            「内部的な [0,1] → カテゴリ index」変換にのみ利用する。
        base_transform:
            既存の candidate_transform がある場合はそれを渡す。
            mixed 用の丸め処理の前にこの transform が適用される。
        fixed_features:
            optimize_acqf_evo に渡す fixed_features と同じ dict。
            ここで指定された次元は、この transform 内でも上書きしない。

    Returns:
        transform(X): X (..., q, d) -> X_transformed (..., q, d)
    """
    fixed_features = fixed_features or {}
    # あらかじめ必要な情報を Tensor 化しておく
    cat_info = {}
    d = bounds.size(-1)

    for dim, values in categorical_features.items():
        if dim < 0 or dim >= d:
            raise ValueError(f"categorical_features の dim={dim} が bounds の次元数 d={d} を超えています。")
        vals_t = torch.as_tensor(values, dtype=bounds.dtype)
        if vals_t.numel() == 0:
            continue
        lower = bounds[0, dim].item()
        upper = bounds[1, dim].item()
        cat_info[dim] = {
            "values": vals_t,
            "lower": lower,
            "upper": upper,
        }

    def transform(X: torch.Tensor) -> torch.Tensor:
        # まず既存 transform を適用
        if base_transform is not None:
            X = base_transform(X)

        if not cat_info:
            return X

        X_new = X.clone()
        device = X_new.device
        dtype = X_new.dtype

        for dim, info in cat_info.items():
            # fixed_features に含まれている次元は触らない
            if dim in fixed_features:
                continue

            vals_t = info["values"].to(device=device, dtype=dtype)
            k = vals_t.numel()
            if k == 0:
                continue

            lower = torch.as_tensor(info["lower"], device=device, dtype=dtype)
            upper = torch.as_tensor(info["upper"], device=device, dtype=dtype)
            width = (upper - lower).clamp_min(1e-12)

            # X[..., :, dim] を [0,1] に正規化
            x_raw = X_new[..., :, dim]
            t = (x_raw - lower) / width
            t = t.clamp(0.0, 0.999999)  # ほぼ1.0 まで（idx=k にならないように）

            # t ∈ [0,1) を [0, k-1] の index に写像
            idx = torch.floor(t * k).long().clamp(0, k - 1)  # (..., q)

            # カテゴリ値に置き換え
            X_new[..., :, dim] = vals_t[idx]

        return X_new

    return transform

from typing import Sequence
import torch

def _project_sum_box_1d(
    x: torch.Tensor,   # (m,)
    lo: torch.Tensor,  # (m,)
    hi: torch.Tensor,  # (m,)
    rhs: float,
    n_bisect: int = 60,
) -> torch.Tensor:
    """
    min ||y-x||^2 s.t. sum(y)=rhs, lo<=y<=hi
    解は y = clamp(x + λ, lo, hi)（λを二分探索）
    """
    # 実現可能域に rhs を丸め（不可能な場合の暴走防止）
    rhs_min = float(lo.sum().item())
    rhs_max = float(hi.sum().item())
    rhs_eff = min(max(float(rhs), rhs_min), rhs_max)

    # λ の探索区間：min(lo-x) ～ max(hi-x)
    lam_lo = float((lo - x).min().item())
    lam_hi = float((hi - x).max().item())

    for _ in range(n_bisect):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        y = torch.clamp(x + lam_mid, lo, hi)
        s = float(y.sum().item())
        if s < rhs_eff:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return torch.clamp(x + lam_hi, lo, hi)


def enforce_sum_on_support(
    X: torch.Tensor,          # (q, d) or (d,)
    sum_idx: Sequence[int],   # 和を合わせる次元
    rhs: float,
    bounds: torch.Tensor,     # (2, d)
    support_eps: float = 1e-12,
) -> torch.Tensor:
    """
    sum_idx のうち「非ゼロ（|x|>eps）」の成分のみを動かして、和=rhs に補正する。
    ゼロ成分はゼロのまま（<=k を崩さない）。
    """
    X2 = X.clone()
    device, dtype = X2.device, X2.dtype
    idx_t = torch.tensor(list(sum_idx), device=device, dtype=torch.long)

    lo_all = bounds[0, idx_t].to(device=device, dtype=dtype)
    hi_all = bounds[1, idx_t].to(device=device, dtype=dtype)

    if X2.ndim == 1:
        X2 = X2.unsqueeze(0)  # (1, d)

    for i in range(X2.size(0)):  # q の各点
        g = X2[i, idx_t]  # (m,)
        active = g.abs() > support_eps
        if active.sum() == 0:
            continue

        x = g[active]
        lo = lo_all[active]
        hi = hi_all[active]

        # 0固定の成分は動かさないので、その分 rhs を調整（通常は 0 なので同じ）
        rhs_eff = float(rhs - g[~active].sum().item())

        y = _project_sum_box_1d(x, lo, hi, rhs_eff)
        g_new = torch.zeros_like(g)
        g_new[active] = y
        X2[i, idx_t] = g_new

    if X.ndim == 1:
        return X2[0]
    return X2

# ================================
# メインラッパー
# ================================

def optimize_acqf_evo(
    acq_function: AcquisitionFunction,
    bounds: torch.Tensor,
    q: int = 1,
    method: str = "ga",  # "ga", "pso", "sa", "cmaes"
    num_restarts: int = 10,   # 互換性用（未使用）
    raw_samples: int = 512,   # 互換性用（未使用）
    inequality_constraints: Optional[List[Tuple[List[int], List[float], float]]] = None,
    equality_constraints: Optional[List[Tuple[List[int], List[float], float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    batch_initial_conditions: Optional[torch.Tensor] = None,
    return_best_only: bool = True,  # 現状 True 前提
    sequential: bool = False,
    options: Optional[Dict] = None,
    candidate_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    X_pending: Optional[torch.Tensor] = None,
):
    """
    BoTorch の optimize_acqf に近いインターフェースで、
    GA / PSO / SA / CMA-ES を backend に使う最適化器。

    - method: "ga", "pso", "sa", "cmaes"
    - q>1 & sequential=True で X_pending を更新しながら逐次的に候補生成する。
    - post_processing_func は
        - 評価時 (evaluate_population 内)
        - 返り値の X にも
      適用される。
    """
    if not return_best_only:
        raise NotImplementedError("return_best_only=False は未サポートです。")

    if options is None:
        options = {}
    method_l = method.lower()
    penalty_factor = float(options.get("penalty_factor", 1e3))

    device = bounds.device
    dtype = bounds.dtype

    # acq_function に既に X_pending が入っていれば、それをベースにする
    base_X_pending = X_pending
    if base_X_pending is None and hasattr(acq_function, "X_pending"):
        base_X_pending = getattr(acq_function, "X_pending")

    def _run_single(q_local: int, X_pending_local: Optional[torch.Tensor]):
        # X_pending をセット
        _set_X_pending_on_acqf(acq_function, X_pending_local)

        # 共通 decode / evaluate を構築
        free_idx, decode, evaluate_population = _build_decode_and_evaluate(
            acq_function=acq_function,
            bounds=bounds,
            q=q_local,
            fixed_features=fixed_features,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            candidate_transform=candidate_transform,
            post_processing_func=post_processing_func,
            penalty_factor=penalty_factor,
        )
        d_free = len(free_idx)

        # backend を選択
        if method_l == "ga":
            best_z, best_val = _optimize_acqf_ga_core(
                evaluate_population=evaluate_population,
                q=q_local,
                d_free=d_free,
                bounds=bounds,
                batch_initial_conditions=batch_initial_conditions,
                options=options,
            )
        elif method_l == "pso":
            best_z, best_val = _optimize_acqf_pso_core(
                evaluate_population=evaluate_population,
                q=q_local,
                d_free=d_free,
                bounds=bounds,
                batch_initial_conditions=batch_initial_conditions,
                options=options,
            )
        elif method_l == "sa":
            best_z, best_val = _optimize_acqf_sa_core(
                evaluate_population=evaluate_population,
                q=q_local,
                d_free=d_free,
                bounds=bounds,
                batch_initial_conditions=batch_initial_conditions,
                options=options,
            )
        elif method_l == "cmaes":
            best_z, best_val = _optimize_acqf_cmaes_core(
                evaluate_population=evaluate_population,
                q=q_local,
                d_free=d_free,
                bounds=bounds,
                batch_initial_conditions=batch_initial_conditions,
                options=options,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # best_z: (q_local, d_free) -> X_best
        z_batch = best_z.unsqueeze(0)  # (1, q_local, d_free)
        X_dec = decode(z_batch)[0]     # (q_local, d_total)
        if post_processing_func is not None:
            X_best = post_processing_func(X_dec.unsqueeze(0))[0]
        else:
            X_best = X_dec

        sum_spec = options.get("final_sum_constraint", None) if options is not None else None
        if sum_spec is not None:
            sum_idx, rhs = sum_spec  # (Sequence[int], float)
            X_best = enforce_sum_on_support(X_best, sum_idx=sum_idx, rhs=rhs, bounds=bounds)

        return X_best, best_val

    # q=1 or 非逐次モード
    if (not sequential) or q == 1:
        return _run_single(q_local=q, X_pending_local=base_X_pending)

    # q>1 & sequential=True: 逐次的に候補を追加
    selected: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []

    for i in range(q):
        cur_pending = base_X_pending
        if len(selected) > 0:
            X_sel = torch.stack(selected, dim=-2)  # (i, d)
            if cur_pending is None:
                cur_pending = X_sel
            else:
                cur_pending = torch.cat([cur_pending, X_sel], dim=-2)

        X_i, v_i = _run_single(q_local=1, X_pending_local=cur_pending)  # X_i: (1, d)
        # selected.append(X_i[0].detach())
        selected.append(X_i.detach())
        vals.append(v_i[0].detach())

    X_all = torch.stack(selected, dim=-2)      # (q, d)
    vals_all = torch.stack(vals).view(q, 1)    # (q, 1)
    return X_all[0], vals_all

def optimize_acqf_evo_mixed(
    acq_function,
    bounds: torch.Tensor,
    q: int = 1,
    method: str = "ga",
    categorical_features: Optional[Dict[int, Sequence[float]]] = None,
    num_restarts: int = 10,
    raw_samples: int = 512,
    inequality_constraints=None,
    equality_constraints=None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    batch_initial_conditions: Optional[torch.Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: Optional[Dict] = None,
    candidate_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    X_pending: Optional[torch.Tensor] = None,
    # ★追加（任意）：最終和合わせ
    # 例: final_sum_constraint=(comp_idx, 1.0)
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
):
    categorical_features = categorical_features or {}
    fixed_features = fixed_features or {}
    options = dict(options or {})  # 呼び出し側を汚さない

    # mixed 用 transform を組み立てる（decode 中に適用される）
    mixed_transform = candidate_transform
    if categorical_features:
        mixed_transform = candidate_transform_mixed_factory(
            categorical_features=categorical_features,
            bounds=bounds,
            base_transform=candidate_transform,
            fixed_features=fixed_features,
        )

    # # ★重要：post_processing の後にもう一度カテゴリ丸め & 最終和合わせ
    # def mixed_post_processing(X: torch.Tensor) -> torch.Tensor:
    #     # 1) まず既存の repair 等
    #     if post_processing_func is not None:
    #         X = post_processing_func(X)

    #     # 2) repair がカテゴリを崩す可能性があるので、最後にカテゴリ丸めを再適用
    #     if mixed_transform is not None:
    #         X = mixed_transform(X)

    #     # 3) 最終和合わせ（必要な場合のみ）
    #     if final_sum_constraint is not None:
    #         sum_idx, rhs = final_sum_constraint
    #         # X は (..., q, d) 想定だが、enforce_sum_on_support 側で (q,d) も扱える実装なら
    #         # ここでは各バッチをまとめて渡してもOKにできます。
    #         # 簡単のため、X が (q,d) の場合と (...,q,d) の場合で分岐:
    #         if X.ndim == 2:
    #             X = enforce_sum_on_support(X, sum_idx=sum_idx, rhs=rhs, bounds=bounds)
    #         elif X.ndim >= 3:
    #             # (..., q, d) をフラット化して点ごとに補正して戻す
    #             orig = X.shape
    #             Xf = X.reshape(-1, orig[-2], orig[-1])  # (N, q, d)
    #             for i in range(Xf.size(0)):
    #                 Xf[i] = enforce_sum_on_support(Xf[i], sum_idx=sum_idx, rhs=rhs, bounds=bounds)
    #             X = Xf.reshape(orig)

    #     return X

    # あとは普通の evo 版に流すだけ
    return optimize_acqf_evo(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        method=method,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,   # ★差し替え
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        sequential=sequential,
        options=options,
        candidate_transform=mixed_transform,          # decode 時の丸めも維持
        X_pending=X_pending,
    )