"""Utilities to add categorical-variable support to SAAS fully Bayesian GP models.

This module provides ``MixedSaasFullyBayesianSingleTaskGP`` with an API close to
``MixedSingleTaskGP``: users can pass ``cat_dims`` that mark categorical input
columns. Categorical columns are expanded with one-hot encoding before fitting a
``SaasFullyBayesianSingleTaskGP``.

The implementation intentionally keeps the model surface area small and delegates
all Bayesian inference and posterior behavior to the base SAAS model.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from itertools import product
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Any

import torch
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models import SingleTaskGP

# ====================================================================================
# SAAS
# ====================================================================================

@dataclass(frozen=True)
class _CategoricalSpec:
    """Metadata for one categorical source column in the original feature space."""

    source_dim: int
    categories: torch.Tensor
    encoded_indices: torch.Tensor


class MixedSaasFullyBayesianSingleTaskGP(SaasFullyBayesianSingleTaskGP):
    r"""SAAS fully Bayesian GP with categorical-variable support.

    The class mirrors the ``cat_dims`` user experience from ``MixedSingleTaskGP``
    while retaining the SAAS fully Bayesian backend. Categorical columns are
    converted to one-hot blocks, then passed to
    :class:`SaasFullyBayesianSingleTaskGP`.

    Args:
        train_X: Training features of shape ``n x d``.
        train_Y: Training targets of shape ``n x m`` (typically ``m=1``).
        cat_dims: Indices of categorical columns in the *original* ``train_X``.
            If omitted, this class behaves exactly like the base SAAS model.
        train_Yvar: Optional observation noise.
        **kwargs: Forwarded to ``SaasFullyBayesianSingleTaskGP``.
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        cat_dims: Optional[Sequence[int]] = None,
        train_Yvar: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        self._raw_dim = train_X.shape[-1]
        self._cat_dims = sorted(set(int(i) for i in (cat_dims or [])))
        self._cat_specs = self._infer_cat_specs(train_X=train_X, cat_dims=self._cat_dims)
        encoded_train_X = self._encode_X(train_X)

        if "input_transform" in kwargs:
            kwargs["input_transform"] = self._maybe_expand_input_transform(
                kwargs["input_transform"]
            )

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            **kwargs,
        )

    @property
    def cat_dims(self) -> List[int]:
        """Categorical dimensions in the original (pre-encoded) feature space."""
        return list(self._cat_dims)

    @property
    def encoded_dim(self) -> int:
        """Feature dimension after categorical one-hot expansion."""
        if not self._cat_specs:
            return self._raw_dim
        max_encoded_idx = max(int(spec.encoded_indices.max().item()) for spec in self._cat_specs.values())
        return max_encoded_idx + 1

    @property
    def raw_dim(self) -> int:
        """Feature dimension before one-hot expansion."""
        return self._raw_dim

    @property
    def encoded_cat_dims(self) -> Dict[int, List[int]]:
        """Mapping: raw categorical dim -> one-hot encoded indices."""
        return {
            d: [int(i) for i in spec.encoded_indices.tolist()]
            for d, spec in self._cat_specs.items()
        }

    def get_optimize_acqf_mixed_fixed_features_list(self) -> List[Dict[int, float]]:
        """Return encoded `fixed_features_list` for `optimize_acqf_mixed`.

        This enumerates all categorical assignments exactly like mixed
        optimization does conceptually, but in the encoded one-hot feature space.
        """
        if not self._cat_specs:
            return []

        per_dim_assignments: List[List[Dict[int, float]]] = []
        for d in self._cat_dims:
            spec = self._cat_specs[d]
            assignments: List[Dict[int, float]] = []
            for active in range(len(spec.categories)):
                one = {
                    int(encoded_idx): float(i == active)
                    for i, encoded_idx in enumerate(spec.encoded_indices.tolist())
                }
                assignments.append(one)
            per_dim_assignments.append(assignments)

        fixed_features_list: List[Dict[int, float]] = []
        for combo in product(*per_dim_assignments):
            merged: Dict[int, float] = {}
            for part in combo:
                merged.update(part)
            fixed_features_list.append(merged)
        return fixed_features_list

    def transform_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
        """Transform raw-space bounds `[2, raw_dim]` to encoded-space bounds.

        Encoded categorical one-hot columns always become `[0, 1]`.
        """
        if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] != self._raw_dim:
            raise ValueError(
                f"Expected bounds shape [2, {self._raw_dim}], got {tuple(bounds.shape)}."
            )
        encoded_pieces: List[torch.Tensor] = []
        for d in range(self._raw_dim):
            if d not in self._cat_specs:
                encoded_pieces.append(bounds[:, d : d + 1])
                continue
            spec = self._cat_specs[d]
            cat_bounds = torch.zeros(
                2,
                len(spec.categories),
                dtype=bounds.dtype,
                device=bounds.device,
            )
            cat_bounds[1, :] = 1
            encoded_pieces.append(cat_bounds)
        return torch.cat(encoded_pieces, dim=-1)

    def transform_inputs(
        self,
        X: torch.Tensor,
        input_transform=None,
    ) -> torch.Tensor:  # noqa: N802
        """Transform inputs for the SAAS model.

        - raw-space input (`[..., raw_dim]`) is one-hot encoded.
        - encoded-space input (`[..., encoded_dim]`) is passed through.
        - optional `input_transform` is then applied via BoTorch base logic.
        """
        if X.shape[-1] == self._raw_dim:
            X_enc = self._encode_X(X)
        elif X.shape[-1] == self.encoded_dim:
            X_enc = X
        else:
            raise ValueError(
                f"Expected input dim {self._raw_dim} (raw) or {self.encoded_dim} (encoded), "
                f"but got {X.shape[-1]}."
            )
        return super().transform_inputs(X=X_enc, input_transform=input_transform)

    def decode_inputs(self, X_encoded: torch.Tensor) -> torch.Tensor:
        """Decode encoded inputs back to raw-space representation.

        For categorical dimensions, this maps one-hot blocks to category labels
        using argmax.
        """
        if X_encoded.shape[-1] != self.encoded_dim:
            raise ValueError(
                f"Expected encoded input dim {self.encoded_dim}, but got {X_encoded.shape[-1]}."
            )

        raw_pieces: List[torch.Tensor] = []
        for d in range(self._raw_dim):
            if d not in self._cat_specs:
                # non-categorical dimensions keep a single coordinate in-place
                # due to the deterministic encoding order.
                raw_idx = self._raw_to_single_encoded_index(d)
                raw_pieces.append(X_encoded[..., raw_idx : raw_idx + 1])
                continue

            spec = self._cat_specs[d]
            cat_block = X_encoded[..., spec.encoded_indices]
            argmax_idx = cat_block.argmax(dim=-1)
            category_values = spec.categories[argmax_idx].unsqueeze(-1).to(dtype=X_encoded.dtype)
            raw_pieces.append(category_values)

        return torch.cat(raw_pieces, dim=-1)

    def _maybe_expand_input_transform(self, input_transform):
        """Expand selected raw-space input transforms to encoded-space transforms.

        Currently, this supports ``Normalize`` when provided with ``d == raw_dim``.
        For categorical one-hot columns, encoded bounds are set to ``[0, 1]``.
        """
        if input_transform is None or not self._cat_specs:
            return input_transform

        if isinstance(input_transform, Normalize):
            # transform_d = getattr(input_transform, "d", None)
            bounds = getattr(input_transform, "bounds", None)
            transform_d = bounds.shape[-1]
            if transform_d == self._raw_dim:
                # bounds = getattr(input_transform, "bounds", None)
                encoded_bounds = None
                if isinstance(bounds, torch.Tensor) and bounds.shape[-1] == self._raw_dim:
                    encoded_bounds = self.transform_bounds(bounds)
                warnings.warn(
                    "Expanded Normalize input_transform from raw-space to encoded-space. "
                    "For one-hot categorical columns, normalization bounds are fixed to [0, 1]."
                )
                return Normalize(d=self.encoded_dim, bounds=encoded_bounds)

        transform_d = getattr(input_transform, "d", None)
        if transform_d == self._raw_dim:
            raise ValueError(
                "input_transform appears to be configured for raw feature dimension, "
                "but the internal model uses encoded features. "
                "Pass Normalize(d=raw_dim, bounds=raw_bounds) for auto expansion, "
                "or provide an encoded-dimension input_transform explicitly."
            )
        return input_transform

    def _to_training_feature_space(self, X: torch.Tensor) -> torch.Tensor:
        """Map inputs to the same feature space used by internal train inputs."""
        train_feature_dim = self.train_inputs[0].shape[-1]
        if train_feature_dim == self.encoded_dim:
            if X.shape[-1] == self._raw_dim:
                return self._encode_X(X)
            if X.shape[-1] == self.encoded_dim:
                return X
        elif train_feature_dim == self._raw_dim:
            if X.shape[-1] == self.encoded_dim:
                return self.decode_inputs(X)
            if X.shape[-1] == self._raw_dim:
                return X

        raise ValueError(
            f"Could not map input dim {X.shape[-1]} to training feature dim {train_feature_dim}. "
            f"Expected raw dim {self._raw_dim} or encoded dim {self.encoded_dim}."
        )

    def posterior(self, X: torch.Tensor, *args, **kwargs):
        """Compute posterior for raw or encoded input tensors."""
        X_eval = self._to_training_feature_space(X)
        return super().posterior(X_eval, *args, **kwargs)

    @classmethod
    def construct_inputs(
        cls,
        training_data,
        cat_dims: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> Dict[str, object]:
        """Build constructor kwargs in a style similar to MixedSingleTaskGP.

        This method intentionally preserves BoTorch's ``construct_inputs`` flow
        while adding ``cat_dims``.
        """
        base_inputs = super().construct_inputs(training_data=training_data, **kwargs)
        base_inputs["cat_dims"] = list(cat_dims or [])
        return base_inputs

    def _raw_to_single_encoded_index(self, raw_dim: int) -> int:
        """Get encoded index for a non-categorical raw dimension."""
        encoded_idx = 0
        for d in range(self._raw_dim):
            if d == raw_dim:
                return encoded_idx
            if d in self._cat_specs:
                encoded_idx += len(self._cat_specs[d].categories)
            else:
                encoded_idx += 1
        raise RuntimeError(f"Invalid raw dimension {raw_dim}.")

    @staticmethod
    def _infer_cat_specs(
        train_X: torch.Tensor,
        cat_dims: Sequence[int],
    ) -> Mapping[int, _CategoricalSpec]:
        specs: MutableMapping[int, _CategoricalSpec] = {}
        encoded_cursor = 0
        cat_dim_set = set(cat_dims)
        for d in range(train_X.shape[-1]):
            if d not in cat_dim_set:
                encoded_cursor += 1
                continue

            col = train_X[..., d]
            if col.dtype.is_floating_point:
                # MixedSingleTaskGP expects integer-like category labels.
                col = col.round()
            categories = torch.unique(col).sort().values
            encoded_indices = torch.arange(
                encoded_cursor,
                encoded_cursor + len(categories),
                dtype=torch.long,
                device=train_X.device,
            )
            specs[d] = _CategoricalSpec(
                source_dim=d,
                categories=categories,
                encoded_indices=encoded_indices,
            )
            encoded_cursor += len(categories)

        return dict(specs)

    def _encode_X(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != self._raw_dim:
            raise ValueError(
                f"Expected raw input dim {self._raw_dim}, but got {X.shape[-1]}."
            )
        if not self._cat_specs:
            return X

        pieces: List[torch.Tensor] = []
        for d in range(self._raw_dim):
            if d not in self._cat_specs:
                pieces.append(X[..., d : d + 1])
                continue

            spec = self._cat_specs[d]
            x_d = X[..., d : d + 1]
            if x_d.dtype.is_floating_point:
                x_d = x_d.round()

            # Broadcast comparison to produce one-hot block:
            # [..., 1] == [k] -> [..., k]
            oh = (x_d == spec.categories).to(dtype=X.dtype)
            if torch.any(oh.sum(dim=-1) == 0):
                raise ValueError(
                    f"Input includes unseen category in raw dim {d}. "
                    f"Known categories: {spec.categories.tolist()}"
                )
            pieces.append(oh)

        return torch.cat(pieces, dim=-1)

# ====================================================================================
# 次元削減
# ====================================================================================

@dataclass
class PCAConfig:
    n_components: int
    standardize: bool = True
    eps: float = 1e-8


@dataclass
class REMBOConfig:
    n_components: int
    seed: int | None = None
    normalize: bool = True
    eps: float = 1e-8
    projection_matrix: torch.Tensor | None = None


class PCATransformer:
    def __init__(self, config: PCAConfig):
        self.config = config
        self.mean_: torch.Tensor | None = None
        self.scale_: torch.Tensor | None = None
        self.components_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "PCATransformer":
        if x.dim() != 2:
            raise ValueError("x must be 2D tensor [n, d].")
        if self.config.n_components > x.shape[-1]:
            raise ValueError("n_components must be <= input dimension.")

        self.mean_ = x.mean(dim=0, keepdim=True)
        xc = x - self.mean_
        if self.config.standardize:
            self.scale_ = xc.std(dim=0, keepdim=True).clamp_min(self.config.eps)
            xc = xc / self.scale_
        else:
            self.scale_ = torch.ones_like(self.mean_)

        _, _, vh = torch.linalg.svd(xc, full_matrices=False)
        self.components_ = vh[: self.config.n_components].T
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        x_flat, lead_shape = self._flatten_last_dim(x)
        z_flat = ((x_flat - self.mean_) / self.scale_) @ self.components_
        return z_flat.reshape(*lead_shape, self.components_.shape[-1])

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        z_flat, lead_shape = self._flatten_last_dim(z)
        x_flat = (z_flat @ self.components_.T) * self.scale_ + self.mean_
        return x_flat.reshape(*lead_shape, self.components_.shape[0])

    def _flatten_last_dim(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.shape[-1] <= 0:
            raise ValueError("Last dimension must be positive.")
        lead_shape = x.shape[:-1]
        return x.reshape(-1, x.shape[-1]), lead_shape

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("PCATransformer is not fitted yet.")


class REMBOTransformer:
    def __init__(self, config: REMBOConfig):
        self.config = config
        self.mean_: torch.Tensor | None = None
        self.scale_: torch.Tensor | None = None
        self.projection_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "REMBOTransformer":
        if x.dim() != 2:
            raise ValueError("x must be 2D tensor [n, d].")
        d = x.shape[-1]
        if self.config.n_components > d:
            raise ValueError("n_components must be <= input dimension.")

        self.mean_ = x.mean(dim=0, keepdim=True)
        xc = x - self.mean_
        if self.config.normalize:
            self.scale_ = xc.std(dim=0, keepdim=True).clamp_min(self.config.eps)
        else:
            self.scale_ = torch.ones_like(self.mean_)

        proj = self.config.projection_matrix
        if proj is None:
            gen = None
            if self.config.seed is not None:
                gen = torch.Generator(device=x.device)
                gen.manual_seed(self.config.seed)
            proj = torch.randn(d, self.config.n_components, dtype=x.dtype, device=x.device, generator=gen)
        if proj.shape != (d, self.config.n_components):
            raise ValueError(
                f"projection_matrix must have shape {(d, self.config.n_components)}, got {tuple(proj.shape)}."
            )
        self.projection_ = proj / proj.norm(dim=0, keepdim=True).clamp_min(self.config.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        x_flat, lead_shape = self._flatten_last_dim(x)
        z_flat = ((x_flat - self.mean_) / self.scale_) @ self.projection_
        return z_flat.reshape(*lead_shape, self.projection_.shape[-1])

    def _flatten_last_dim(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.shape[-1] <= 0:
            raise ValueError("Last dimension must be positive.")
        lead_shape = x.shape[:-1]
        return x.reshape(-1, x.shape[-1]), lead_shape

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None or self.projection_ is None:
            raise RuntimeError("REMBOTransformer is not fitted yet.")


class REMBOSingleTaskGP(SingleTaskGP):
    """`SingleTaskGP` style REMBO model (random embedding for continuous dimensions)."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor | None = None,
        likelihood: Any | None = None,
        covar_module: Any | None = None,
        mean_module: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: Any | None = None,
        rembo_config: REMBOConfig | None = None,
        n_components: int | None = None,
    ):
        if train_X.dim() != 2:
            raise ValueError("train_X must be 2D tensor [n, d].")
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        n_comp = n_components if n_components is not None else train_X.shape[-1]
        self.rembo_config = rembo_config or REMBOConfig(n_components=n_comp)
        self.rembo = REMBOTransformer(self.rembo_config)
        self.input_dim_original = train_X.shape[-1]

        self.train_X_original = train_X
        self.train_Y_original = train_Y
        self.train_Yvar_original = train_Yvar

        latent_X = self.rembo.fit(train_X).transform(train_X)

        super().__init__(
            train_X=latent_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    def posterior(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        Z = self._to_latent(X)
        return super().posterior(Z, *args, **kwargs)

    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, **kwargs):
        # Y: (..., m) を最低限保証（BoTorch は fantasize で高次元を渡す）
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
    
        Z = self._to_latent(X)  # Xが2Dでも3DでもOK（末次元だけ見て変換）
        return super().condition_on_observations(Z, Y, **kwargs)

    def _to_latent(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] == self.input_dim_original:
            return self.rembo.transform(X)
        if X.shape[-1] == self.rembo_config.n_components:
            return X
        raise ValueError(
            f"X.shape[-1] must be original dim {self.input_dim_original} "
            f"or latent dim {self.rembo_config.n_components}."
        )


class REMBOMixedSingleTaskGP(MixedSingleTaskGP):
    """`MixedSingleTaskGP` style REMBO model for mixed continuous/categorical inputs."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        cat_dims: list[int],
        train_Yvar: torch.Tensor | None = None,
        cont_kernel_factory: Any | None = None,
        likelihood: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: Any | None = None,
        rembo_config: REMBOConfig | None = None,
        n_components: int | None = None,
    ):
        if train_X.dim() != 2:
            raise ValueError("train_X must be 2D tensor [n, d].")
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        self.input_dim_original = train_X.shape[-1]
        self.cat_dims_original = sorted(cat_dims)
        self._validate_cat_dims(self.cat_dims_original)

        self.cont_dims_original = [d for d in range(self.input_dim_original) if d not in self.cat_dims_original]
        if not self.cont_dims_original:
            raise ValueError("At least one continuous dimension is required for REMBO.")

        cont_X = train_X[..., self.cont_dims_original]
        n_comp = n_components if n_components is not None else cont_X.shape[-1]
        self.rembo_config = rembo_config or REMBOConfig(n_components=n_comp)
        self.rembo = REMBOTransformer(self.rembo_config)

        self.train_X_original = train_X
        self.train_Y_original = train_Y
        self.train_Yvar_original = train_Yvar

        latent_X = self._to_internal(train_X, fit_rembo=True)
        latent_cat_dims = list(range(self.rembo_config.n_components, latent_X.shape[-1]))

        super().__init__(
            train_X=latent_X,
            train_Y=train_Y,
            cat_dims=latent_cat_dims,
            train_Yvar=train_Yvar,
            cont_kernel_factory=cont_kernel_factory,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    def posterior(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        Z = self._to_internal(X)
        return super().posterior(Z, *args, **kwargs)

    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, **kwargs):
        # Y: (..., m) を最低限保証（BoTorch は fantasize で高次元を渡す）
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
    
        Z = self._to_internal(X)  # Xが2Dでも3DでもOK（末次元だけ見て変換）
        return super().condition_on_observations(Z, Y, **kwargs)

    def _validate_cat_dims(self, cat_dims: list[int]) -> None:
        for d in cat_dims:
            if d < 0 or d >= self.input_dim_original:
                raise ValueError(f"cat_dim {d} is out of range for input dim {self.input_dim_original}.")

    def _to_internal(self, X: torch.Tensor, fit_rembo: bool = False) -> torch.Tensor:
        if X.shape[-1] == self.rembo_config.n_components + len(self.cat_dims_original):
            return X
        if X.shape[-1] != self.input_dim_original:
            raise ValueError(
                f"X.shape[-1] must be original dim {self.input_dim_original} "
                f"or internal dim {self.rembo_config.n_components + len(self.cat_dims_original)}."
            )

        x_cont = X[..., self.cont_dims_original]
        if fit_rembo:
            x_cont = self.rembo.fit(x_cont).transform(x_cont)
        else:
            x_cont = self.rembo.transform(x_cont)
        x_cat = X[..., self.cat_dims_original]
        return torch.cat([x_cont, x_cat], dim=-1)


class PCASingleTaskGP(SingleTaskGP):
    """Drop-inに近い `SingleTaskGP` 拡張。

    - 初期化引数は `SingleTaskGP` 準拠
    - 受け取る `X` は元空間
    - 内部でPCA潜在空間へ変換して `SingleTaskGP` を学習
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor | None = None,
        likelihood: Any | None = None,
        covar_module: Any | None = None,
        mean_module: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: Any | None = None,
        pca_config: PCAConfig | None = None,
        n_components: int | None = None,
    ):
        if train_X.dim() != 2:
            raise ValueError("train_X must be 2D tensor [n, d].")
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        n_comp = n_components if n_components is not None else train_X.shape[-1]
        self.pca_config = pca_config or PCAConfig(n_components=n_comp)
        self.pca = PCATransformer(self.pca_config)
        self.input_dim_original = train_X.shape[-1]

        self.train_X_original = train_X
        self.train_Y_original = train_Y
        self.train_Yvar_original = train_Yvar

        latent_X = self.pca.fit(train_X).transform(train_X)

        super().__init__(
            train_X=latent_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    def posterior(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        Z = self._to_latent(X)
        return super().posterior(Z, *args, **kwargs)

    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, **kwargs):
        # Y: (..., m) を最低限保証（BoTorch は fantasize で高次元を渡す）
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
    
        Z = self._to_latent(X)  # Xが2Dでも3DでもOK（末次元だけ見て変換）
        return super().condition_on_observations(Z, Y, **kwargs)

    def _to_latent(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] == self.input_dim_original:
            return self.pca.transform(X)
        if X.shape[-1] == self.pca_config.n_components:
            return X
        raise ValueError(
            f"X.shape[-1] must be original dim {self.input_dim_original} "
            f"or latent dim {self.pca_config.n_components}."
        )


class PCAMixedSingleTaskGP(MixedSingleTaskGP):
    """`MixedSingleTaskGP` に PCA を組み合わせたモデル。

    カテゴリカル次元 (`cat_dims`) はそのまま使い、
    連続次元のみ PCA で圧縮した潜在表現を内部モデルへ渡します。
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        cat_dims: list[int],
        train_Yvar: torch.Tensor | None = None,
        cont_kernel_factory: Any | None = None,
        likelihood: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: Any | None = None,
        pca_config: PCAConfig | None = None,
        n_components: int | None = None,
    ):
        if train_X.dim() != 2:
            raise ValueError("train_X must be 2D tensor [n, d].")
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        self.input_dim_original = train_X.shape[-1]
        self.cat_dims_original = sorted(cat_dims)
        self._validate_cat_dims(self.cat_dims_original)

        self.cont_dims_original = [
            d for d in range(self.input_dim_original) if d not in self.cat_dims_original
        ]
        if not self.cont_dims_original:
            raise ValueError("At least one continuous dimension is required for PCA.")

        cont_X = train_X[..., self.cont_dims_original]
        n_comp = n_components if n_components is not None else cont_X.shape[-1]
        self.pca_config = pca_config or PCAConfig(n_components=n_comp)
        self.pca = PCATransformer(self.pca_config)

        self.train_X_original = train_X
        self.train_Y_original = train_Y
        self.train_Yvar_original = train_Yvar

        latent_X = self._to_internal(train_X, fit_pca=True)
        latent_cat_dims = list(range(self.pca_config.n_components, latent_X.shape[-1]))

        super().__init__(
            train_X=latent_X,
            train_Y=train_Y,
            cat_dims=latent_cat_dims,
            train_Yvar=train_Yvar,
            cont_kernel_factory=cont_kernel_factory,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    def posterior(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        Z = self._to_internal(X)
        return super().posterior(Z, *args, **kwargs)

    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, **kwargs):
        # Y: (..., m) を最低限保証（BoTorch は fantasize で高次元を渡す）
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
    
        Z = self._to_internal(X)  # Xが2Dでも3DでもOK（末次元だけ見て変換）
        return super().condition_on_observations(Z, Y, **kwargs)

    def _validate_cat_dims(self, cat_dims: list[int]) -> None:
        for d in cat_dims:
            if d < 0 or d >= self.input_dim_original:
                raise ValueError(f"cat_dim {d} is out of range for input dim {self.input_dim_original}.")

    def _to_internal(self, X: torch.Tensor, fit_pca: bool = False) -> torch.Tensor:
        if X.shape[-1] == self.pca_config.n_components + len(self.cat_dims_original):
            return X
        if X.shape[-1] != self.input_dim_original:
            raise ValueError(
                f"X.shape[-1] must be original dim {self.input_dim_original} "
                f"or internal dim {self.pca_config.n_components + len(self.cat_dims_original)}."
            )

        x_cont = X[..., self.cont_dims_original]
        if fit_pca:
            x_cont = self.pca.fit(x_cont).transform(x_cont)
        else:
            x_cont = self.pca.transform(x_cont)
        x_cat = X[..., self.cat_dims_original]
        return torch.cat([x_cont, x_cat], dim=-1)