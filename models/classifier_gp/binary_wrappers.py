from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor
from gpytorch.likelihoods import BernoulliLikelihood

from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform

from .models import (
    SimpleBernoulliPosterior,
    GPClassificationModel,
    DeepGPClassificationModel,
    DeepKernelClassificationModel,
    DeepKernelDeepGPClassificationModel,
    GPClassificationMixedModel,
    DeepMixedGPClassificationModel,
    DeepKernelMixedClassificationModel,
    DeepKernelDeepMixedGPClassificationModel,
)


class ClassifierGPBinaryFromMulticlass(Model):
    """Binary wrapper from multiclass labels for continuous-only features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        target_class: Union[int, Sequence[int]],
        model: Optional[GPClassificationModel] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",
        deep_gp: bool = False,
        deep_kernel: bool = False,
        list_hidden_dims: Optional[List[int]] = None,
        num_inducing_points: int = 20,
    ):
        super().__init__()

        self.train_inputs = (train_X,)

        if isinstance(target_class, int):
            self.target_class_set = {target_class}
        else:
            self.target_class_set = set(target_class)

        self.train_targets = torch.tensor(
            [float(y.item() in self.target_class_set) for y in train_Y],
            device=train_Y.device,
            dtype=train_X.dtype,
        )

        self.input_transform = input_transform
        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)
            _ = self.input_transform(train_X)
            self.input_transform.eval()

        self.train_inputs_raw = (train_X,)
        transformed_train_X = self.input_transform(train_X) if self.input_transform else train_X

        if model is not None:
            self.model = model
        elif deep_gp and deep_kernel:
            self.model = DeepKernelDeepGPClassificationModel(
                transformed_train_X,
                self.train_targets,
                list_hidden_dims=list_hidden_dims,
                num_inducing_points=max(num_inducing_points, 32),
            )
        elif deep_gp:
            self.model = DeepGPClassificationModel(
                transformed_train_X,
                self.train_targets,
                list_hidden_dims=list_hidden_dims,
                num_inducing_points=max(num_inducing_points, 32),
            )
        elif deep_kernel:
            self.model = DeepKernelClassificationModel(
                transformed_train_X, self.train_targets, num_inducing_points
            )
        else:
            self.model = GPClassificationModel(
                transformed_train_X, self.train_targets, num_inducing_points
            )
        self.likelihood = likelihood if likelihood is not None else BernoulliLikelihood()
        self.to(train_X)

    def set_train_data(self, inputs=None, targets=None, strict: bool = True):
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
        if targets is not None:
            self.train_targets = targets

    def forward(self, X: Tensor):
        if isinstance(X, tuple):
            X = X[0]
        transformed_X = self.input_transform(X) if self.input_transform else X
        return self.model(transformed_X)

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs) -> SimpleBernoulliPosterior:
        self.eval()
        self.likelihood.eval()

        if isinstance(X, tuple):
            X = X[0]

        transformed_X = self.input_transform(X) if self.input_transform else X
        latent = self.model(transformed_X)
        preds = self.likelihood(latent)
        p = preds.mean
        var = preds.variance

        noise_model = getattr(self, "noise_model", None)
        if observation_noise and noise_model is not None:
            noise_in = transformed_X if getattr(self, "noise_model_uses_transformed_inputs", True) else X
            noise_log_var = noise_model.posterior(noise_in).mean
            noise_var = torch.exp(noise_log_var).clamp_min(1e-9)
            noise_var = noise_var.reshape_as(var)
            var = var + noise_var

        return SimpleBernoulliPosterior(mean=p, variance=var)

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return self.model.batch_shape


class ClassifierMixedGPBinaryFromMulticlass(Model):
    """Binary wrapper from multiclass labels for mixed (continuous/categorical) features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        target_class: Union[int, Sequence[int]],
        cat_dims,
        model=None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform=None,
        deep_gp: bool = False,
        deep_kernel: bool = False,
        list_hidden_dims: Optional[List[int]] = None,
        num_inducing_points: int = 20,
    ):
        super().__init__()

        self.train_inputs_raw = (train_X,)
        self.train_inputs = (train_X,)

        if isinstance(target_class, int):
            self.target_class_set = {target_class}
        else:
            self.target_class_set = set(target_class)

        self.train_targets = torch.tensor(
            [float(y.item() in self.target_class_set) for y in train_Y],
            device=train_Y.device,
            dtype=train_X.dtype,
        )

        if isinstance(input_transform, str):
            self.input_transform = None
        else:
            self.input_transform = input_transform

        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)
            _ = self.input_transform(train_X)
            self.input_transform.eval()

        transformed_train_X = self.input_transform(train_X) if self.input_transform else train_X

        if model is not None:
            self.model = model
        elif deep_gp and deep_kernel:
            self.model = DeepKernelDeepMixedGPClassificationModel(
                transformed_train_X,
                self.train_targets,
                cat_dims=cat_dims,
                list_hidden_dims=list_hidden_dims,
                num_inducing_points=max(num_inducing_points, 32),
            )
        elif deep_gp:
            self.model = DeepMixedGPClassificationModel(
                transformed_train_X,
                self.train_targets,
                cat_dims=cat_dims,
                list_hidden_dims=list_hidden_dims,
                num_inducing_points=max(num_inducing_points, 32),
            )
        elif deep_kernel:
            self.model = DeepKernelMixedClassificationModel(
                transformed_train_X, self.train_targets, cat_dims, num_inducing_points
            )
        else:
            self.model = GPClassificationMixedModel(
                transformed_train_X, self.train_targets, cat_dims, num_inducing_points
            )

        self.likelihood = likelihood if likelihood is not None else BernoulliLikelihood()
        self.to(device=train_X.device, dtype=train_X.dtype)

    def set_train_data(self, inputs=None, targets=None, strict: bool = True):
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
            self.train_inputs_raw = inputs
        if targets is not None:
            self.train_targets = targets

    def forward(self, X: Tensor):
        if isinstance(X, tuple):
            X = X[0]
        transformed_X = self.input_transform(X) if self.input_transform else X
        return self.model(transformed_X)

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs) -> SimpleBernoulliPosterior:
        self.eval()
        self.likelihood.eval()

        if isinstance(X, tuple):
            X = X[0]

        transformed_X = self.input_transform(X) if self.input_transform else X
        latent = self.model(transformed_X)
        preds = self.likelihood(latent)
        p = preds.mean
        var = preds.variance

        noise_model = getattr(self, "noise_model", None)
        if observation_noise and noise_model is not None:
            noise_in = transformed_X if getattr(self, "noise_model_uses_transformed_inputs", True) else X
            noise_log_var = noise_model.posterior(noise_in).mean
            noise_var = torch.exp(noise_log_var).clamp_min(1e-9)
            noise_var = noise_var.reshape_as(var)
            var = var + noise_var

        return SimpleBernoulliPosterior(mean=p, variance=var)

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])
