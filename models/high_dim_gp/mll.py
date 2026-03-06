from typing import Any
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.model_list_gp_regression import ModelListGP

def fit_saas_nuts(model: Any, **nuts_kwargs: Any) -> Any:
    """SAASモデルをNUTSで学習（ModelListGPなら各モデルに適用）。"""
    if isinstance(model, ModelListGP):
        for m in model.models:
            fit_fully_bayesian_model_nuts(m, **nuts_kwargs)
    else:
        fit_fully_bayesian_model_nuts(model, **nuts_kwargs)
    return model