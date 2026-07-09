"""get_pipeline — build an ImbPipeline of preprocessor + sampler + classifier (same as the notebook).

Step names follow the notebook convention: 'scaler'=preprocessor, 'sampler'=resampler, 'clf'=model.
"""
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from data.preprocessor import CustomPreprocessor
from data.sampler import CustomSampler


def get_pipeline(model_class, model_params, preprocessor_params, X_for_categories,
                 sampler_params=None, random_state=67):
    if sampler_params is None:
        sampler_params = {"method": None, "random_state": random_state}

    s_params = sampler_params.copy()
    if "random_state" not in s_params:
        s_params["random_state"] = random_state

    method = sampler_params.get("method")

    if method == "smote_enn":
        if "smote_param" not in sampler_params:
            raise ValueError("You need smote_param when using smote_enn method")
        s_params["smote"] = SMOTE(**s_params["smote_param"])
        del s_params["smote_param"]
        del s_params["method"]
        sampler = SMOTEENN(**s_params)

    elif method == "smote_tomek":
        if "smote_param" not in sampler_params:
            raise ValueError("You need smote_param when using smote_tomek method")
        s_params["smote"] = SMOTE(**s_params["smote_param"])
        del s_params["smote_param"]
        del s_params["method"]
        sampler = SMOTETomek(**s_params)

    else:
        # For other methods, use the CustomSampler factory to build a real imblearn sampler (or None)
        sampler = CustomSampler(**sampler_params)._get_sampler()

    steps = [("scaler", CustomPreprocessor(**preprocessor_params,
                                           X_for_categories=X_for_categories))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", model_class(**model_params)))

    return ImbPipeline(steps=steps)
