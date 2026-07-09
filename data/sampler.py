"""CustomSampler — imbalanced-learn resampler factory (same as the notebook).

The combine methods (smote_enn, smote_tomek) require a nested smote_param dict.
"""


class CustomSampler:
    """Unified interface for the various resampling methods."""

    def __init__(self, method=None, random_state=67, **kwargs):
        self.method = method
        self.random_state = random_state
        self.kwargs = kwargs

    def _get_sampler(self):
        if self.method in ["none", None]:
            return None

        params = self.kwargs.copy()
        if self.method not in ["tomek", "nearmiss"]:
            params["random_state"] = self.random_state

        if self.method == "smote":
            from imblearn.over_sampling import SMOTE
            return SMOTE(**params)

        elif self.method == "adasyn":
            from imblearn.over_sampling import ADASYN
            return ADASYN(**params)

        elif self.method == "borderline_smote":
            from imblearn.over_sampling import BorderlineSMOTE
            return BorderlineSMOTE(**params)

        elif self.method == "random_over":
            from imblearn.over_sampling import RandomOverSampler
            return RandomOverSampler(**params)

        elif self.method == "random_under":
            from imblearn.under_sampling import RandomUnderSampler
            return RandomUnderSampler(**params)

        elif self.method == "tomek":
            from imblearn.under_sampling import TomekLinks
            return TomekLinks(**params)

        elif self.method == "smote_tomek":
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
            assert self.kwargs.get("smote_param") is not None, \
                "You need smote_param for smote_tomek method"
            smote_sampler = SMOTE(**self.kwargs["smote_param"])
            del params["smote_param"]
            return SMOTETomek(smote=smote_sampler, **params)

        elif self.method == "smote_enn":
            from imblearn.combine import SMOTEENN
            from imblearn.over_sampling import SMOTE
            assert self.kwargs.get("smote_param") is not None, \
                "You need smote_param for smote_enn method"
            smote_sampler = SMOTE(**self.kwargs["smote_param"])
            del params["smote_param"]
            return SMOTEENN(smote=smote_sampler, **params)

        else:
            raise ValueError(f"Unknown resampling method: {self.method}")
