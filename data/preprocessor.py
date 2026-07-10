"""CustomPreprocessor — per-column-type preprocessing transformer.

- binary + nominal : most-frequent imputation -> OneHotEncoder(drop='first')   [NOT scaled]
- ordinal + continuous + discrete : IterativeImputer(BayesianRidge) -> StandardScaler

Only the numeric block is standardized. Every estimator is still fit on the training
fold alone, so this stays leakage-free.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge


class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """Raw DataFrame -> numeric DataFrame ready for the model.

    X_for_categories: the DataFrame whose column value-sets seed the OneHotEncoder's
    `categories`, so the one-hot column order is stable across splits. The runners pass
    the full X here. That is deliberate and not label leakage: the admissible levels of a
    categorical variable are known a priori from the data dictionary (all 22 one-hot
    columns are plain binaries), no label or outcome statistic is read, and every
    estimator below is still fit on the training fold alone. Unseen levels at predict
    time are absorbed by handle_unknown='ignore'.
    """

    def __init__(self, binary_cols=[], nominal_cols=[], ordinal_cols=[],
                 continuous_cols=[], discrete_cols=[], random_state=67,
                 X_for_categories=None):
        # Note: per the sklearn clone() contract, __init__ must not modify the
        # parameters it receives (building a new object via `or []`/copy fails the
        # clone check) -> store them as-is.
        self.binary_cols = binary_cols
        self.nominal_cols = nominal_cols
        self.ordinal_cols = ordinal_cols
        self.continuous_cols = continuous_cols
        self.discrete_cols = discrete_cols
        self.random_state = random_state

        self.preprocessor = None
        self.feature_names_ = None
        self.one_hot_categories_ = None
        self.X_for_categories = X_for_categories
        if X_for_categories is not None:
            self._init_ohe_categories()

    def _init_ohe_categories(self):
        X = self.X_for_categories
        all_cols = list(X.columns)
        norm_binary_cols = [c for c in (self.binary_cols + self.nominal_cols)
                            if c in all_cols]
        if not norm_binary_cols:
            self.one_hot_categories_ = None
            return
        self.one_hot_categories_ = self._build_categories_for_ohe(X, norm_binary_cols)

    def _build_pipeline(self, X):
        all_cols = list(X.columns)
        norm_binary_cols = [c for c in (self.binary_cols + self.nominal_cols)
                            if c in all_cols]
        other_cols = [c for c in (self.ordinal_cols + self.continuous_cols + self.discrete_cols)
                      if c in all_cols]

        if norm_binary_cols and self.one_hot_categories_ is None:
            self.one_hot_categories_ = self._build_categories_for_ohe(X, norm_binary_cols)
        one_hot_categories = self.one_hot_categories_

        nominal_binary_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=True,
                categories=one_hot_categories if one_hot_categories is not None else "auto",
            )),
        ])

        other_pipe = Pipeline(steps=[
            ("imputer", IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=10,
                tol=1e-3,
                random_state=self.random_state,
                initial_strategy="most_frequent",
            )),
            # Scale ONLY the numeric block. One-hot columns are left as 0/1 so the L1/L2
            # penalty (and the sampler's neighbour distances) see them on their natural scale.
            ("scaler", StandardScaler()),
        ])

        transformers = []
        if norm_binary_cols:
            transformers.append(("nominal_binary", nominal_binary_pipe, norm_binary_cols))
        if other_cols:
            transformers.append(("others", other_pipe, other_cols))

        preprocess_by_type = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
            sparse_threshold=0.0,  # force dense output -> StandardScaler(with_mean=True) works
        )

        self.preprocessor = Pipeline(steps=[("preprocess_type", preprocess_by_type)])
        return self.preprocessor

    def _build_categories_for_ohe(self, X, norm_binary_cols):
        categories = []
        for col in norm_binary_cols:
            if col in X.columns:
                cats = X[col].dropna().unique()
                categories.append(sorted(cats))
        return categories

    def fit(self, X, y=None):
        preprocessor = self._build_pipeline(X)
        preprocessor.fit(X)
        self.feature_names_ = preprocessor.named_steps["preprocess_type"].get_feature_names_out()
        return self

    def transform(self, X):
        X_arr = self.preprocessor.transform(X)
        # OneHotEncoder may return a sparse matrix, so convert to dense
        if hasattr(X_arr, "toarray"):
            X_arr = X_arr.toarray()
        return pd.DataFrame(X_arr, columns=self.feature_names_, index=X.index)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
