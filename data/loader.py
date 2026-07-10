"""Data loading (same logic as the notebook's data_load cell)."""
import numpy as np
import pandas as pd


def load_subject_meta(data_path, cols=("famid",)):
    """Return a DataFrame of subject-level metadata indexed by subject id.

    Used to attach grouping columns (e.g. famid for family-cluster structure) to
    the saved predictions without ever feeding them to the model as features.
    """
    raw = pd.read_excel(data_path, usecols=["id", *cols])
    return raw.set_index("id")[list(cols)]


def data_load(data_path, label_column="bpdp_4years", is_binary_classification=False,
              nan_subject_level_threshold=0.3, nan_feature_level_threshold=0.3):
    """Read the raw .xlsx, select the cohort, filter missingness, return (X, y).

    Inclusion criteria: mdd_baseline in {2, 3} AND bpdp_baseline == 1
    Excluded columns: id, famid, bpdp_baseline, and future columns ending in
    _4years / _10years.
    """
    assert label_column in ("bpdp_4years", "bpdp_10years"), \
        "Label column should be 'bpdp_4years' or 'bpdp_10years'"

    data = pd.read_excel(data_path)

    # Inclusion criteria: mdd baseline (2 or 3) -> bpdp baseline (1)
    mdd_baseline_data = data[
        ((data["mdd_baseline"] == 2) | (data["mdd_baseline"] == 3))
        & (data["bpdp_baseline"] == 1)
    ].reset_index(drop=True)

    y = mdd_baseline_data[label_column]
    mdd_baseline_data = mdd_baseline_data[~y.isna().values]
    # reset_index so y stays positionally aligned with x/uid below; without it the
    # boolean subject_mask (indexed 0..M-1) cannot align when NaN labels were dropped.
    y = y[~y.isna()].astype("int64").reset_index(drop=True)

    # Exclusion: columns ending in _4years / _10years + identifiers + single-value column
    excluded_columns = ["id", "famid", "bpdp_baseline"]
    excluded_columns += [
        col for col in mdd_baseline_data.columns
        if col.endswith("_4years") or col.endswith("_10years")
    ]
    x = mdd_baseline_data.drop(columns=excluded_columns).reset_index(drop=True)
    # Track the subject UID separately, positionally aligned to x. It is kept out
    # of x so it never enters the missingness fractions below (cohort unchanged).
    uid = mdd_baseline_data["id"].reset_index(drop=True)

    # Subject-level missingness filter
    subject_mask = x.isna().mean(axis=1) <= nan_subject_level_threshold
    y = y[subject_mask].astype("int64")
    x = x[subject_mask]
    uid = uid[subject_mask]

    # Feature-level missingness filter
    x = x.drop(columns=x.columns[x.isna().mean(axis=0) > nan_feature_level_threshold])

    if is_binary_classification:
        y = pd.Series(np.where(y.isin([2, 3]), 1, 0), name=y.name)

    # Attach the subject UID as the index so predictions can be traced per subject.
    # (Order is preserved through every step above, so this is a positional assignment.)
    x = x.set_axis(uid.to_numpy(), axis=0)
    x.index.name = "subject_id"
    y = y.set_axis(uid.to_numpy())
    y.index.name = "subject_id"

    return x, y
