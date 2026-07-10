"""Reproduce the exact train/valid/test splits used by the train/ runners.

The splits depend only on the outer-CV counter, not on the model or strategy, so a
single list indexed by (repeat*k + fold) serves every (model, strategy) combination.
Keep this in lockstep with the CV loop in train/run_models.py and train/run_tabpfn.py.
"""
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


def fold_splits(X, y, seed, k=5, repeats=5, val_frac=0.2):
    """-> list of (train_idx, valid_idx, test_idx), one per fold, in run order."""
    splits, ctr = [], 0
    for trainval, test in RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats,
                                                  random_state=seed).split(X, y):
        ctr += 1
        try:
            tr, va = train_test_split(trainval, test_size=val_frac,
                                      stratify=y.iloc[trainval], random_state=seed + ctr)
        except ValueError:
            tr, va = train_test_split(trainval, test_size=val_frac, random_state=seed + ctr)
        splits.append((tr, va, test))
    return splits


def feature_map(X, cols):
    """Preprocessed column names -> raw column names, by construction order.

    CustomPreprocessor emits the one-hot block first (binary + nominal, drop='first',
    one column per 2-level binary) then the numeric block, so the mapping is positional.
    """
    ohe = [c for c in (cols["binary"] + cols.get("nominal", [])) if c in X.columns]
    num = [c for c in (cols["ordinal"] + cols["continuous"] + cols["discrete"]) if c in X.columns]
    return ohe + num
