"""Permutation feature importance for TabPFN-3 (its only option: no coef_, no gain).

Mirrors eval/run_importance.py: each fold's model is rebuilt from the inference settings
recorded in <prefix>_folds.csv, and the rebuilt test AUC is asserted against the stored one.

importance[j] = baseline_test_AUC - mean over n_repeats of AUC(test with column j shuffled)

Speed: TabPFN's cost is dominated by the in-context training set, which is reused across
queries. All n_features x n_repeats permuted copies of the test fold are stacked into ONE
predict_proba call instead of one call each (~12x faster).

Runs in a SEPARATE process from XGBoost (shared-process libomp+torch clash -> SIGSEGV).

Usage:
  python -m eval.run_importance_tabpfn --folds outputs/tabpfn_spec_at_sens_folds.csv --n-repeats 3
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TABPFN_DISABLE_MLX", "1")
import sys
import ast
import gc
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ml_code/ on path
from data.loader import data_load
from data.preprocessor import CustomPreprocessor
from utils.paths import CONFIG as CONFIG_PATH, TABPFN_CKPT, resolve_data_path
from utils.splits import fold_splits, feature_map
from train.run_tabpfn import _sampler
from tabpfn import TabPFNClassifier


def permutation_dauc(clf, Xte, yte, base, n_repeats, rng):
    """dAUC per column, via a single stacked predict_proba over all permuted copies."""
    n, p = Xte.shape
    big = np.empty((p * n_repeats * n, p), dtype=np.float32)
    for j in range(p):
        for k in range(n_repeats):
            Z = Xte.copy()
            Z[:, j] = rng.permutation(Z[:, j])
            big[(j * n_repeats + k) * n:(j * n_repeats + k + 1) * n] = Z
    probs = clf.predict_proba(big)[:, 1].reshape(p, n_repeats, n)
    return np.array([base - np.mean([roc_auc_score(yte, probs[j, k]) for k in range(n_repeats)])
                     for j in range(p)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, help="a *_folds.csv written by train/run_tabpfn.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-repeats", type=int, default=3, help="shuffles per feature per fold")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG_PATH)); seed = int(cfg.get("seed", 67)); c = cfg["columns"]
    d = cfg["data"]
    X, y = data_load(resolve_data_path(d["path"]), label_column=d["label_column"],
                     is_binary_classification=d["is_binary_classification"],
                     nan_subject_level_threshold=d["nan_subject_level_threshold"],
                     nan_feature_level_threshold=d["nan_feature_level_threshold"])
    pp = {"binary_cols": c["binary"], "ordinal_cols": c["ordinal"], "continuous_cols": c["continuous"],
          "discrete_cols": c["discrete"], "random_state": seed}
    smote_param = dict(cfg["sampler"]["smote_param"])

    splits = fold_splits(X, y, seed, a.k, a.repeats, a.val_frac)
    raw_names = feature_map(X, c)

    F = pd.read_csv(a.folds)
    out = a.out or a.folds.replace("_folds.csv", "_importance.csv")
    if os.path.abspath(out) == os.path.abspath(a.folds):
        raise ValueError(f"--out would overwrite the input folds file: {out}")

    clf = TabPFNClassifier(model_path=TABPFN_CKPT, device="cpu")   # loaded ONCE, reused
    print(f"TabPFN permutation importance | {len(F)} folds | n_repeats={a.n_repeats}", flush=True)

    rows = []
    for (mname, strat), g in F.groupby(["model", "strategy"]):
        samp = _sampler(strat, seed, smote_param)
        for _, r in g.iterrows():
            i = int(r["repeat"]) * a.k + int(r["fold"])
            tr, va, test = splits[i]
            pre = CustomPreprocessor(**pp, X_for_categories=X).fit(X.iloc[tr], y.iloc[tr])
            names = list(pre.feature_names_)
            Xtr = np.asarray(pre.transform(X.iloc[tr]), dtype=np.float32)
            Xte = np.asarray(pre.transform(X.iloc[test]), dtype=np.float32)
            ytr = y.iloc[tr].values; yte = y.iloc[test].values
            if samp is not None:
                Xtr, ytr = samp.fit_resample(Xtr, ytr)
                Xtr = np.asarray(Xtr, dtype=np.float32); ytr = np.asarray(ytr).astype(int)

            hp = ast.literal_eval(r["hp"]) if isinstance(r["hp"], str) and r["hp"].strip() else {}
            clf.set_params(**hp).fit(Xtr, ytr)
            base = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
            if not np.isclose(base, r["auc"], atol=1e-10, rtol=0):
                raise AssertionError(f"reconstruction mismatch TabPFN:{strat} r{r['repeat']}f{r['fold']}: "
                                     f"rebuilt AUC {base!r} != recorded {r['auc']!r}")

            rng = np.random.default_rng(seed + i)
            imp = permutation_dauc(clf, Xte, yte, base, a.n_repeats, rng)

            Ztr = np.asarray(pre.transform(X.iloc[tr]), dtype=float)
            ytr_f = y.iloc[tr].values.astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.array([np.corrcoef(Ztr[:, j], ytr_f)[0, 1] for j in range(Ztr.shape[1])])
            corr = np.nan_to_num(corr)

            for j, nm in enumerate(names):
                rows.append({"model": "TabPFN", "strategy": strat, "repeat": int(r["repeat"]),
                             "fold": int(r["fold"]), "feature": raw_names[j], "encoded": nm,
                             "importance": float(imp[j]), "method": "permutation_dauc",
                             "direction": float(np.sign(corr[j])), "baseline_auc": float(base)})
            del Xtr, Xte, Ztr; gc.collect()
            print(f"  [TabPFN:{strat}] r{int(r['repeat'])}f{int(r['fold'])} base_auc={base:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nreconstruction verified on all {len(F)} folds (test AUC == recorded AUC)")
    print(f"-> {out}   ({len(rows)} rows)")


if __name__ == "__main__":
    main()
