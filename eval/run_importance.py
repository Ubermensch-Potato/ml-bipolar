"""Per-fold feature importance for LR_L1 / LR_L2 / XGBoost / BalancedRF.

Each fold's model is rebuilt from the hyper-parameters recorded in <prefix>_folds.csv,
so no Optuna search is repeated. The rebuilt model's test AUC is asserted to equal the
AUC stored in that file -- if the reconstruction were wrong, the run aborts.

Importance is the model's NATIVE measure (TabPFN has none; see run_importance_tabpfn.py):
  LR_L1 / LR_L2 -> coef_                    (signed; numeric = per 1 SD, binary = 0->1)
  XGBoost       -> feature_importances_     (gain, normalized to sum 1 per fold)
  BalancedRF    -> feature_importances_     (Gini impurity decrease; biased to continuous)

Units differ per model, so importances are NOT comparable across model families in
absolute terms -- compare ranks, or normalize per (model, strategy).

Usage:
  python -m eval.run_importance --folds outputs/models_spec_at_sens_folds.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import ast
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ml_code/ on path
from data.loader import data_load
from utils.paths import CONFIG as CONFIG_PATH, resolve_data_path
from utils.splits import fold_splits, feature_map
from train.run_models import model_specs, build


def _native(mname, clf, n_features):
    """(importance vector, method name). Signed for LR, non-negative for the trees."""
    if mname.startswith("LR_"):
        return np.asarray(clf.coef_).ravel(), "coef"
    if mname == "XGBoost":
        clf.importance_type = "gain"          # read-only switch; does not refit
        return np.asarray(clf.feature_importances_, dtype=float), "gain"
    return np.asarray(clf.feature_importances_, dtype=float), "gini"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, help="a *_folds.csv written by train/run_models.py")
    ap.add_argument("--out", default=None)
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

    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    specs = model_specs(seed, neg, pos)
    splits = fold_splits(X, y, seed, a.k, a.repeats, a.val_frac)
    raw_names = feature_map(X, c)

    F = pd.read_csv(a.folds)
    out = a.out or a.folds.replace("_folds.csv", "_importance.csv")
    if os.path.abspath(out) == os.path.abspath(a.folds):
        raise ValueError(f"--out would overwrite the input folds file: {out}")

    rows = []
    for (mname, strat), g in F.groupby(["model", "strategy"]):
        for _, r in g.iterrows():
            i = int(r["repeat"]) * a.k + int(r["fold"])
            tr, va, test = splits[i]
            hp = ast.literal_eval(r["hp"]) if isinstance(r["hp"], str) and r["hp"].strip() else {}
            pipe = build(specs[mname], strat, X, pp, seed, smote_param, extra=hp).fit(X.iloc[tr], y.iloc[tr])

            prob = pipe.predict_proba(X.iloc[test])[:, 1]
            auc = roc_auc_score(y.iloc[test].values, prob)
            if not np.isclose(auc, r["auc"], atol=1e-10, rtol=0):
                raise AssertionError(f"reconstruction mismatch {mname}:{strat} r{r['repeat']}f{r['fold']}: "
                                     f"rebuilt AUC {auc!r} != recorded {r['auc']!r}")

            pre = pipe.named_steps["scaler"]
            names = list(pre.feature_names_)
            imp, method = _native(mname, pipe.named_steps["clf"], len(names))
            # descriptive direction: sign of each feature's correlation with y on the TRAIN fold
            Ztr = pre.transform(X.iloc[tr]).values
            ytr = y.iloc[tr].values.astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.array([np.corrcoef(Ztr[:, j], ytr)[0, 1] for j in range(Ztr.shape[1])])
            corr = np.nan_to_num(corr)

            for j, nm in enumerate(names):
                rows.append({"model": mname, "strategy": strat, "repeat": int(r["repeat"]),
                             "fold": int(r["fold"]), "feature": raw_names[j], "encoded": nm,
                             "importance": float(imp[j]), "method": method,
                             "direction": float(np.sign(corr[j])), "baseline_auc": float(auc)})
        print(f"  done {mname}:{strat}", flush=True)

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nreconstruction verified on all {len(F)} folds (test AUC == recorded AUC)")
    print(f"-> {out}   ({len(rows)} rows)")


if __name__ == "__main__":
    main()
