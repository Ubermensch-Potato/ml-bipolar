"""TabPFN-3 under the SAME unbiased 5x5 CV as run_models.py.

The 212MB checkpoint is loaded ONCE and reused across folds (rebuilding it per
fold segfaults from repeated native loads). Inference settings are tuned per fold
with Optuna via set_params (no reload) -- objective = sens @ spec>=0.5 on valid.
Preprocessing is sophie's CustomPreprocessor, fit per fold (train only).

Tunable inference params (no train-time HP; TabPFN is a frozen PFN):
  n_estimators, softmax_temperature, balance_probabilities, average_before_softmax

Usage:
  python run_tabpfn.py                              # tuned, prefix=tabpfn_tuned
  python run_tabpfn.py --no-tune --prefix tabpfn    # defaults only (baseline)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TABPFN_DISABLE_MLX", "1")
import gc
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ml_code/ on path
from data.loader import data_load
from data.preprocessor import CustomPreprocessor
from tabpfn import TabPFNClassifier
from utils.paths import OUTPUTS as OUT, CONFIG as CONFIG_PATH, TABPFN_CKPT
from utils.metrics import (select_threshold as _thr, sens_spec as _ss,
                           sens_at_spec as _sens_at_spec, spec_at_sens as _spec_at_sens)


def _sampler(method, seed):
    """Over-sampler applied to the preprocessed TRAIN fold (None = no sampling)."""
    if method in (None, "builtin", "none"):
        return None
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    if method == "smote": return SMOTE(k_neighbors=5, random_state=seed)
    if method == "adasyn": return ADASYN(random_state=seed)
    if method == "smote_enn": return SMOTEENN(smote=SMOTE(k_neighbors=5, random_state=seed), random_state=seed)
    raise ValueError(f"unknown strategy '{method}'")


def tune_tabpfn(clf, Xtr, ytr, Xva, yva, seed, n_trials, obj="sens_at_spec"):
    """Per-fold Optuna over TabPFN inference settings (set_params, no reload)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(t):
        hp = dict(n_estimators=t.suggest_int("n_estimators", 4, 24),
                  softmax_temperature=t.suggest_float("softmax_temperature", 0.5, 1.5),
                  balance_probabilities=t.suggest_categorical("balance_probabilities", [True, False]),
                  average_before_softmax=t.suggest_categorical("average_before_softmax", [True, False]))
        clf.set_params(**hp).fit(Xtr, ytr)
        prob = clf.predict_proba(Xva)[:, 1]
        return _spec_at_sens(yva, prob, 0.75) if obj == "spec_at_sens" else _sens_at_spec(yva, prob, 0.5)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12, help="per-fold Optuna trials (0 = defaults only)")
    ap.add_argument("--tune", dest="tune", action="store_true", default=True)
    ap.add_argument("--no-tune", dest="tune", action="store_false")
    ap.add_argument("--prefix", default="tabpfn_tuned")
    ap.add_argument("--raw", action="store_true",
                    help="feed RAW features + categorical_features_indices (no one-hot; TabPFN's native mode)")
    ap.add_argument("--objective", choices=["sens_at_spec", "spec_at_sens"], default="spec_at_sens")
    ap.add_argument("--strategies", nargs="*", default=["builtin"],
                    help="builtin (no sampling) and/or smote / adasyn / smote_enn")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG_PATH)); c = cfg["columns"]; seed = 67
    X, y = data_load(cfg["data"]["path"], label_column="bpdp_10years", is_binary_classification=True)
    ppk = dict(binary_cols=c["binary"], ordinal_cols=c["ordinal"], continuous_cols=c["continuous"],
               discrete_cols=c["discrete"], random_state=seed)
    ckpt = TABPFN_CKPT

    n_trials = a.trials if a.tune else 0
    print(f"TabPFN-3 sweep | model loaded ONCE | outer 5x5 | "
          f"{'per-fold Optuna '+str(n_trials)+' trials (obj=sens@spec>=0.5)' if n_trials else 'defaults'}", flush=True)
    clf = TabPFNClassifier(model_path=ckpt, device="cpu")   # loaded once, reused

    cat_cols = num_cols = None
    if a.raw:
        cat_cols = [k for k in c["binary"] + c.get("nominal", []) if k in X.columns]
        num_cols = [k for k in c["ordinal"] + c["continuous"] + c["discrete"] if k in X.columns]
        clf.set_params(categorical_features_indices=list(range(len(cat_cols))))
        print(f"  RAW mode: {len(cat_cols)} categorical + {len(num_cols)} numeric (no one-hot)", flush=True)

    folds, preds = [], []
    for strat in a.strategies:
        samp = _sampler(strat, seed)
        ctr = 0
        for trainval, test in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed).split(X, y):
            rep, fold = ctr // 5, ctr % 5; ctr += 1
            ytv = y.iloc[trainval]
            try:
                tr, va = train_test_split(trainval, test_size=0.2, stratify=ytv, random_state=seed + ctr)
            except ValueError:
                tr, va = train_test_split(trainval, test_size=0.2, random_state=seed + ctr)
            if a.raw:
                from sklearn.impute import SimpleImputer
                numimp = SimpleImputer(strategy="median").fit(X.iloc[tr][num_cols]) if num_cols else None
                catimp = SimpleImputer(strategy="most_frequent").fit(X.iloc[tr][cat_cols]) if cat_cols else None
                def _tf(idx):
                    parts = []
                    if cat_cols: parts.append(catimp.transform(X.iloc[idx][cat_cols]))
                    if num_cols: parts.append(numimp.transform(X.iloc[idx][num_cols]))
                    return np.hstack(parts).astype(np.float32)
                Xtr, Xva, Xte = _tf(tr), _tf(va), _tf(test)
            else:
                pre = CustomPreprocessor(**ppk, X_for_categories=X).fit(X.iloc[tr], y.iloc[tr])
                Xtr = np.asarray(pre.transform(X.iloc[tr]), dtype=np.float32)
                Xva = np.asarray(pre.transform(X.iloc[va]), dtype=np.float32)
                Xte = np.asarray(pre.transform(X.iloc[test]), dtype=np.float32)
            ytr = y.iloc[tr].values
            if samp is not None:                                       # resample TRAIN only
                Xtr, ytr = samp.fit_resample(Xtr, ytr)
                Xtr = np.asarray(Xtr, dtype=np.float32); ytr = np.asarray(ytr).astype(int)

            best_hp = tune_tabpfn(clf, Xtr, ytr, Xva, y.iloc[va].values, seed, n_trials, a.objective) if n_trials else {}
            clf.set_params(**best_hp).fit(Xtr, ytr)
            tau = _thr(y.iloc[va].values, clf.predict_proba(Xva)[:, 1])
            prob = clf.predict_proba(Xte)[:, 1]
            pred = (prob >= tau).astype(int); yt = y.iloc[test].values
            se, sp = _ss(yt, pred)
            try: auc = roc_auc_score(yt, prob)
            except ValueError: auc = np.nan
            folds.append({"model": "TabPFN", "strategy": strat, "repeat": rep, "fold": fold,
                          "tau": round(tau, 4), "sens": se, "spec": sp, "auc": auc, "hp": str(best_hp)})
            for j, idx in enumerate(test):
                preds.append({"model": "TabPFN", "strategy": strat, "subject_id": X.index[idx],
                              "repeat": rep, "fold": fold, "tau": round(tau, 4),
                              "y_true": int(yt[j]), "prob": float(prob[j]), "pred": int(pred[j])})
            del Xtr, Xva, Xte; gc.collect()
            print(f"  [{strat}] r{rep}f{fold} tau={tau:.2f} sens={se:.2f} spec={sp:.2f} auc={auc:.3f}", flush=True)

    fdf = pd.DataFrame(folds)
    pd.DataFrame(preds).to_csv(os.path.join(OUT, f"{a.prefix}_predictions.csv"), index=False)
    fdf.to_csv(os.path.join(OUT, f"{a.prefix}_folds.csv"), index=False)
    rows = []
    for strat, g in fdf.groupby("strategy"):
        rows.append({"model": "TabPFN", "strategy": strat, "mean_sens": g.sens.mean(), "std_sens": g.sens.std(),
                     "mean_spec": g.spec.mean(), "mean_auc": g.auc.mean(), "tau_std": g.tau.std(), "n_folds": len(g)})
    sdf = pd.DataFrame(rows).sort_values("mean_auc", ascending=False)
    sdf.to_csv(os.path.join(OUT, f"{a.prefix}_summary.csv"), index=False)
    print(f"\nTabPFN-3 ({'tuned' if n_trials else 'default'}) — fold mean±std")
    for _, x in sdf.iterrows():
        ok = "OK" if x.mean_spec >= 0.5 else "FAIL"
        print(f"  TabPFN:{x.strategy:10s} sens {x.mean_sens:.3f}±{x.std_sens:.2f}  spec {x.mean_spec:.3f}  "
              f"AUC {x.mean_auc:.3f}  tau±{x.tau_std:.2f}  {ok}")
    print(f"-> outputs/{a.prefix}_summary.csv, {a.prefix}_folds.csv, {a.prefix}_predictions.csv")


if __name__ == "__main__":
    main()
