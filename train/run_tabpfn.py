"""TabPFN-3 under the SAME unbiased 5x5 CV as run_models.py.

The 212MB checkpoint is loaded ONCE and reused across folds (rebuilding per fold
segfaults). Tuned per fold via set_params (no reload). TabPFN runs in a SEPARATE
process from XGBoost (shared-process libomp+torch clash -> SIGSEGV).

Structured to PARALLEL run_models.py section-by-section (diff the two files):
  [imports] [model-specific helpers + _fold] [main: args | data | setup | CV loop | outputs]
Only the model-specific block differs; the CV loop and write_outputs are byte-identical.

Usage:
  python -m train.run_tabpfn --strategies builtin smote adasyn smote_enn
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
from utils.paths import OUTPUTS as OUT, CONFIG as CONFIG_PATH, TABPFN_CKPT, resolve_data_path
from utils.metrics import (select_threshold as _thr, sens_spec as _ss,
                           sens_at_spec as _sens_at_spec, spec_at_sens as _spec_at_sens)
# ---- model-specific imports ----
from sklearn.impute import SimpleImputer
from data.preprocessor import CustomPreprocessor
from tabpfn import TabPFNClassifier

STRATEGIES = ["builtin", "smote", "adasyn", "smote_enn"]


# ======================= model-specific: sampler + HP search =======================
def _sampler(method, seed, smote_param):
    """Over-sampler applied to the preprocessed TRAIN fold (None = no sampling).

    smote_param comes from config.yaml so 'smote_enn' resamples IDENTICALLY to run_models.py
    (which builds its inner SMOTE from the same config); otherwise the two runners would
    oversample the minority to different ratios and the cross-model comparison is confounded.
    """
    if method in (None, "builtin", "none"):
        return None
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    if method == "smote": return SMOTE(k_neighbors=5, random_state=seed)
    if method == "adasyn": return ADASYN(random_state=seed)
    if method == "smote_enn":
        return SMOTEENN(smote=SMOTE(**smote_param, random_state=seed), random_state=seed)
    raise ValueError(f"unknown strategy '{method}'")


def tune_tabpfn(clf, Xtr, ytr, Xva, yva, seed, n_trials, obj, min_sens):
    """Per-fold Optuna over TabPFN inference settings via set_params, no reload (obj = spec@sens / sens@spec)."""
    if n_trials <= 0:
        return {}
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(t):
        hp = dict(n_estimators=t.suggest_int("n_estimators", 4, 24),
                  softmax_temperature=t.suggest_float("softmax_temperature", 0.5, 1.5),
                  balance_probabilities=t.suggest_categorical("balance_probabilities", [True, False]),
                  average_before_softmax=t.suggest_categorical("average_before_softmax", [True, False]))
        try:
            clf.set_params(**hp).fit(Xtr, ytr)
            prob = clf.predict_proba(Xva)[:, 1]
        except Exception:
            return 0.0          # a bad trial scores 0 instead of killing the whole fold
        return _spec_at_sens(yva, prob, min_sens) if obj == "spec_at_sens" else _sens_at_spec(yva, prob, 0.5)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ======================= shared: write outputs (IDENTICAL in run_models.py) =======================
def write_outputs(a, folds, preds):
    if not folds:
        print("every fold failed — no results to write"); return
    fdf = pd.DataFrame(folds)
    pd.DataFrame(preds).to_csv(os.path.join(OUT, f"{a.prefix}_predictions.csv"), index=False)
    fdf.to_csv(os.path.join(OUT, f"{a.prefix}_folds.csv"), index=False)
    rows = []
    for (m, st), g in fdf.groupby(["model", "strategy"]):
        rows.append({"model": m, "strategy": st, "mean_sens": g.sens.mean(), "std_sens": g.sens.std(),
                     "mean_spec": g.spec.mean(), "mean_auc": g.auc.mean(), "tau_std": g.tau.std(), "n_folds": len(g)})
    sdf = pd.DataFrame(rows).sort_values("mean_auc", ascending=False)
    sdf.to_csv(os.path.join(OUT, f"{a.prefix}_summary.csv"), index=False)
    print("\n" + "=" * 74)
    print(f"SWEEP  (unbiased {a.k}x{a.repeats}, fold mean+-std)  [{a.prefix}]")
    print(f"{'model:strategy':26s}{'mean_sens':>12}{'spec':>8}{'AUC':>8}{'tau_std':>9}  spec>=.5")
    for _, x in sdf.iterrows():
        ok = "OK" if x.mean_spec >= 0.5 else "FAIL"
        print(f"{x.model+':'+x.strategy:26s}{x.mean_sens:8.3f}+-{x.std_sens:.2f}{x.mean_spec:8.3f}"
              f"{x.mean_auc:8.3f}{x.tau_std:9.2f}  {ok}")
    print(f"-> outputs/{a.prefix}_summary.csv, {a.prefix}_folds.csv, {a.prefix}_predictions.csv")


def main():
    # ---- args ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--trials", type=int, default=12, help="per-fold Optuna trials (0 = no tuning)")
    ap.add_argument("--tune", dest="tune", action="store_true", default=True)
    ap.add_argument("--no-tune", dest="tune", action="store_false", help="defaults only (no Optuna)")
    ap.add_argument("--prefix", default="tabpfn_tuned", help="output filename prefix")
    ap.add_argument("--objective", choices=["sens_at_spec", "spec_at_sens"], default="spec_at_sens",
                    help="Optuna objective: spec@sens>=0.75 (main) / sens@spec>=0.5")
    ap.add_argument("--strategies", nargs="*", choices=STRATEGIES, default=["builtin"],
                    help="builtin (no sampling) and/or smote / adasyn / smote_enn")
    ap.add_argument("--raw", action="store_true",
                    help="feed RAW features + categorical_features_indices (no one-hot; TabPFN native)")
    a = ap.parse_args()

    # ---- data ----
    cfg = yaml.safe_load(open(CONFIG_PATH)); seed = int(cfg.get("seed", 67)); c = cfg["columns"]
    d = cfg["data"]
    X, y = data_load(resolve_data_path(d["path"]), label_column=d["label_column"],
                     is_binary_classification=d["is_binary_classification"],
                     nan_subject_level_threshold=d["nan_subject_level_threshold"],
                     nan_feature_level_threshold=d["nan_feature_level_threshold"])
    pp = {"binary_cols": c["binary"], "ordinal_cols": c["ordinal"], "continuous_cols": c["continuous"],
          "discrete_cols": c["discrete"], "random_state": seed}
    smote_param = dict(cfg["sampler"]["smote_param"])          # shared with run_models.py
    min_sens = float(cfg["threshold"]["min_target_sensitivity"])

    # ---- model setup (model-specific) ----
    clf = TabPFNClassifier(model_path=TABPFN_CKPT, device="cpu")   # loaded ONCE, reused
    sweep = [("TabPFN", s) for s in a.strategies]
    cat_cols = num_cols = None
    if a.raw:
        cat_cols = [k for k in c["binary"] + c.get("nominal", []) if k in X.columns]
        num_cols = [k for k in c["ordinal"] + c["continuous"] + c["discrete"] if k in X.columns]
        clf.set_params(categorical_features_indices=list(range(len(cat_cols))))

    def _preprocess(tr, va, test):
        if a.raw:
            numimp = SimpleImputer(strategy="median").fit(X.iloc[tr][num_cols]) if num_cols else None
            catimp = SimpleImputer(strategy="most_frequent").fit(X.iloc[tr][cat_cols]) if cat_cols else None
            def _tf(idx):
                parts = []
                if cat_cols: parts.append(catimp.transform(X.iloc[idx][cat_cols]))
                if num_cols: parts.append(numimp.transform(X.iloc[idx][num_cols]))
                return np.hstack(parts).astype(np.float32)
            return _tf(tr), _tf(va), _tf(test)
        pre = CustomPreprocessor(**pp, X_for_categories=X).fit(X.iloc[tr], y.iloc[tr])
        return (np.asarray(pre.transform(X.iloc[tr]), dtype=np.float32),
                np.asarray(pre.transform(X.iloc[va]), dtype=np.float32),
                np.asarray(pre.transform(X.iloc[test]), dtype=np.float32))

    def _fold(mname, strat, tr, va, test):
        Xtr, Xva, Xte = _preprocess(tr, va, test)
        ytr = y.iloc[tr].values
        samp = _sampler(strat, seed, smote_param)
        if samp is not None:                                       # resample TRAIN only
            Xtr, ytr = samp.fit_resample(Xtr, ytr)
            Xtr = np.asarray(Xtr, dtype=np.float32); ytr = np.asarray(ytr).astype(int)
        best_hp = tune_tabpfn(clf, Xtr, ytr, Xva, y.iloc[va].values, seed,
                              a.trials, a.objective, min_sens) if a.tune else {}
        clf.set_params(**best_hp).fit(Xtr, ytr)
        tau = _thr(y.iloc[va].values, clf.predict_proba(Xva)[:, 1], min_sens)
        prob = clf.predict_proba(Xte)[:, 1]
        del Xtr, Xva, Xte; gc.collect()
        return prob, tau, best_hp

    # ---- CV loop (IDENTICAL to run_models.py) ----
    tune_desc = f"per-fold Optuna {a.trials} trials (obj={a.objective})" if a.tune else "fixed HP"
    print(f"SWEEP  N={len(X)} (TabPFN loaded once) | outer {a.k}x{a.repeats} | {tune_desc} | thr on valid", flush=True)
    folds, preds = [], []
    for mname, strat in sweep:
        tag = f"{mname}:{strat}"; ctr = 0
        for trainval, test in RepeatedStratifiedKFold(n_splits=a.k, n_repeats=a.repeats,
                                                      random_state=seed).split(X, y):
            rep, fold = ctr // a.k, ctr % a.k; ctr += 1
            ytv = y.iloc[trainval]
            try:
                tr, va = train_test_split(trainval, test_size=a.val_frac, stratify=ytv, random_state=seed + ctr)
            except ValueError:
                tr, va = train_test_split(trainval, test_size=a.val_frac, random_state=seed + ctr)
            try:
                prob, tau, best_hp = _fold(mname, strat, tr, va, test)
            except Exception as e:
                print(f"  [{tag}] r{rep}f{fold} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True); continue
            pred = (prob >= tau).astype(int); yt = y.iloc[test].values
            se, sp = _ss(yt, pred)
            try: auc = roc_auc_score(yt, prob)
            except ValueError: auc = np.nan
            folds.append({"model": mname, "strategy": strat, "repeat": rep, "fold": fold,
                          "tau": round(tau, 4), "sens": se, "spec": sp, "auc": auc, "hp": str(best_hp)})
            for j, idx in enumerate(test):
                preds.append({"model": mname, "strategy": strat, "subject_id": X.index[idx],
                              "repeat": rep, "fold": fold, "tau": round(tau, 4),
                              "y_true": int(yt[j]), "prob": float(prob[j]), "pred": int(pred[j])})
            print(f"  [{tag}] r{rep}f{fold} tau={tau:.2f} sens={se:.2f} spec={sp:.2f} auc={auc:.3f}", flush=True)

    # ---- outputs (IDENTICAL to run_models.py) ----
    write_outputs(a, folds, preds)


if __name__ == "__main__":
    main()
