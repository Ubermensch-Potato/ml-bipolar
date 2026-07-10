"""Sweep tabular models x imbalance strategies under an UNBIASED 5x5 CV.

Models: LR-L1, LR-L2, XGBoost, BalancedRF.  (TabPFN -> run_tabpfn.py, separate process)
Each fold: fit on train (sampler resamples train only), pick threshold on validation
(sens>=0.75 -> max spec), predict the independent test fold. No leakage.

Structured to PARALLEL run_tabpfn.py section-by-section (diff the two files):
  [imports] [model-specific helpers + _fold] [main: args | data | setup | CV loop | outputs]
Only the model-specific block differs; the CV loop and write_outputs are byte-identical.

Usage:
  python -m train.run_models --models LR_L1 XGBoost --repeats 5
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TABPFN_DISABLE_MLX", "1")
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
from utils.paths import OUTPUTS as OUT, CONFIG as CONFIG_PATH, resolve_data_path
from utils.metrics import (select_threshold as _thr, sens_spec as _ss,
                           sens_at_spec as _sens_at_spec, spec_at_sens as _spec_at_sens)
# ---- model-specific imports ----
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data.pipeline import get_pipeline

MODELS = ["LR_L1", "LR_L2", "XGBoost", "BalancedRF"]   # TabPFN -> train/run_tabpfn.py


# ======================= model-specific: HP search + build =======================
def _suggest(t, name, s):
    if s[0] == "float_log": return t.suggest_float(name, s[1], s[2], log=True)
    if s[0] == "float": return t.suggest_float(name, s[1], s[2])
    return t.suggest_int(name, s[1], s[2])


def model_specs(seed, neg, pos):
    from xgboost import XGBClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    spw = neg / pos
    return {
        "LR_L1": dict(cls=LogisticRegression,
                      base=dict(penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=seed),
                      wkey="class_weight", won="balanced",
                      strats=["weight", "smote", "adasyn", "smote_enn"],
                      space={"C": ("float_log", 1e-4, 10.0)}),
        "LR_L2": dict(cls=LogisticRegression,
                      base=dict(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=seed),
                      wkey="class_weight", won="balanced",
                      strats=["weight", "smote", "adasyn", "smote_enn"],
                      space={"C": ("float_log", 1e-4, 10.0)}),
        "XGBoost": dict(cls=XGBClassifier,
                        base=dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                                  eval_metric="logloss", n_jobs=2, verbosity=0),
                        wkey="scale_pos_weight", won=spw,
                        strats=["weight", "smote", "adasyn", "smote_enn"],
                        space={"max_depth": ("int", 2, 5), "learning_rate": ("float_log", 0.01, 0.3),
                               "n_estimators": ("int", 100, 300), "reg_lambda": ("float_log", 0.1, 10.0),
                               "subsample": ("float", 0.6, 1.0)}),
        "BalancedRF": dict(cls=BalancedRandomForestClassifier,
                           base=dict(n_estimators=300, random_state=seed, n_jobs=2,
                                     sampling_strategy="all", replacement=True, bootstrap=True),
                           wkey=None, won=None,
                           strats=["builtin", "smote", "adasyn", "smote_enn"],
                           space={"n_estimators": ("int", 200, 500), "max_features": ("float", 0.3, 1.0),
                                  "max_depth": ("int", 3, 15)}),
    }


def build(spec, strat, X, pp, seed, smote_param, extra=None):
    params = dict(spec["base"])
    if extra:
        params.update(extra)
    if strat == "weight" and spec["wkey"]:
        params[spec["wkey"]] = spec["won"]; sampler = {"method": None}
    elif strat in ("weight", "builtin"):
        sampler = {"method": None}
    else:
        sampler = {"method": strat, "random_state": seed}
        if strat == "smote_enn":
            sampler["smote_param"] = {**smote_param, "random_state": seed}
    return get_pipeline(spec["cls"], params, pp, X_for_categories=X, sampler_params=sampler, random_state=seed)


def tune_hp(spec, strat, X, y, pp, seed, smote_param, tr, va, n_trials, obj, min_sens):
    """Per-fold Optuna over sklearn HP on the valid holdout (obj = spec@sens / sens@spec / accuracy)."""
    space = spec.get("space")
    if not space or n_trials <= 0:
        return {}
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]

    def objective(t):
        hp = {n: _suggest(t, n, s) for n, s in space.items()}
        try:
            pipe = build(spec, strat, X, pp, seed, smote_param, extra=hp).fit(Xtr, ytr)
            if obj == "accuracy":
                return accuracy_score(yva.values, pipe.predict(Xva))
            if obj == "spec_at_sens":
                return _spec_at_sens(yva.values, pipe.predict_proba(Xva)[:, 1], min_sens)
            return _sens_at_spec(yva.values, pipe.predict_proba(Xva)[:, 1], 0.5)
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ======================= shared: write outputs (IDENTICAL in run_tabpfn.py) =======================
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
    ap.add_argument("--no-tune", dest="tune", action="store_false", help="use fixed HP (no Optuna)")
    ap.add_argument("--prefix", default="models", help="output filename prefix")
    ap.add_argument("--objective", choices=["sens_at_spec", "spec_at_sens", "accuracy"], default="spec_at_sens",
                    help="Optuna objective: spec@sens>=0.75 (main) / sens@spec>=0.5 / accuracy (sophie)")
    ap.add_argument("--models", nargs="*", choices=MODELS, default=MODELS)
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
    smote_param = dict(cfg["sampler"]["smote_param"])          # shared with run_tabpfn.py
    min_sens = float(cfg["threshold"]["min_target_sensitivity"])

    # ---- model setup (model-specific) ----
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    specs = model_specs(seed, neg, pos)
    sweep = [(m, s) for m in a.models for s in specs[m]["strats"]]

    def _fold(mname, strat, tr, va, test):
        spec = specs[mname]
        best_hp = tune_hp(spec, strat, X, y, pp, seed, smote_param, tr, va,
                          a.trials, a.objective, min_sens) if a.tune else {}
        pipe = build(spec, strat, X, pp, seed, smote_param, extra=best_hp).fit(X.iloc[tr], y.iloc[tr])
        tau = _thr(y.iloc[va].values, pipe.predict_proba(X.iloc[va])[:, 1], min_sens)
        prob = pipe.predict_proba(X.iloc[test])[:, 1]
        return prob, tau, best_hp

    # ---- CV loop (IDENTICAL to run_tabpfn.py) ----
    tune_desc = f"per-fold Optuna {a.trials} trials (obj={a.objective})" if a.tune else "fixed HP"
    print(f"SWEEP  N={len(X)} (pos {pos}/neg {neg}) | outer {a.k}x{a.repeats} | {tune_desc} | thr on valid", flush=True)
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

    # ---- outputs (IDENTICAL to run_tabpfn.py) ----
    write_outputs(a, folds, preds)


if __name__ == "__main__":
    main()
