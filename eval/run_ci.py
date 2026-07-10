"""Confidence intervals for sens / spec / acc / AUC per (model, strategy).

Reads a *_predictions.csv from train/run_models.py or train/run_tabpfn.py
(5x5 => each subject predicted 5x), aggregates to the SUBJECT level
(mean prob, majority-vote pred over the repeats), then on the 160 subjects:
  - sens / spec / acc  -> Wilson score 95% CI (proportions)
  - AUC                -> bootstrap 95% CI (seeded; n from config evaluation.bootstrap_n)

Usage:
  python -m eval.run_ci --pred outputs/models_spec_at_sens_predictions.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ml_code/ on path
from utils.paths import CONFIG as CONFIG_PATH
from utils.metrics import wilson, boot_auc


def _out_path(pred, out):
    """Derive the CI path; never reuse the input path (that would overwrite it)."""
    if out:
        return out
    suffix = "_predictions.csv"
    stem = pred[:-len(suffix)] if pred.endswith(suffix) else os.path.splitext(pred)[0]
    return f"{stem}_ci.csv"


def _majority_vote(pv, prob):
    """Majority vote over the per-repeat hard predictions; ties broken by mean probability."""
    pred = (pv > 0.5).astype(int)
    tie = np.isclose(pv, 0.5)
    if tie.any():
        print(f"  warning: {int(tie.sum())} subject(s) had a tied vote (even #repeats); "
              f"breaking ties by mean probability >= 0.5", flush=True)
        pred[tie] = (prob[tie] >= 0.5).astype(int)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="a *_predictions.csv written by a train/ runner")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG_PATH)); seed = int(cfg.get("seed", 67))
    n_boot = int(cfg.get("evaluation", {}).get("bootstrap_n", 2000))

    out = _out_path(a.pred, a.out)
    if os.path.abspath(out) == os.path.abspath(a.pred):
        raise ValueError(f"--out would overwrite the input predictions file: {out}")

    df = pd.read_csv(a.pred)
    if df.empty:
        print(f"no predictions in {a.pred} — nothing to do"); return

    rows = []
    for (m, s), g in df.groupby(["model", "strategy"]):
        agg = g.groupby("subject_id").agg(y=("y_true", "first"), prob=("prob", "mean"),
                                          pv=("pred", "mean"))
        y = agg.y.values.astype(int); prob = agg.prob.values
        if len(np.unique(y)) < 2:
            print(f"  skip {m}:{s} — only one class among subjects"); continue
        pred = _majority_vote(agg.pv.values, prob)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        se = wilson(tp, tp + fn); sp = wilson(tn, tn + fp)
        ac = wilson(tp + tn, len(y)); au = boot_auc(y, prob, n_boot=n_boot, seed=seed)
        rows.append({"model": m, "strategy": s, "n": len(y),
                     "sens": se[0], "sens_lo": se[1], "sens_hi": se[2],
                     "spec": sp[0], "spec_lo": sp[1], "spec_hi": sp[2],
                     "acc": ac[0], "acc_lo": ac[1], "acc_hi": ac[2],
                     "auc": au[0], "auc_lo": au[1], "auc_hi": au[2]})
    if not rows:
        print(f"no (model, strategy) group in {a.pred} could be scored"); return

    r = pd.DataFrame(rows).sort_values("auc", ascending=False)
    r.to_csv(out, index=False)

    def c(p, lo, hi): return f"{p:.3f} [{lo:.3f},{hi:.3f}]"
    print(f"CI (subject-level, N={rows[0]['n']}) — Wilson (sens/spec/acc), bootstrap (AUC)   [{os.path.basename(a.pred)}]")
    print(f"{'model:strategy':22s}{'Sensitivity':>21}{'Specificity':>21}{'Accuracy':>21}{'AUC':>21}")
    print("-" * 106)
    for _, x in r.iterrows():
        print(f"{x.model+':'+x.strategy:22s}{c(x.sens,x.sens_lo,x.sens_hi):>21}"
              f"{c(x.spec,x.spec_lo,x.spec_hi):>21}{c(x.acc,x.acc_lo,x.acc_hi):>21}"
              f"{c(x.auc,x.auc_lo,x.auc_hi):>21}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
