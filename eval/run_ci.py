"""Confidence intervals for sens / spec / acc / AUC per (model, strategy).

Reads a *_predictions.csv from run_models.py (5x5 => each subject predicted 5x),
aggregates to the SUBJECT level (mean prob, majority-vote pred over the repeats),
then on the 160 subjects computes:
  - sens / spec / acc  -> Wilson score 95% CI (proportions)
  - AUC                -> bootstrap 95% CI (2000 resamples, seeded)

Usage:
  python run_ci.py                                  # models_tuned_predictions.csv
  python run_ci.py --pred outputs/models_acc_predictions.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ml_code/ on path
from utils.paths import OUTPUTS as OUT
from utils.metrics import wilson, boot_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(OUT, "models_tuned_predictions.csv"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.pred)

    rows = []
    for (m, s), g in df.groupby(["model", "strategy"]):
        agg = g.groupby("subject_id").agg(y=("y_true", "first"), prob=("prob", "mean"),
                                          pv=("pred", "mean"))
        y = agg.y.values.astype(int); prob = agg.prob.values
        pred = (agg.pv.values >= 0.5).astype(int)      # majority vote over the repeats
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        se = wilson(tp, tp + fn); sp = wilson(tn, tn + fp)
        ac = wilson(tp + tn, len(y)); au = boot_auc(y, prob)
        rows.append({"model": m, "strategy": s, "n": len(y),
                     "sens": se[0], "sens_lo": se[1], "sens_hi": se[2],
                     "spec": sp[0], "spec_lo": sp[1], "spec_hi": sp[2],
                     "acc": ac[0], "acc_lo": ac[1], "acc_hi": ac[2],
                     "auc": au[0], "auc_lo": au[1], "auc_hi": au[2]})
    r = pd.DataFrame(rows).sort_values("auc", ascending=False)
    out = a.out or a.pred.replace("_predictions.csv", "_ci.csv")
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
