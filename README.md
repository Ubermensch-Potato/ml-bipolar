# ml_code — Tabular ML benchmark (unbiased 5×5 CV + CI)

Predict **bipolar-disorder conversion** from adolescent tabular EHR data across several
tabular ML models × imbalance strategies, evaluated with a **leakage-free (unbiased) 5×5
cross-validation** and reported with **subject-level confidence intervals (CI)**.
Fully reproducible under **seed 67**.

- **Cohort**: N=160 (43 positive / 117 negative, ~27% imbalance), label `bpdp_10years`.
- **Models**: LR (L1 / L2), XGBoost, BalancedRandomForest, TabPFN-3.
- **Imbalance strategies**: `weight` (model-native weighting) / `smote` / `adasyn` /
  `smote_enn` / `builtin` (model's internal handling).
- **Protocol**: every fold splits into **train / valid / test** — HP and the decision
  threshold are chosen on **valid**, the test fold is predicted **independently**.

---

## 1. Layout (package hierarchy)

```
ml_code/
├── config.yaml            # column-type definitions + dataset path
├── requirements.txt       # pinned package versions (Python 3.11)
├── download_tabpfn.py     # download the TabPFN-3 checkpoint (see §3)
├── data/                  # data loading + preprocessing pipeline
│   ├── loader.py          #   data_load, load_subject_meta
│   ├── preprocessor.py    #   CustomPreprocessor (impute · scale · one-hot)
│   ├── sampler.py         #   CustomSampler (SMOTE / ADASYN / ...)
│   └── pipeline.py        #   get_pipeline (preprocessor → sampler → model)
├── utils/                 # shared utilities
│   ├── paths.py           #   ROOT / CONFIG / OUTPUTS / TABPFN_CKPT
│   └── metrics.py         #   threshold · sens_at_spec · spec_at_sens · Wilson · bootstrap
├── train/                 # training sweeps (entry points)
│   ├── run_models.py      #   LR · XGBoost · BalancedRF, per-fold Optuna, unbiased 5×5
│   └── run_tabpfn.py      #   TabPFN-3 (model loaded once + set_params tuning)
├── eval/                  # evaluation
│   └── run_ci.py          #   predictions → subject aggregation (160) → Wilson + bootstrap CI
├── tabpfn_models/         # TabPFN-3 checkpoint (see §3)
└── outputs/               # result CSVs
```

> All paths are resolved from the **ml_code/ root** in `utils/paths.py`, so results always
> land in `ml_code/outputs/` regardless of which sub-package you launch from.

---

## 2. Environment setup

```bash
conda create -n bipolar-ml python=3.11 -y
conda run -n bipolar-ml python -m pip install -r requirements.txt
conda activate bipolar-ml
```

Key packages: `scikit-learn 1.8`, `xgboost 3.2`, `imbalanced-learn 0.14`, `optuna 4.9`,
`tabpfn 8.0.8` + `torch 2.11` + `huggingface-hub` (to download the TabPFN weights).

---

## 3. TabPFN-3 checkpoint (one-time)

TabPFN-3 weights live in a **gated HuggingFace repo** and require authentication to
download (research / benchmarking use is permitted by the license).

1. Accept the terms at https://huggingface.co/Prior-Labs/tabpfn_3 (**Access repository**).
2. Create a **read token** at https://huggingface.co/settings/tokens.
3. Download:
```bash
HF_TOKEN=<hf_read_token> python download_tabpfn.py
```
→ `tabpfn_models/tabpfn-v3-classifier-v3_default.ckpt` (~213 MB). `train/run_tabpfn.py`
loads this local file via `model_path`, so **no token is needed at run time**.

---

## 4. Running

Run from `ml-bipolar/` in **`python -m` module form** (so package imports resolve).
This is the **exact sequence used to produce the reported results** — two objectives
(`sens_at_spec`, `spec_at_sens`), each written under a per-objective prefix.

```bash
cd ml-bipolar
export KMP_DUPLICATE_LIB_OK=TRUE      # avoids the XGBoost(libomp)+torch macOS segfault

# 1) Tabular models — per-fold Optuna (unbiased 5x5), one run per objective
python -m train.run_models --models LR_L1 LR_L2 XGBoost BalancedRF \
  --repeats 5 --trials 20 --objective sens_at_spec --prefix models_sens_at_spec
python -m train.run_models --models LR_L1 LR_L2 XGBoost BalancedRF \
  --repeats 5 --trials 20 --objective spec_at_sens --prefix models_spec_at_sens

# 2) TabPFN-3 — SEPARATE process (same process as XGBoost segfaults, libomp clash)
python -m train.run_tabpfn --strategies builtin smote adasyn smote_enn \
  --trials 20 --objective sens_at_spec --prefix tabpfn_sens_at_spec
python -m train.run_tabpfn --strategies builtin smote adasyn smote_enn \
  --trials 20 --objective spec_at_sens --prefix tabpfn_spec_at_sens

# 3) subject-level CI (sens/spec/acc = Wilson, AUC = bootstrap)
python -m eval.run_ci --pred outputs/models_sens_at_spec_predictions.csv
python -m eval.run_ci --pred outputs/models_spec_at_sens_predictions.csv
python -m eval.run_ci --pred outputs/tabpfn_sens_at_spec_predictions.csv
python -m eval.run_ci --pred outputs/tabpfn_spec_at_sens_predictions.csv
```
> Later, once performance is in, keep only **one** objective — run just the two
> `*_sens_at_spec` (or `*_spec_at_sens`) commands plus their two CI lines.
> Direct execution (`python train/run_models.py ...`) also works — each script adds
> `ml-bipolar/` to `sys.path`.
>
> This produces, under `outputs/`, one `{summary,folds,predictions,ci}.csv` set per prefix:
> `models_sens_at_spec_*`, `models_spec_at_sens_*`, `tabpfn_sens_at_spec_*`, `tabpfn_spec_at_sens_*`.

### `train/run_models.py` options
| option | default | description |
|--------|---------|-------------|
| `--models` | all 4 | which models (run TabPFN via `train/run_tabpfn.py`) |
| `--repeats` | 5 | outer-CV repeats (k=5 fixed → 5×5) |
| `--trials` | 20 | per-fold Optuna trials (0 = no tuning) |
| `--objective` | `sens_at_spec` | `sens_at_spec` (sens@spec≥0.5) / `spec_at_sens` (spec@sens≥0.75, matches the threshold rule) / `accuracy` (sophie's original) |
| `--no-tune` | — | run with fixed HP |
| `--prefix` | `models` | output-file prefix |

### `train/run_tabpfn.py` options
| option | default | description |
|--------|---------|-------------|
| `--strategies` | `builtin` | any of `builtin` / `smote` / `adasyn` / `smote_enn` |
| `--trials` | 20 | per-fold Optuna trials over TabPFN inference settings |
| `--objective` | `sens_at_spec` | `sens_at_spec` or `spec_at_sens` |
| `--raw` | off | feed RAW features + `categorical_features_indices` (no one-hot) |
| `--no-tune` | — | defaults only (no tuning) |
| `--prefix` | `tabpfn_tuned` | output-file prefix |

---

## 5. Objectives · threshold · CI

- **HP-tuning objective** (`--objective`)
  - `sens_at_spec` — maximize **sensitivity** subject to specificity ≥ 0.5 (clinical goal).
  - `spec_at_sens` — maximize **specificity** subject to sensitivity ≥ 0.75 (matches the
    deployment threshold rule).
  - `accuracy` — maximize accuracy at the 0.5 cutoff (**sophie's original**; sacrifices
    specificity under imbalance).
  - *TabPFN* is a frozen prior-fitted network, so it has **no train-time HP**; the tuner
    searches its inference settings (`n_estimators`, `softmax_temperature`,
    `balance_probabilities`, `average_before_softmax`).
- **Threshold rule** (all objectives): on valid, **sensitivity ≥ 0.75 → max specificity**
  (Youden's J fallback).
- **CI** (`eval/run_ci.py`, subject level, N=160)
  - sens / spec / acc → **Wilson score** 95% CI.
  - AUC → **bootstrap** 95% CI (2000 resamples, seed 67).
  - The 5 predictions per subject (from the 5 repeats) are aggregated by **mean probability
    and majority vote**.

---

## 6. Pseudocode

`perf` denotes the bundle of performance metrics (sens / spec / acc / auc); the concrete
metrics are encapsulated in `evaluate` / `evaluate_with_CI`.

### `train/run_models.py`
```
X, y = data_load(); seed = 67
for model in models:
    for strat in strategies[model]:
        for (trainval, test) in RepeatedStratifiedKFold(5, 5).split(X, y):
            tr, va = stratified_split(trainval, 0.2)
            hp   = tune_hp(model, strat, tr, va, objective)
            pipe = build(model, strat, hp).fit(X[tr], y[tr])
            tau  = select_threshold(y[va], predict(va))
            prob = predict(test); pred = prob >= tau
            perf = evaluate(y[test], prob, pred)
            folds.append(model, strat, repeat, fold, tau, perf, hp)
            for subject in test: preds.append(model, strat, subject, repeat, fold, prob, pred, y_true)
summary = folds.groupby(model, strat).mean(perf)
write folds.csv, predictions.csv, summary.csv
```

### `train/run_tabpfn.py`
```
clf = TabPFNClassifier(ckpt, cpu)                              [loaded ONCE]
for strat in strategies:
    for (trainval, test) in RepeatedStratifiedKFold(5, 5).split(X, y):
        tr, va = stratified_split(trainval, 0.2)
        Xtr, Xva, Xte = preprocess.fit(tr).transform(tr, va, test)
        if strat != builtin: Xtr, ytr = sampler.fit_resample(Xtr, ytr)
        hp  = tune_tabpfn(clf, Xtr, Xva, objective)            [set_params only]
        clf.set_params(hp).fit(Xtr, ytr)
        tau = select_threshold(y[va], clf.predict(Xva))
        prob = clf.predict(Xte); pred = prob >= tau
        perf = evaluate(y[test], prob, pred)
        folds.append(TabPFN, strat, repeat, fold, tau, perf, hp)
        for subject in test: preds.append(TabPFN, strat, subject, repeat, fold, prob, pred, y_true)
summary = folds.groupby(strat).mean(perf)
write folds.csv, predictions.csv, summary.csv                 [separate process from XGBoost]
```

### `eval/run_ci.py`
```
df = read(predictions.csv)
for (model, strat) in groupby:
    per_subject = group.by(subject).agg(prob=mean, pred=majority_vote)   [5 repeats -> 1, N=160]
    tn, fp, fn, tp = confusion_matrix(y, pred)
    sens = wilson(tp, tp+fn); spec = wilson(tn, tn+fp); acc = wilson(tp+tn, N)   [Wilson score 95% CI]
    auc  = bootstrap_auc(y, prob, n_boot=2000, seed=67)                          [percentile 95% CI]
    rows.append(model, strat, sens, spec, acc, auc)                              [each = point, lo, hi]
write ci.csv

wilson(k, n, z=1.96):                                    [closed-form CI for a proportion k/n]
    p = k/n
    center = (p + z^2/2n) / (1 + z^2/n)
    half   = z * sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)
    return p, clip(center - half), clip(center + half)

bootstrap_auc(y, prob, n_boot, seed):                    [nonparametric CI for AUC]
    for i in 1..n_boot: resample N subjects with replacement; collect roc_auc(y*, prob*)
    return roc_auc(y, prob), percentile(2.5%), percentile(97.5%)
```
> `evaluate_with_CI` = **Wilson score interval** for the proportions (sens / spec / acc)
> and **bootstrap percentile CI** for AUC — both at the subject level (N=160).

### Shared invariants
```
fit(preprocess, sampler, HP, threshold) uses TRAIN/VALID only; TEST predicted independently -> no leakage
fold: perf -> folds.csv | raw prob -> predictions.csv | mean -> summary.csv
run_ci: recompute subject-level perf + CI from predictions.csv probs only -> ci.csv
seed = 67 everywhere -> reproducible
```

---

## 7. Outputs (`outputs/`)

| file | contents |
|------|----------|
| `<prefix>_summary.csv` | per (model, strategy) fold mean ± std |
| `<prefix>_folds.csv` | per-fold sens/spec/auc/tau + chosen HP |
| `<prefix>_predictions.csv` | one row per prediction = (model, strategy, subject, repeat, fold) |
| `<prefix>_ci.csv` | subject-level point estimate + 95% CI (sens/spec/acc/auc) |

---

## 8. Reproducibility

- **seed 67** everywhere: `random_state` of LR (saga) / XGBoost / BalancedRF, Optuna
  `TPESampler`, SMOTE/ADASYN/SMOTE-ENN, and the bootstrap `default_rng`.
- Running the same command twice yields **bit-identical** `predictions/folds/summary/ci`
  (verified for LR, XGBoost, TabPFN, and the CI step).
- TabPFN is deterministic — in-context, CPU inference, no RNG.

---

## 9. Notes

- **`KMP_DUPLICATE_LIB_OK=TRUE`** — avoids a macOS segfault when XGBoost (libomp) and torch
  are loaded in the same process.
- **Run TabPFN only via `train/run_tabpfn.py` (separate process)** — putting it in
  `run_models.py` alongside XGBoost segfaults (confirmed, order-independent). TabPFN *is*
  Optuna-tuned there (inference settings); merge its CI into each objective's table.
- The dataset path in `config.yaml` is absolute — edit it if the data lives elsewhere.
- `tabpfn_models/` may be a symlink (`../py_codes/tabpfn_models`) in this repo — copy the
  real checkpoint in for a standalone deployment.
