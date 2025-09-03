## 3-Fold Cross-Validation Guide (fold_script.py)

### Overview
Runs 3-fold cross-validation for all or selected models, producing two CSVs per model:
- kfold_standard.csv: metrics on standard multiclass validation split (no negative controls)
- kfold_one_vs_all.csv: metrics using one-vs-all with negative controls (same-superfamily)

Outputs are saved to:
`benchmark_results_same_sup/kfold/<level>/<model>/csv_files/`

### Prerequisites
- Python 3.8+
- Install dependencies (adjust if you already have them):
```bash
pip install numpy pandas scikit-learn matplotlib torch xgboost
```

### Data
- Input data: `data_source/data_new.csv`
- Superfamily map: `data_source/fam2supefamily.csv`
- Feature params and output root are from `benchmark_scripts/config_same_sup.py`.

### Usage
Run from the project root:
```bash
python benchmark_scripts/fold_script.py --level <subfamily|family> --model <all|model_key>
```

Examples:
```bash
# All models at subfamily level
python benchmark_scripts/fold_script.py --level subfamily --model all

# Single model at family level (e.g., random_forest)
python benchmark_scripts/fold_script.py --level family --model random_forest
```

### Models
Valid `model_key` values come from `benchmark_scripts/models.py` (e.g., random_forest, extra_trees, logistic_regression, knn, naive_bayes, neural_network, xgboost if installed).

### What the script does
1. Loads data and builds features using `feature_engineering.build_features`.
2. Creates 3 stratified folds once (fixed `random_state=42`). All models share the same splits per fold.
3. For each fold:
   - Trains each requested model (no model artifacts are saved).
   - Standard evaluation on the fold validation set.
   - Generates negative controls once for the fold and evaluates in one-vs-all mode for each model using the same negatives.
4. Writes two CSVs per model under `benchmark_results_same_sup/kfold/<level>/<model>/csv_files/`.

### CSV Format
Columns: `Iteration, Num_Classes, Accuracy(%), Precision(%), Recall(%), AUC(%), F1-Score(%)`
Rows: iterations 0,1,2 and a final `Average` row.

### Accuracy Equations
- Standard (multiclass): `Accuracy(%) = (1/N) * sum(1[y_i == y_hat_i]) * 100`
- One-vs-all (with negatives): `Accuracy(%) = (TP + TN) / (TP + TN + FP + FN) * 100`

### Reproducibility
- Folds are fixed via `StratifiedKFold(..., random_state=42)`.
- Negative control sampling uses a per-fold seed to be consistent across models.

### GPU (optional)
- If running `neural_network`, PyTorch will use GPU if available; otherwise CPU. Configure training hyperparameters in `config_same_sup.py`.

### Troubleshooting
- Import errors: ensure packages are installed in your active environment.
- XGBoost class error: the script remaps per-fold labels to contiguous indices internally; no action needed. To skip XGBoost, run with another `--model`.
- Slow runs: start with a single model (e.g., `--model random_forest`).


