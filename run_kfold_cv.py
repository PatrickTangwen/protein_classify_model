import argparse
import os
import json
import csv
import numpy as np
from typing import List, Dict, Any

from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# Use the same-supefamily results tree by default as requested
import config_same_sup as config

from data_loader import load_protein_data
from feature_engineering import build_features
from models import MODELS

# Ensure non-interactive Matplotlib backend to avoid Tkinter issues in worker contexts
os.environ.setdefault('MPLBACKEND', 'Agg')


def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray) -> Dict[str, float]:
    """Compute fold metrics.

    - Accuracy by (TP+TN)/(TP+TN+FP+FN) using the overall confusion matrix
    - Macro precision/recall/F1 over classes
    - AUC treating correctness (y_true==y_pred) as positive and confidence as score
    """
    # Confusion-based accuracy (equivalent to standard accuracy for multi-class)
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp_total = np.trace(cm)
    fp_total = cm.sum(axis=0) - np.diag(cm)
    fn_total = cm.sum(axis=1) - np.diag(cm)
    tn_total = cm.sum() - (tp_total + fp_total.sum() + fn_total.sum())
    denom = tp_total + tn_total + fp_total.sum() + fn_total.sum()
    acc_conf = float((tp_total + tn_total) / denom) if denom > 0 else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # AUC on correctness vs confidence (refer evaluation_same_sup.py style)
    y_true_binary = (y_true == y_pred).astype(int)
    if len(np.unique(y_true_binary)) == 2 and confidences is not None and len(confidences) == len(y_true):
        fpr, tpr, _ = roc_curve(y_true_binary, confidences)
        auc_val = float(auc(fpr, tpr))
    else:
        auc_val = float('nan')

    return {
        'accuracy': acc_conf,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': auc_val,
    }


def run_kfold(level: str, model: str, n_splits: int, shuffle: bool, random_state: int, cv_mode: str) -> None:
    print("=" * 80)
    print(f"K-Fold Cross-Validation - {level.capitalize()} Level | Folds={n_splits}")
    print("=" * 80)

    # 1) Load data
    print("\n=== Step 1: Loading Data ===")
    df = load_protein_data(config.PROTEIN_DATA_PATH, level=level)
    print(f"Dataset shape: {df.shape}")

    # 2) Build features (reuse existing engineering)
    print("\n=== Step 2: Building Features ===")
    X_norm, y, label_encoder, domain_vocab, feature_stats = build_features(
        df,
        level=level,
        max_domains=config.MAX_DOMAINS,
        max_separators=config.MAX_SEPARATORS,
        evalue_threshold=config.EVALUE_THRESHOLD,
    )
    print(f"Feature matrix shape: {X_norm.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    # Undo global normalization to avoid leakage, then re-normalize per fold
    feature_mean, feature_std = feature_stats
    X_raw = X_norm * feature_std + feature_mean

    # 3) K-Fold split setup with stratification when feasible
    # If any class has fewer than n_splits samples, fall back to unstratified K-Fold.
    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
    use_stratified = min_class_count >= 2 and n_splits <= min_class_count
    if not use_stratified:
        print(
            f"Warning: StratifiedKFold not feasible (min class count = {min_class_count}, folds = {n_splits}). "
            f"Falling back to unstratified KFold. Consider reducing --folds or merging rare classes."
        )
        splitter = KFold(n_splits=max(2, min(n_splits, len(y))), shuffle=shuffle, random_state=random_state)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Select models to run
    models_to_run = list(MODELS.keys()) if model == 'all' else [model]
    for m in models_to_run:
        if m not in MODELS:
            print(f"Warning: Model '{m}' not found. Skipping.")
    models_to_run = [m for m in models_to_run if m in MODELS]
    if not models_to_run:
        print("No valid models selected. Exiting.")
        return

    print(f"Models to run: {models_to_run}")

    # Base results directory: benchmark_results_same_sup/kfold/<level>/<model>/
    base_kfold_dir = os.path.join(config.RESULTS_DIR, 'kfolds', level)
    os.makedirs(base_kfold_dir, exist_ok=True)

    for model_name in models_to_run:
        print("\n" + "=" * 60)
        print(f"Processing model (K-Fold): {model_name}")
        print("=" * 60)

        model_info = MODELS[model_name].copy()
        model_output_root = os.path.join(base_kfold_dir, model_name)
        os.makedirs(model_output_root, exist_ok=True)

        fold_metrics: List[Dict[str, Any]] = []
        table_rows: List[Dict[str, Any]] = []  # for CSV table

        # Option A: Use scikit-learn cross_validate for sklearn models
        # XGBoost expects contiguous class labels per fit; prefer manual per-fold for it
        is_xgb = (model_name == 'xgboost')

        if cv_mode == 'sklearn' and model_info['type'] == 'sklearn' and not is_xgb:
            print("Using scikit-learn cross_validate with StratifiedKFold and StandardScaler pipeline.")
            from sklearn.base import clone
            estimator = clone(model_info['model'])
            pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=True), estimator)

            scoring = {'acc': 'accuracy', 'f1': 'f1_macro'}
            cv = splitter
            cv_result = cross_validate(
                pipeline,
                X_raw,
                y,
                scoring=scoring,
                cv=cv,
                return_train_score=False,
                n_jobs=1,
            )

            for i in range(n_splits):
                fold_metrics.append({
                    'fold': i + 1,
                    'accuracy': float(cv_result['test_acc'][i]),
                    'macro_f1': float(cv_result['test_f1'][i]),
                    'fit_time': float(cv_result['fit_time'][i]),
                    'score_time': float(cv_result['score_time'][i]),
                })

            print("Fold scores (acc):", [f"{m['accuracy']:.4f}" for m in fold_metrics])
            print("Fold scores (f1):", [f"{m['macro_f1']:.4f}" for m in fold_metrics])

            # Extra per-fold loop to compute confidences and AUC for the table CSV
            print("Computing detailed per-fold metrics for CSV table...")
            for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_raw, y), start=1):
                from sklearn.base import clone as skl_clone
                est = skl_clone(model_info['model'])
                pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), est)
                pipe.fit(X_raw[train_idx], y[train_idx])
                preds = pipe.predict(X_raw[val_idx])
                if hasattr(pipe, 'predict_proba'):
                    probs = pipe.predict_proba(X_raw[val_idx])
                    confidences = np.max(probs, axis=1)
                else:
                    # Fallback if predict_proba is unavailable
                    confidences = np.ones_like(preds, dtype=float)

                metrics_dict = _compute_fold_metrics(y[val_idx], preds, confidences)
                table_rows.append({
                    'iteration': fold_index - 1,
                    'num_classes': int(len(label_encoder.classes_)),
                    **metrics_dict,
                })

        else:
            # Option B: Manual loop (supports sklearn and pytorch via existing training pipeline)
            # Lazy imports to avoid triggering Matplotlib/Tkinter unless needed
            from training import train_and_evaluate_model
            from evaluation import get_predictions

            for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_raw, y), start=1):
                print(f"\n--- Fold {fold_index}/{n_splits} ---")

                # Per-fold normalization to prevent leakage
                train_mean = X_raw[train_idx].mean(axis=0)
                train_std = X_raw[train_idx].std(axis=0)
                train_std[train_std == 0] = 1

                X_train = (X_raw[train_idx] - train_mean) / train_std
                y_train = y[train_idx]
                X_val = (X_raw[val_idx] - train_mean) / train_std
                y_val = y[val_idx]

                # Per-fold output directory
                fold_dir = os.path.join(model_output_root, f"fold_{fold_index}")
                os.makedirs(fold_dir, exist_ok=True)

                # Clone sklearn estimators to avoid state carry-over
                if model_info['type'] == 'sklearn':
                    from sklearn.base import clone
                    model_info['model'] = clone(model_info['model'])

                # For XGBoost, reindex training labels to a contiguous 0..K-1 set per fold
                if is_xgb:
                    fold_le = LabelEncoder().fit(y_train)
                    y_train_fold = fold_le.transform(y_train)
                    data_split = {
                        'X_train': X_train,
                        'y_train': y_train_fold,
                        'X_val': X_val,
                        'y_val': y_val,
                        'num_classes': len(np.unique(y_train_fold)),
                        'input_dim': X_train.shape[1],
                    }
                else:
                    data_split = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_val': X_val,
                        'y_val': y_val,
                        'num_classes': len(label_encoder.classes_),
                        'input_dim': X_train.shape[1],
                    }

                # Train and evaluate on this fold
                trained_model = train_and_evaluate_model(model_name, model_info, data_split, fold_dir)

                preds, confs = get_predictions(trained_model, model_info['type'], X_val)
                if is_xgb:
                    # Map predictions back to the original label space for scoring
                    preds_orig = fold_le.inverse_transform(preds)
                    acc = accuracy_score(y_val, preds_orig)
                    macro_f1 = f1_score(y_val, preds_orig, average='macro')
                    metrics_dict = _compute_fold_metrics(y_val, preds_orig, confs)
                else:
                    acc = accuracy_score(y_val, preds)
                    macro_f1 = f1_score(y_val, preds, average='macro')
                    metrics_dict = _compute_fold_metrics(y_val, preds, confs)

                fold_metrics.append({
                    'fold': fold_index,
                    'num_train': int(len(train_idx)),
                    'num_val': int(len(val_idx)),
                    'accuracy': float(acc),
                    'macro_f1': float(macro_f1),
                })

                print(f"Fold {fold_index}: accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")

                table_rows.append({
                    'iteration': fold_index - 1,
                    'num_classes': int(len(label_encoder.classes_)),
                    **metrics_dict,
                })

        # Aggregate metrics across folds
        accs = [fm['accuracy'] for fm in fold_metrics]
        f1s = [fm['macro_f1'] for fm in fold_metrics]
        summary = {
            'model': model_name,
            'level': level,
            'folds': n_splits,
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0),
            'macro_f1_mean': float(np.mean(f1s)),
            'macro_f1_std': float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0),
        }

        # Save JSON with per-fold and summary metrics
        out_payload = {
            'summary': summary,
            'per_fold': fold_metrics,
        }
        out_json = os.path.join(model_output_root, 'kfold_metrics.json')
        with open(out_json, 'w') as f:
            json.dump(out_payload, f, indent=2)

        # Write additional CSV table (per-fold rows + average), intended for 3-fold CV
        if table_rows:
            csv_path = os.path.join(model_output_root, 'kfold_metrics_table.csv')
            headers = [
                'Iteration', 'Num_Classes', 'Accuracy(%)', 'Precision(%)', 'Recall(%)', 'AUC(%)', 'F1-Score(%)'
            ]

            # Compute averages
            avg_acc = float(np.mean([r['accuracy'] for r in table_rows]))
            avg_prec = float(np.mean([r['precision'] for r in table_rows]))
            avg_rec = float(np.mean([r['recall'] for r in table_rows]))
            # AUC may be NaN for some folds; average only over finite values
            auc_vals = [r['auc'] for r in table_rows if np.isfinite(r['auc'])]
            avg_auc = float(np.mean(auc_vals)) if auc_vals else float('nan')
            avg_f1 = float(np.mean([r['f1'] for r in table_rows]))

            with open(csv_path, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(headers)
                for r in table_rows:
                    writer.writerow([
                        r['iteration'],
                        r['num_classes'],
                        f"{r['accuracy']*100:.2f}",
                        f"{r['precision']*100:.2f}",
                        f"{r['recall']*100:.2f}",
                        f"{(r['auc']*100 if np.isfinite(r['auc']) else float('nan')):.2f}",
                        f"{r['f1']*100:.2f}",
                    ])
                writer.writerow([
                    'Average', '',
                    f"{avg_acc*100:.2f}",
                    f"{avg_prec*100:.2f}",
                    f"{avg_rec*100:.2f}",
                    f"{(avg_auc*100 if np.isfinite(avg_auc) else float('nan')):.2f}",
                    f"{avg_f1*100:.2f}",
                ])

            print(f"Saved detailed K-Fold table CSV: {csv_path}")

        print("\n" + "-" * 60)
        print(f"K-Fold summary for {model_name} ({level}):")
        print(json.dumps(summary, indent=2))
        print(f"Saved: {out_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stratified K-Fold cross-validation for protein classification (no report files)."
    )
    parser.add_argument('--level', type=str, required=True, choices=['subfamily', 'family'],
                        help='Classification level to run.')
    parser.add_argument('--model', type=str, default='all',
                        help="Model to run. Use 'all' or a specific key from models.py")
    parser.add_argument('--folds', type=int, default=5, help='Number of folds (K).')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle data before splitting into batches.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--cv-mode', type=str, default='manual', choices=['manual', 'sklearn'],
                        help='Use manual per-fold training (default) or sklearn cross_validate for sklearn models.')

    args = parser.parse_args()

    run_kfold(level=args.level, model=args.model, n_splits=args.folds, shuffle=args.shuffle, random_state=args.seed, cv_mode=args.cv_mode)


if __name__ == '__main__':
    main()

