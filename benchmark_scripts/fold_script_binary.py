import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config_same_sup as config
from data_loader import load_protein_data, load_superfamily_map
from feature_engineering import build_features
from models import MODELS, ImprovedProteinClassifier
from data_splitting_same_sup import generate_negative_controls
from evaluation_same_sup import get_predictions, evaluate_model_detailed


def _train_sklearn(model, X_train: np.ndarray, y_train: np.ndarray):
    cloned = clone(model)
    cloned.fit(X_train, y_train)
    return cloned


def _train_pytorch(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(np.concatenate((y_train, y_val))))
    model = model_class(input_dim=input_dim, num_classes=num_classes).to(device)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=config.PYTORCH_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.PYTORCH_BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.PYTORCH_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.PYTORCH_EPOCHS):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = (100.0 * correct / total) if total > 0 else 0.0
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PYTORCH_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _compute_auc_from_confidence(correct_flags: np.ndarray, confidences: np.ndarray) -> float:
    if len(np.unique(correct_flags)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(correct_flags, confidences))
    except Exception:
        return 0.0


def _aggregate_binary_metrics(report: Dict[str, Dict[str, float]]) -> Tuple[float, float, float, float, int, int, int, int]:
    tp = sum(m['TP'] for m in report.values())
    fp = sum(m['FP'] for m in report.values())
    tn = sum(m['TN'] for m in report.values())
    fn = sum(m['FN'] for m in report.values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return accuracy, precision, recall, f1, tp, fp, tn, fn


def _format_rows(rows: List[Dict[str, float]]) -> "np.ndarray":
    import pandas as pd
    df_rows = []
    for i, row in enumerate(rows):
        df_rows.append({
            'Iteration': i,
            'Num_Classes': row['num_classes'],
            'Accuracy(%)': f"{row['accuracy'] * 100:.2f}",
            'Precision(%)': f"{row['precision'] * 100:.2f}",
            'Recall(%)': f"{row['recall'] * 100:.2f}",
            'AUC(%)': f"{row['auc'] * 100:.2f}",
            'F1-Score(%)': f"{row['f1'] * 100:.2f}",
        })
    if rows:
        import numpy as _np
        avg_acc = float(_np.mean([r['accuracy'] for r in rows]))
        avg_prec = float(_np.mean([r['precision'] for r in rows]))
        avg_rec = float(_np.mean([r['recall'] for r in rows]))
        avg_auc = float(_np.mean([r['auc'] for r in rows]))
        avg_f1 = float(_np.mean([r['f1'] for r in rows]))
        df_rows.append({
            'Iteration': 'Average',
            'Num_Classes': '',
            'Accuracy(%)': f"{avg_acc * 100:.2f}",
            'Precision(%)': f"{avg_prec * 100:.2f}",
            'Recall(%)': f"{avg_rec * 100:.2f}",
            'AUC(%)': f"{avg_auc * 100:.2f}",
            'F1-Score(%)': f"{avg_f1 * 100:.2f}",
        })
    return pd.DataFrame(df_rows, columns=['Iteration', 'Num_Classes', 'Accuracy(%)', 'Precision(%)', 'Recall(%)', 'AUC(%)', 'F1-Score(%)'])


def main():
    parser = argparse.ArgumentParser(description="3-Fold CV with binary accuracy for both outputs (with/without negatives)")
    parser.add_argument('--level', type=str, required=True, choices=['subfamily', 'family'], help='Classification level')
    parser.add_argument('--model', type=str, default='all', help="'all' or a model key from models.py")
    args = parser.parse_args()

    # Load data and features
    df = load_protein_data(config.PROTEIN_DATA_PATH, level=args.level)
    superfamily_map = load_superfamily_map(config.SUPERFAMILY_MAP_PATH)
    X, y, label_encoder, _, _ = build_features(
        df,
        level=args.level,
        max_domains=config.MAX_DOMAINS,
        max_separators=config.MAX_SEPARATORS,
        evalue_threshold=config.EVALUE_THRESHOLD,
    )

    # Prepare folds once
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds: List[Tuple[np.ndarray, np.ndarray]] = list(kf.split(X, y))
    print(f"Prepared {len(folds)} stratified folds (same for all models)")

    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    models_to_run = [m for m in models_to_run if m in MODELS]
    if not models_to_run:
        print("No valid models specified.")
        return

    # Accumulators per model
    per_model_no_neg: Dict[str, List[Dict[str, float]]] = {m: [] for m in models_to_run}
    per_model_with_neg: Dict[str, List[Dict[str, float]]] = {m: [] for m in models_to_run}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print("\n" + "=" * 70)
        print(f"Fold {fold_idx} | Train: {len(train_idx)} | Test: {len(test_idx)}")

        # Build negatives once per fold
        random.seed(42 + fold_idx)
        negative_control_dict, target_to_test_indices = generate_negative_controls(
            df, test_idx.tolist(), train_idx.tolist(), superfamily_map, args.level
        )
        final_test_indices = [idx for indices in target_to_test_indices.values() for idx in indices]
        test_indices_with_negatives = final_test_indices.copy()
        is_negative_control = {idx: False for idx in final_test_indices}
        target_test_mapping_with_neg = {}
        all_negative_indices = set()
        for target_class, class_test_indices in target_to_test_indices.items():
            negs = negative_control_dict.get(target_class, [])
            all_negative_indices.update(negs)
            target_test_mapping_with_neg[target_class] = {'positive': class_test_indices, 'negative': negs}
        for idx in all_negative_indices:
            if idx not in test_indices_with_negatives:
                test_indices_with_negatives.append(idx)
                is_negative_control[idx] = True

        # Build mapping without negatives
        target_test_mapping_no_neg = {}
        # Consider only classes present in this test fold
        target_col = 'Family' if args.level == 'family' else 'Subfamily'
        test_classes = df.iloc[test_idx][target_col].unique().tolist()
        for cls in test_classes:
            pos_indices = df.index[(df[target_col] == cls) & (df.index.isin(test_idx))].tolist()
            target_test_mapping_no_neg[cls] = {'positive': pos_indices, 'negative': []}

        for model_name in models_to_run:
            model_info = MODELS[model_name]
            print(f"- Training {model_name}...")
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[test_idx], y[test_idx]

            # Build predict_fn handling XGBoost remapping
            if model_info['type'] == 'sklearn':
                if model_name == 'xgboost':
                    present_classes = np.array(sorted(np.unique(y_train)))
                    class_to_local = {c: i for i, c in enumerate(present_classes)}
                    y_train_local = np.array([class_to_local[c] for c in y_train], dtype=np.int64)
                    trained_model = _train_sklearn(model_info['model'], X_train, y_train_local)

                    def predict_fn(batch_X: np.ndarray):
                        preds_local = trained_model.predict(batch_X)
                        probs = trained_model.predict_proba(batch_X)
                        confidences = np.max(probs, axis=1)
                        preds_global = present_classes[preds_local]
                        return preds_global, confidences
                else:
                    trained_model = _train_sklearn(model_info['model'], X_train, y_train)

                    def predict_fn(batch_X: np.ndarray):
                        return get_predictions(trained_model, model_info['type'], batch_X)
            elif model_info['type'] == 'pytorch':
                trained_model = _train_pytorch(ImprovedProteinClassifier, X_train, y_train, X_val, y_val)

                def predict_fn(batch_X: np.ndarray):
                    return get_predictions(trained_model, model_info['type'], batch_X)
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")

            # Without negatives: use test_idx only
            preds_no_neg, conf_no_neg = predict_fn(X_val)
            report_no_neg, results_df_no_neg = evaluate_model_detailed(
                df,
                preds_no_neg,
                conf_no_neg,
                label_encoder,
                np.array(test_idx.tolist()),
                np.array(train_idx.tolist()),
                {i: False for i in test_idx.tolist()},
                target_test_mapping_no_neg,
                args.level,
                superfamily_map,
            )
            acc, prec, rec, f1, tp, fp, tn, fn = _aggregate_binary_metrics(report_no_neg)
            auc_val = _compute_auc_from_confidence(
                (results_df_no_neg['True_Label'] == results_df_no_neg['Predicted_Label']).astype(int).to_numpy(),
                results_df_no_neg['Confidence'].to_numpy(),
            )
            per_model_no_neg[model_name].append({
                'num_classes': len(label_encoder.classes_),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc_val,
            })

            # With negatives: use shared negatives per fold
            preds_w_neg, conf_w_neg = predict_fn(X[test_indices_with_negatives])
            report_w_neg, results_df_w_neg = evaluate_model_detailed(
                df,
                preds_w_neg,
                conf_w_neg,
                label_encoder,
                np.array(test_indices_with_negatives),
                np.array(train_idx.tolist()),
                is_negative_control,
                target_test_mapping_with_neg,
                args.level,
                superfamily_map,
            )
            acc2, prec2, rec2, f12, tp2, fp2, tn2, fn2 = _aggregate_binary_metrics(report_w_neg)
            auc2 = _compute_auc_from_confidence(
                (results_df_w_neg['True_Label'] == results_df_w_neg['Predicted_Label']).astype(int).to_numpy(),
                results_df_w_neg['Confidence'].to_numpy(),
            )
            per_model_with_neg[model_name].append({
                'num_classes': len(label_encoder.classes_),
                'accuracy': acc2,
                'precision': prec2,
                'recall': rec2,
                'f1': f12,
                'auc': auc2,
            })

            print(f"  {model_name}: NoNeg Acc={acc*100:.2f}% | WithNeg Acc={acc2*100:.2f}%")

    # Write CSVs
    import pandas as pd
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_base = os.path.join(base_dir, 'benchmark_results_same_sup', 'kfold_new')
    for model_name in models_to_run:
        out_dir = os.path.join(results_base, args.level, model_name, 'csv_files')
        os.makedirs(out_dir, exist_ok=True)
        df_no_neg = _format_rows(per_model_no_neg[model_name])
        df_with_neg = _format_rows(per_model_with_neg[model_name])
        df_no_neg.to_csv(os.path.join(out_dir, 'kfold_binary_no_negatives.csv'), index=False)
        df_with_neg.to_csv(os.path.join(out_dir, 'kfold_binary_with_negatives.csv'), index=False)
        print(f"Saved: {os.path.join(out_dir, 'kfold_binary_no_negatives.csv')}")
        print(f"Saved: {os.path.join(out_dir, 'kfold_binary_with_negatives.csv')}")


if __name__ == '__main__':
    main()


