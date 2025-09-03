import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone

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
    # Following evaluation_same_sup.py approach: ROC of correctness vs confidence
    if len(np.unique(correct_flags)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(correct_flags, confidences))
    except Exception:
        return 0.0


def _evaluate_standard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
) -> Tuple[float, float, float, float]:
    accuracy = (y_true == y_pred).mean() if len(y_true) > 0 else 0.0
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    correct_flags = (y_true == y_pred).astype(int)
    auc_val = _compute_auc_from_confidence(correct_flags, confidences)
    return accuracy, precision, recall, f1, auc_val


def _evaluate_one_vs_all(
    df,
    label_encoder,
    X: np.ndarray,
    predict_fn,
    train_indices: List[int],
    test_indices_with_negatives: List[int],
    is_negative_control: Dict[int, bool],
    target_test_mapping: Dict[str, Dict[str, List[int]]],
    level: str,
    family_to_superfamily_map: Dict[str, str],
) -> Tuple[float, float, float, float, float]:
    # Compute predictions/confidences for the combined validation set (incl. negatives)
    X_val_full = X[test_indices_with_negatives]
    predictions_sel, confidences_sel = predict_fn(X_val_full)

    report, results_df = evaluate_model_detailed(
        df,
        predictions_sel,
        confidences_sel,
        label_encoder,
        np.array(test_indices_with_negatives),
        np.array(train_indices),
        is_negative_control,
        target_test_mapping,
        level,
        family_to_superfamily_map,
    )

    total_tp = sum(m['TP'] for m in report.values())
    total_fp = sum(m['FP'] for m in report.values())
    total_tn = sum(m['TN'] for m in report.values())
    total_fn = sum(m['FN'] for m in report.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0  # not exported
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

    # AUC from correctness vs confidence
    correct_flags = (results_df['True_Label'] == results_df['Predicted_Label']).astype(int).to_numpy()
    auc_val = _compute_auc_from_confidence(correct_flags, results_df['Confidence'].to_numpy())

    return accuracy, precision, recall, f1, auc_val


def _format_rows(rows: List[Dict[str, float]]) -> pd.DataFrame:
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
    # Average row
    if rows:
        avg_acc = float(np.mean([r['accuracy'] for r in rows]))
        avg_prec = float(np.mean([r['precision'] for r in rows]))
        avg_rec = float(np.mean([r['recall'] for r in rows]))
        avg_auc = float(np.mean([r['auc'] for r in rows]))
        avg_f1 = float(np.mean([r['f1'] for r in rows]))
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
    parser = argparse.ArgumentParser(description="3-Fold Cross-Validation for Protein Classification")
    parser.add_argument('--level', type=str, required=True, choices=['subfamily', 'family'], help='Classification level')
    parser.add_argument('--model', type=str, default='all', help="Model to run: 'all' or a key from models.py")
    args = parser.parse_args()

    # Load data and features
    df = load_protein_data(config.PROTEIN_DATA_PATH, level=args.level)
    superfamily_map = load_superfamily_map(config.SUPERFAMILY_MAP_PATH)
    X, y, label_encoder, domain_vocab, feature_stats = build_features(
        df,
        level=args.level,
        max_domains=config.MAX_DOMAINS,
        max_separators=config.MAX_SEPARATORS,
        evalue_threshold=config.EVALUE_THRESHOLD,
    )

    # Build folds ONCE to guarantee identical splits across models
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds: List[Tuple[np.ndarray, np.ndarray]] = list(kf.split(X, y))
    print(f"Prepared {len(folds)} stratified folds (same splits will be used for all models).")

    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    models_to_run = [m for m in models_to_run if m in MODELS]
    if not models_to_run:
        print("No valid models specified.")
        return

    # Accumulators per model across folds
    per_model_results_std: Dict[str, List[Dict[str, float]]] = {m: [] for m in models_to_run}
    per_model_results_ova: Dict[str, List[Dict[str, float]]] = {m: [] for m in models_to_run}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print("\n" + "=" * 70)
        print(f"Fold {fold_idx} | Train size: {len(train_idx)} | Test size: {len(test_idx)}")
        print("Generating negative controls for this fold (shared across models)...")

        # Ensure reproducibility of negative control sampling per fold
        random.seed(42 + fold_idx)
        negative_control_dict, target_to_test_indices = generate_negative_controls(
            df, test_idx.tolist(), train_idx.tolist(), superfamily_map, args.level
        )
        final_test_indices = [idx for indices in target_to_test_indices.values() for idx in indices]
        test_indices_with_negatives = final_test_indices.copy()
        is_negative_control = {idx: False for idx in final_test_indices}
        target_test_mapping = {}
        all_negative_indices = set()
        for target_class, class_test_indices in target_to_test_indices.items():
            negative_indices = negative_control_dict.get(target_class, [])
            all_negative_indices.update(negative_indices)
            target_test_mapping[target_class] = {
                'positive': class_test_indices,
                'negative': negative_indices,
            }
        for idx in all_negative_indices:
            if idx not in test_indices_with_negatives:
                test_indices_with_negatives.append(idx)
                is_negative_control[idx] = True

        for model_name in models_to_run:
            model_info = MODELS[model_name]
            print(f"- Training {model_name}...")
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[test_idx], y[test_idx]

            # Build a predict_fn that outputs (preds_global, confidences)
            if model_info['type'] == 'sklearn':
                # Special handling for XGBoost: requires contiguous labels 0..K-1
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

            # Standard evaluation on this fold
            preds_std, conf_std = predict_fn(X_val)
            acc, prec, rec, f1, auc_val = _evaluate_standard(y_val, preds_std, conf_std)
            per_model_results_std[model_name].append({
                'num_classes': len(label_encoder.classes_),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc_val,
            })

            # One-vs-all evaluation (shared negative controls)
            acc_ova, prec_ova, rec_ova, f1_ova, auc_ova = _evaluate_one_vs_all(
                df=df,
                label_encoder=label_encoder,
                X=X,
                predict_fn=predict_fn,
                train_indices=train_idx.tolist(),
                test_indices_with_negatives=test_indices_with_negatives,
                is_negative_control=is_negative_control,
                target_test_mapping=target_test_mapping,
                level=args.level,
                family_to_superfamily_map=superfamily_map,
            )
            per_model_results_ova[model_name].append({
                'num_classes': len(label_encoder.classes_),
                'accuracy': acc_ova,
                'precision': prec_ova,
                'recall': rec_ova,
                'f1': f1_ova,
                'auc': auc_ova,
            })

            print(f"  {model_name}: Standard Acc={acc*100:.2f}% | OVA Acc={acc_ova*100:.2f}%")

    # After all folds, write CSVs per model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_base = os.path.join(base_dir, 'benchmark_results_same_sup', 'kfold')
    for model_name in models_to_run:
        out_dir = os.path.join(results_base, args.level, model_name, 'csv_files')
        os.makedirs(out_dir, exist_ok=True)

        df_std = _format_rows(per_model_results_std[model_name])
        df_ova = _format_rows(per_model_results_ova[model_name])
        std_path = os.path.join(out_dir, 'kfold_standard.csv')
        ova_path = os.path.join(out_dir, 'kfold_one_vs_all.csv')
        df_std.to_csv(std_path, index=False)
        df_ova.to_csv(ova_path, index=False)
        print(f"Saved: {std_path}")
        print(f"Saved: {ova_path}")


if __name__ == '__main__':
    main()


