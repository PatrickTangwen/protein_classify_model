import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json

def get_predictions(model, model_type, X_val):
    """
    Gets predictions and probabilities from a trained model.

    Args:
        model: The trained model object.
        model_type (str): 'pytorch' or 'sklearn'.
        X_val (np.ndarray): The validation feature matrix.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The predicted labels.
            - np.ndarray: The prediction confidences/probabilities.
    """
    if model_type == 'pytorch':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(X_val).to(device)
            outputs = model(features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            return predictions.cpu().numpy(), confidences.cpu().numpy()
    elif model_type == 'sklearn':
        predictions = model.predict(X_val)
        probs = model.predict_proba(X_val)
        confidences = np.max(probs, axis=1)
        return predictions, confidences
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def analyze_misclassification(true_class, pred_class, level, family_to_superfamily_map):
    """
    Analyzes if a misclassification is within the same family/superfamily.
    """
    if level == 'subfamily':
        true_family = '.'.join(str(true_class).split('.')[:3])
        pred_family = '.'.join(str(pred_class).split('.')[:3])
        return 'same_family' if true_family == pred_family else 'different_family'
    elif level == 'family':
        true_superfamily = family_to_superfamily_map.get(str(true_class), 'Unknown')
        pred_superfamily = family_to_superfamily_map.get(str(pred_class), 'Unknown')
        return 'same_superfamily' if true_superfamily != 'Unknown' and true_superfamily == pred_superfamily else 'different_superfamily'
    return 'unknown'

def evaluate_model_detailed(
    df, predictions, confidences, label_encoder, val_indices,
    train_indices, is_negative_control, target_test_mapping,
    level, family_to_superfamily_map
):
    """
    Performs a comprehensive, generalized evaluation for any model.
    """
    target_col = 'Family' if level == 'family' else 'Subfamily'
    true_labels_all = df[target_col].astype('category')
    true_labels_encoded = true_labels_all.cat.codes
    
    results_df = pd.DataFrame({
        'Protein': df.loc[val_indices, 'Accession'].values,
        'True_Label': label_encoder.inverse_transform(true_labels_encoded[val_indices]),
        'Predicted_Label': label_encoder.inverse_transform(predictions),
        'Confidence': confidences,
        'Index': val_indices
    })
    results_df['is_negative_control'] = results_df['Index'].apply(lambda x: is_negative_control.get(x, False))
    
    # Initialize report structure
    report_metrics = defaultdict(lambda: {
        'train_count': 0, 'test_count': 0, 'correct': 0, 'size': 0,
        'misclassified': [], 'same_level_errors': 0, 'diff_level_errors': 0,
        'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'negative_count': 0
    })

    # Populate counts
    class_counts = df[target_col].value_counts()
    for class_name, count in class_counts.items():
        report_metrics[class_name]['size'] = count
    
    train_counts = df.iloc[train_indices][target_col].value_counts()
    for class_name, count in train_counts.items():
        report_metrics[class_name]['train_count'] = count
        
    for class_name, mapping in target_test_mapping.items():
        report_metrics[class_name]['negative_count'] = len(mapping['negative'])

    # Calculate metrics from results
    for _, row in results_df.iterrows():
        true_label, pred_label, data_idx = row['True_Label'], row['Predicted_Label'], row['Index']
        
        if not row['is_negative_control']:
            report_metrics[true_label]['test_count'] += 1
            if true_label == pred_label:
                report_metrics[true_label]['correct'] += 1
            else:
                error_type = analyze_misclassification(true_label, pred_label, level, family_to_superfamily_map)
                if 'same' in error_type:
                    report_metrics[true_label]['same_level_errors'] += 1
                else:
                    report_metrics[true_label]['diff_level_errors'] += 1
                report_metrics[true_label]['misclassified'].append({
                    'Protein': row['Protein'], 'Predicted_as': pred_label,
                    'Confidence': row['Confidence'], 'Error_Type': error_type
                })
        
        for class_name, mapping in target_test_mapping.items():
            if data_idx in mapping['positive']:
                if pred_label == class_name: report_metrics[class_name]['TP'] += 1
                else: report_metrics[class_name]['FN'] += 1
            elif data_idx in mapping['negative']:
                if pred_label != class_name: report_metrics[class_name]['TN'] += 1
                else: report_metrics[class_name]['FP'] += 1
    
    final_report = {}
    for class_name in class_counts.index:
        metrics = report_metrics[class_name]
        tp, tn, fp, fn = metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        binary_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        final_report[class_name] = metrics
        final_report[class_name].update({
            'Accuracy_Original_Test_Set': (metrics['correct'] / metrics['test_count'] * 100) if metrics['test_count'] > 0 else 0,
            'Precision': precision, 'Recall': recall, 'Specificity': specificity, 'F1_Score': f1,
            'Accuracy_incl_Negative_Controls': binary_accuracy * 100
        })
        
    return final_report, results_df

def generate_verbose_report_text(df, report, target_test_mapping, level, train_indices):
    """
    Generate the verbose classification report text in the exact format of the original scripts.
    """
    content = f"\n=== Detailed {level.capitalize()} Classification Report ===\n"
    content += "-" * 100 + "\n"
    
    target_col = 'Family' if level == 'family' else 'Subfamily'
    error_type_same = 'same_superfamily' if level == 'family' else 'same_family'
    error_type_diff = 'different_superfamily' if level == 'family' else 'different_family'
    
    for class_name, metrics in report.items():
        total_size = metrics['size']
        content += f"\n{target_col}: {class_name}\n"
        content += f"Total Size: {total_size} members\n"
        
        # Print data split information based on size
        if total_size == 1:
            content += "Data Split: Single member (used in both training and testing)\n"
        elif total_size == 2:
            content += "Data Split: Two members (1 for training, 1 for testing)\n"
        else:
            content += f"Data Split: {total_size} members (80% training, 20% testing)\n"
        
        content += "\nTraining Set Statistics:\n"
        content += f"  - Number of training proteins: {metrics['train_count']}\n"
        
        # Print training proteins details
        training_proteins = []
        for idx in train_indices:
            if idx < len(df):  # Ensure index is valid
                protein = df.iloc[idx]
                if protein[target_col] == class_name:
                    training_proteins.append((idx, protein))
        
        if training_proteins:
            content += "    Training Protein" + ("s:" if len(training_proteins) > 1 else ":") + "\n"
            for idx, protein in training_proteins:
                content += f"      - Accession: {protein['Accession']} | Subfamily: {protein['Subfamily']} | Family: {protein['Family']}\n"
        
        content += "\nTesting Set Statistics:\n"
        content += f"  - Number of test proteins: {metrics['test_count']}\n"
        
        # Print test proteins details
        if class_name in target_test_mapping:
            positive_indices = target_test_mapping[class_name]['positive']
            if positive_indices:
                content += "    Test Protein" + ("s:" if len(positive_indices) > 1 else ":") + "\n"
                for idx in positive_indices:
                    protein = df.iloc[idx]
                    content += f"      - Accession: {protein['Accession']} | Subfamily: {protein['Subfamily']} | Family: {protein['Family']}\n"
        
        content += f"  - Number of negative controls: {metrics['negative_count']}\n"
        
        # Print negative controls details
        if class_name in target_test_mapping:
            negative_indices = target_test_mapping[class_name]['negative']
            if negative_indices:
                content += "    Negative Controls:\n"
                for i, idx in enumerate(negative_indices, 1):
                    protein = df.iloc[idx]
                    content += f"      {i}. Accession: {protein['Accession']} | Subfamily: {protein['Subfamily']} | Family: {protein['Family']}\n"
        
        content += f"  - Correct predictions: {metrics['correct']}\n"
        content += f"  - Accuracy (Original Test Set): {metrics['Accuracy_Original_Test_Set']:.2f}%\n"
        
        # Binary classification metrics with negative controls
        content += "\nBinary Classification Metrics (with negative controls):\n"
        content += f"  - True Positives (TP): {metrics['TP']}\n"
        content += f"  - False Positives (FP): {metrics['FP']}\n"
        content += f"  - True Negatives (TN): {metrics['TN']}\n"
        content += f"  - False Negatives (FN): {metrics['FN']}\n"
        content += f"  - Precision: {metrics['Precision']:.4f}\n"
        content += f"  - Recall/Sensitivity: {metrics['Recall']:.4f}\n"
        content += f"  - Specificity: {metrics['Specificity']:.4f}\n"
        content += f"  - F1 Score: {metrics['F1_Score']:.4f}\n"
        content += f"  - Accuracy (incl. Negative Controls): {metrics['Accuracy_incl_Negative_Controls']:.4f}\n"
        
        misclassified_count = len(metrics['misclassified'])
        if misclassified_count > 0:
            content += "\nMisclassification Analysis:\n"
            content += f"  - Total misclassifications: {misclassified_count}\n"
            content += f"  - {error_type_same.replace('_', ' ').title()} errors: {metrics['same_level_errors']}\n"
            content += f"  - {error_type_diff.replace('_', ' ').title()} errors: {metrics['diff_level_errors']}\n"
            
            content += "\nMisclassified Proteins Details:\n"
            for misc in metrics['misclassified']:
                content += f"  - Protein: {misc['Protein']}\n"
                content += f"    Predicted as: {misc['Predicted_as']}\n"
                content += f"    Confidence: {misc['Confidence']:.4f}\n"
                content += f"    Error Type: {misc['Error_Type']}\n"
        
        content += "-" * 100 + "\n"
    
    return content

def save_reports(df, report, results_df, target_test_mapping, train_indices, output_dir, level):
    print(f"Saving reports to {output_dir}")
    
    # Determine file suffix based on level
    suffix = f"_{level}" if level == 'family' else ""
    
    # File 1: classification_report.txt (or classification_report_family.txt)
    verbose_content = generate_verbose_report_text(df, report, target_test_mapping, level, train_indices)
    report_filename = f'classification_report{suffix}.txt'
    with open(os.path.join(output_dir, report_filename), 'w', encoding='utf-8') as f:
        f.write(verbose_content)
        
    # File 2: detailed_classification_results.csv (or detailed_classification_results_family.csv)
    results_filename = f'detailed_classification_results{suffix}.csv'
    # Rename columns to match original format
    results_output = results_df.copy()
    results_output = results_output.rename(columns={
        'Protein': 'Protein',
        'True_Label': f'True_{level.capitalize()}',
        'Predicted_Label': f'Predicted_{level.capitalize()}',
        'Confidence': 'Confidence'
    })
    results_output.to_csv(os.path.join(output_dir, results_filename), index=False)

    # File 3: binary_classification_metrics.csv (or binary_classification_metrics_family.csv)
    binary_metrics_data = []
    for class_name, metrics in report.items():
        binary_metrics_data.append({
            level.capitalize(): class_name,
            'TP': metrics['TP'],
            'FP': metrics['FP'], 
            'TN': metrics['TN'],
            'FN': metrics['FN'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'Specificity': metrics['Specificity'],
            'F1_Score': metrics['F1_Score'],
            'Accuracy_Original_Test_Set': f"{metrics['Accuracy_Original_Test_Set']:.2f}%",
            'Accuracy_incl_Negative_Controls': f"{metrics['Accuracy_incl_Negative_Controls']:.4f}",
            'Test_Samples': metrics['test_count'],
            'Negative_Controls': metrics['negative_count']
        })
    binary_metrics_filename = f'binary_classification_metrics{suffix}.csv'
    pd.DataFrame(binary_metrics_data).to_csv(os.path.join(output_dir, binary_metrics_filename), index=False)

    # File 4: classification_stats.txt (or classification_stats_family.txt)
    stats_filename = f'classification_stats{suffix}.txt'
    stats_content = generate_classification_stats(results_df, report, level)
    with open(os.path.join(output_dir, stats_filename), 'w', encoding='utf-8') as f:
        f.write(stats_content)
    
    # File 5: summary_metrics.json (for benchmarking)
    total_test = sum(m['test_count'] for m in report.values())
    total_correct = sum(m['correct'] for m in report.values())
    overall_accuracy = (total_correct / total_test * 100) if total_test > 0 else 0
    
    summary_metrics = {
        'overall_accuracy_original_set': overall_accuracy,
        'total_test_samples': total_test,
        'total_correct_predictions': total_correct
    }
    
    with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary_metrics, f, indent=2)

def generate_classification_stats(results_df, report, level):
    """Generate the classification_stats file content to match original format exactly."""
    content = ""
    
    # === Confidence Threshold Analysis ===
    content += "=== Confidence Threshold Analysis ===\n\n"
    
    original_test_df = results_df[~results_df['is_negative_control']]
    total_original = len(original_test_df)
    confidence_data = []
    
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        if threshold == 0.0:
            retained_df = original_test_df
            threshold_label = "0.0 (Full Test Set)"
        else:
            retained_df = original_test_df[original_test_df['Confidence'] >= threshold]
            threshold_label = f"{threshold:.1f}"
        
        samples_retained = len(retained_df)
        percent_retained = (samples_retained / total_original * 100) if total_original > 0 else 0
        
        if samples_retained > 0:
            correct_predictions = (retained_df['True_Label'] == retained_df['Predicted_Label']).sum()
            accuracy = (correct_predictions / samples_retained) * 100
        else:
            accuracy = 0.0
        
        confidence_data.append([
            threshold_label,
            samples_retained,
            f"{percent_retained:.1f}",
            f"{accuracy:.2f}"
        ])
    
    level_name = level.capitalize()
    headers = [
        "Confidence Threshold",
        f"Original Test Proteins Above Threshold", 
        "% of Original Test Set Retained",
        f"{level_name} Accuracy (Original Test Set, %)"
    ]
    
    content += tabulate(confidence_data, headers=headers, tablefmt="grid")
    content += "\n\n"
    
    # Column definitions
    content += """
Column Definitions:
* Confidence Threshold: The threshold value applied. '0.0' indicates the evaluation on the complete test set without filtering by confidence.
* Original Test Proteins Above Threshold: The absolute number of *original test proteins* (excluding negative controls) whose model predictions had a confidence score >= the specified threshold.
* % of Original Test Set Retained: The percentage of the *total original test proteins* that were retained (calculated as 'Original Test Proteins Above Threshold' / Total Original Test Proteins * 100).
* {} Accuracy (Original Test Set, %): The accuracy of {}-level classification calculated *only* on the *original test proteins* that were retained above the specified threshold.
""".format(level_name, level_name)
    
    content += "\n\n"
    
    # === Overall Classification Statistics (Original Test Set) ===
    total_test = sum(m['test_count'] for m in report.values())
    total_correct = sum(m['correct'] for m in report.values())
    overall_accuracy = (total_correct / total_test * 100) if total_test > 0 else 0
    
    classification_stats_data = [
        ["Total Test Proteins (Original Set)", total_test],
        ["Total Correct Predictions (Original Set)", total_correct],
        ["Overall Accuracy (Original Test Set)", f"{overall_accuracy:.2f}%"]
    ]
    
    content += "=== Overall Classification Statistics (Original Test Set) ===\n"
    content += tabulate(classification_stats_data, headers=["Metric", "Value"], tablefmt="grid")
    content += "\n\n"
    
    # === Binary Classification Metrics with Negative Controls ===
    total_tp = sum(m['TP'] for m in report.values())
    total_fp = sum(m['FP'] for m in report.values())
    total_tn = sum(m['TN'] for m in report.values())
    total_fn = sum(m['FN'] for m in report.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_accuracy_binary = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    
    binary_stats_data = [
        ["True Positives (TP)", total_tp],
        ["False Positives (FP)", total_fp],
        ["True Negatives (TN)", total_tn],
        ["False Negatives (FN)", total_fn],
        ["Precision", f"{overall_precision:.4f}"],
        ["Recall/Sensitivity", f"{overall_recall:.4f}"],
        ["Specificity", f"{overall_specificity:.4f}"],
        ["F1 Score", f"{overall_f1:.4f}"],
        ["Accuracy (incl. Negative Controls)", f"{overall_accuracy_binary:.4f}"]
    ]
    
    content += "=== Binary Classification Metrics with Negative Controls ===\n"
    content += tabulate(binary_stats_data, headers=["Metric", "Value"], tablefmt="grid")
    content += "\n\n"
    
    # === Misclassification Statistics (Original Test Set) ===
    total_misclassifications = sum(len(m['misclassified']) for m in report.values())
    total_same_level_errors = sum(m['same_level_errors'] for m in report.values())
    total_diff_level_errors = sum(m['diff_level_errors'] for m in report.values())
    
    content += "=== Misclassification Statistics (Original Test Set) ===\n"
    if total_misclassifications > 0:
        error_type_same = 'Same Superfamily Errors' if level == 'family' else 'Same Family Errors'
        error_type_diff = 'Different Superfamily Errors' if level == 'family' else 'Different Family Errors'
        
        misclassification_data = [
            ["Total Misclassifications", total_misclassifications, "100%"],
            [error_type_same, total_same_level_errors, f"{(total_same_level_errors/total_misclassifications*100):.2f}%"],
            [error_type_diff, total_diff_level_errors, f"{(total_diff_level_errors/total_misclassifications*100):.2f}%"]
        ]
        content += tabulate(misclassification_data, headers=["Error Type", "Count", "Percentage"], tablefmt="grid")
    else:
        content += "No misclassifications found."
    
    return content

    # total_correct = sum(m['correct'] for m in report.values())
    # overall_accuracy = (total_correct / total_test * 100) if total_test > 0 else 0
    # stats_content += "=== Overall Classification Statistics (Original Test Set) ===\n"
    # stats_content += tabulate([["Total Test Proteins", total_test], ["Total Correct", total_correct], ["Overall Accuracy", f"{overall_accuracy:.2f}%"]], headers=["Metric", "Value"]) + "\n\n"
    
    # # Section: Overall Binary Metrics
    # total_tp = sum(m['TP'] for m in report.values())
    # total_fp = sum(m['FP'] for m in report.values())
    # total_tn = sum(m['TN'] for m in report.values())
    # total_fn = sum(m['FN'] for m in report.values())
    # overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    # overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    # overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
    # stats_content += "=== Binary Classification Metrics with Negative Controls ===\n"
    # stats_content += tabulate([["Precision", f"{overall_prec:.4f}"], ["Recall", f"{overall_rec:.4f}"], ["F1 Score", f"{overall_f1:.4f}"]], headers=["Metric", "Value"]) + "\n\n"

    # # Section: Misclassification Stats
    # total_misclass = sum(len(m['misclassified']) for m in report.values())
    # same_level_errors = sum(m['same_level_errors'] for m in report.values())
    # diff_level_errors = sum(m['diff_level_errors'] for m in report.values())
    # stats_content += "=== Misclassification Statistics (Original Test Set) ===\n"
    # if total_misclass > 0:
    #     stats_content += tabulate([
    #         ["Total Misclassifications", total_misclass, "100%"],
    #         [f"Same-{level} Errors", same_level_errors, f"{(same_level_errors/total_misclass*100):.2f}%"],
    #         [f"Different-{level} Errors", diff_level_errors, f"{(diff_level_errors/total_misclass*100):.2f}%"]
    #     ], headers=["Error Type", "Count", "Percentage"])
    # else:
    #     stats_content += "No misclassifications found."
        
    # with open(os.path.join(output_dir, 'classification_stats.txt'), 'w') as f:
    #     f.write(stats_content)

    # # File 5: summary_metrics.json (for benchmarking)
    # with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
    #     json.dump({'overall_accuracy_original_set': overall_accuracy}, f, indent=4)
        
    # print("Reports saved successfully.")

def generate_roc_curve(results_df, output_dir, level):
    if 'Confidence' not in results_df.columns:
        print("Warning: 'Confidence' column not found. Cannot generate ROC curve.")
        return

    y_true_binary = (results_df['True_Label'] == results_df['Predicted_Label']).astype(int)
    y_scores = results_df['Confidence']

    if len(y_true_binary.unique()) < 2:
        print("Warning: Only one class present. Cannot generate ROC curve.")
        return

    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC: Model Confidence vs. Classification Correctness ({level.capitalize()})')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {plot_path}") 