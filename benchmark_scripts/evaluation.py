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
    stats_content = generate_classification_stats(results_df, report, target_test_mapping, level)
    with open(os.path.join(output_dir, stats_filename), 'w', encoding='utf-8') as f:
        f.write(stats_content)
    
    # File 5: summary_metrics.json (for benchmarking)
    total_test = sum(m['test_count'] for m in report.values())
    total_correct = sum(m['correct'] for m in report.values())
    recall_accuracy = (total_correct / total_test * 100) if total_test > 0 else 0
    
    # Calculate one-vs-all binary classification accuracy
    total_tp = sum(m['TP'] for m in report.values())
    total_fp = sum(m['FP'] for m in report.values())
    total_tn = sum(m['TN'] for m in report.values())
    total_fn = sum(m['FN'] for m in report.values())
    binary_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) * 100 if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    
    summary_metrics = {
        'overall_accuracy_original_set': recall_accuracy,
        'one_vs_all_accuracy': binary_accuracy,
        'total_test_samples': total_test,
        'total_correct_predictions': total_correct,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_tn': total_tn,
        'total_fn': total_fn
    }
    
    with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary_metrics, f, indent=2)

def generate_classification_stats(results_df, report, target_test_mapping, level):
    """Generate the classification_stats file content using one-vs-all binary classification accuracy for all thresholds."""
    content = ""
    
    # === Confidence Threshold Analysis ===
    content += "=== Confidence Threshold Analysis ===\n\n"
    
    original_test_df = results_df[~results_df['is_negative_control']]
    total_original = len(original_test_df)
    confidence_data = []
    
    # Calculate overall binary classification metrics for reference
    total_tp = sum(m['TP'] for m in report.values())
    total_fp = sum(m['FP'] for m in report.values())
    total_tn = sum(m['TN'] for m in report.values())
    total_fn = sum(m['FN'] for m in report.values())
    
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        if threshold == 0.0:
            # For full test set
            original_test_retained = original_test_df
            threshold_label = "0.0 (Full Test Set)"
            accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) * 100 if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
        else:
            # Filter original test proteins by confidence threshold
            original_test_retained = original_test_df[original_test_df['Confidence'] >= threshold]
            threshold_label = f"{threshold:.1f}"
            
            # Filter ALL samples (including negative controls) by confidence threshold
            all_samples_retained = results_df[results_df['Confidence'] >= threshold]
            
            if len(all_samples_retained) > 0:
                # Calculate one-vs-all binary classification metrics for threshold-filtered data
                threshold_tp = threshold_tn = threshold_fp = threshold_fn = 0
                
                # Get the indices of samples above threshold
                threshold_indices = set(all_samples_retained['Index'])
                
                # For each class, calculate TP/TN/FP/FN for samples above threshold
                for class_name, mapping in target_test_mapping.items():
                    # Count positive samples above threshold
                    positive_above_threshold = [idx for idx in mapping['positive'] if idx in threshold_indices]
                    negative_above_threshold = [idx for idx in mapping['negative'] if idx in threshold_indices]
                    
                    # For positive samples above threshold
                    for idx in positive_above_threshold:
                        # Get the prediction for this sample
                        sample_row = all_samples_retained[all_samples_retained['Index'] == idx]
                        if not sample_row.empty:
                            pred_label = sample_row.iloc[0]['Predicted_Label']
                            if pred_label == class_name:
                                threshold_tp += 1
                            else:
                                threshold_fn += 1
                    
                    # For negative samples above threshold  
                    for idx in negative_above_threshold:
                        # Get the prediction for this sample
                        sample_row = all_samples_retained[all_samples_retained['Index'] == idx]
                        if not sample_row.empty:
                            pred_label = sample_row.iloc[0]['Predicted_Label']
                            if pred_label != class_name:
                                threshold_tn += 1
                            else:
                                threshold_fp += 1
                
                # Calculate one-vs-all binary classification accuracy
                total_threshold_samples = threshold_tp + threshold_tn + threshold_fp + threshold_fn
                accuracy = (threshold_tp + threshold_tn) / total_threshold_samples * 100 if total_threshold_samples > 0 else 0
            else:
                accuracy = 0.0
        
        samples_retained = len(original_test_retained)
        percent_retained = (samples_retained / total_original * 100) if total_original > 0 else 0
        
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
        f"One-vs-All Accuracy (%)"
    ]
    
    content += tabulate(confidence_data, headers=headers, tablefmt="grid")
    content += "\n\n"
    
    # Column definitions
    content += f"""
Column Definitions:
* Confidence Threshold: The threshold value applied. '0.0' indicates the evaluation on the complete test set without filtering by confidence.
* Original Test Proteins Above Threshold: The absolute number of *original test proteins* (excluding negative controls) whose model predictions had a confidence score >= the specified threshold.
* % of Original Test Set Retained: The percentage of the *total original test proteins* that were retained (calculated as 'Original Test Proteins Above Threshold' / Total Original Test Proteins * 100).
* One-vs-All Accuracy (%): For each {level_name.lower()}, treats the classification as a binary problem (target {level_name.lower()} vs. all others including negative controls). Accuracy is calculated as (TP + TN) / (TP + TN + FP + FN) where:
  - TP: Correctly predicted as target {level_name.lower()}
  - FN: Incorrectly predicted as different {level_name.lower()} (should have been target)
  - TN: Correctly predicted as different {level_name.lower()} (negative control correctly rejected)
  - FP: Incorrectly predicted as target {level_name.lower()} (negative control incorrectly accepted)
"""
    
    content += "\n\n"
    
    # === Overall Classification Statistics (One-vs-All Approach) ===
    # Use one-vs-all binary classification accuracy formula: (TP + TN) / (TP + TN + FP + FN)
    overall_accuracy_binary = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) * 100 if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    
    classification_stats_data = [
        ["Total Test Proteins (Original Set)", sum(m['test_count'] for m in report.values())],
        ["Total Negative Controls", sum(m['negative_count'] for m in report.values())],
        ["Total Test Samples (incl. Negative Controls)", total_tp + total_tn + total_fp + total_fn],
        ["Total Correct Predictions (Original Set)", sum(m['correct'] for m in report.values())],
        [f"One-vs-All Accuracy (incl. Negative Controls)", f"{overall_accuracy_binary:.2f}%"]
    ]
    
    content += f"=== Overall Classification Statistics (One-vs-All Approach) ===\n"
    content += tabulate(classification_stats_data, headers=["Metric", "Value"], tablefmt="grid")
    content += "\n\n"
    
    # === One-vs-All Binary Classification Metrics ===
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_accuracy_binary_decimal = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    
    binary_stats_data = [
        ["True Positives (TP)", total_tp],
        ["False Positives (FP)", total_fp],
        ["True Negatives (TN)", total_tn],
        ["False Negatives (FN)", total_fn],
        ["Precision", f"{overall_precision:.4f}"],
        ["Recall/Sensitivity", f"{overall_recall:.4f}"],
        ["Specificity", f"{overall_specificity:.4f}"],
        ["F1 Score", f"{overall_f1:.4f}"],
        ["One-vs-All Accuracy", f"{overall_accuracy_binary_decimal:.4f}"]
    ]
    
    content += f"=== One-vs-All Binary Classification Metrics ===\n"
    content += f"""
Note: These metrics are calculated by treating each {level_name.lower()} as a separate binary classification problem 
(target {level_name.lower()} vs. all other {level_name.lower()}s + negative controls), then aggregating results across all {level_name.lower()}s.

"""
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

def generate_roc_curve(results_df, output_dir, level):
    """
    Generates a sensitivity vs specificity plot against probability thresholds,
    matching the style shown in the reference image.
    """
    if 'Confidence' not in results_df.columns:
        print("Warning: 'Confidence' column not found. Cannot generate sensitivity/specificity curve.")
        return

    # Prepare the data for sensitivity/specificity calculation
    # For this plot, we need to determine what constitutes a "positive" prediction
    # We'll use whether the prediction was correct as the binary outcome
    y_true_binary = (results_df['True_Label'] == results_df['Predicted_Label']).astype(int)
    y_scores = results_df['Confidence']

    if len(y_true_binary.unique()) < 2:
        print("Warning: Only one class present. Cannot generate sensitivity/specificity curve.")
        return

    # Calculate sensitivity and specificity at different thresholds
    thresholds = np.linspace(0, 1, 101)  # 0.00 to 1.00 in steps of 0.01
    sensitivity_values = []
    specificity_values = []
    
    for threshold in thresholds:
        # Predictions above threshold are considered "positive" (confident)
        y_pred_binary = (y_scores >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Calculate sensitivity (recall) and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivity_values.append(sensitivity)
        specificity_values.append(specificity)
    
    # Convert to numpy arrays
    sensitivity_values = np.array(sensitivity_values)
    specificity_values = np.array(specificity_values)
    
    # Find optimal threshold (closest to top-left corner, or where sensitivity + specificity is maximized)
    optimal_idx = np.argmax(sensitivity_values + specificity_values)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = sensitivity_values[optimal_idx]
    optimal_specificity = specificity_values[optimal_idx]
    
    # Calculate overall accuracy at optimal threshold
    y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
    optimal_accuracy = np.mean(y_true_binary == y_pred_optimal)
    
    # Create the plot matching the reference style
    plt.figure(figsize=(10, 8))
    
    # Plot sensitivity line (blue with circles)
    plt.plot(thresholds, sensitivity_values, 'o-', color='blue', linewidth=2, 
             markersize=4, label='Sensitivity', markerfacecolor='blue')
    
    # Plot specificity line (red/maroon with circles)
    plt.plot(thresholds, specificity_values, 'o-', color='darkred', linewidth=2, 
             markersize=4, label='Specificity', markerfacecolor='darkred')
    
    # Add vertical line at optimal threshold
    plt.axvline(x=optimal_threshold, color='gray', linestyle='--', alpha=0.7, 
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    
    # Styling to match the reference image
    plt.xlabel('Probability cutoff', fontsize=12, fontweight='bold')
    plt.ylabel('Sensitivity/Specificity', fontsize=12, fontweight='bold')
    plt.title(f'Sensitivity vs Specificity Analysis ({level.capitalize()}-Level Classification)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Set axis limits and ticks to clearly show 0.0 points
    plt.xlim(-0.02, 1.02)  # Slightly expand to show 0.0 clearly
    plt.ylim(-0.02, 1.02)  # Slightly expand to show 0.0 clearly
    
    # Set ticks to explicitly include 0.0
    x_ticks = np.arange(0.0, 1.1, 0.25)
    y_ticks = np.arange(0.0, 1.1, 0.25)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    
    # Add grid that includes the 0.0 lines
    plt.grid(True, alpha=0.3, which='major')
    
    # Add explicit lines at x=0 and y=0 to emphasize the origin
    plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
    plt.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)
    
    # Add legend
    plt.legend(loc='center right', fontsize=11)
    
    # Add performance metrics text at the bottom
    metrics_text = f"Sensitivity {optimal_sensitivity*100:.1f}% Specificity {optimal_specificity*100:.1f}%\n"
    metrics_text += f"Correctly Classified {optimal_accuracy*100:.0f}%"
    
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=12, fontweight='bold')
    
    # Adjust layout to make room for the text
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'sensitivity_specificity_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sensitivity/Specificity curve saved to {plot_path}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"At optimal threshold - Sensitivity: {optimal_sensitivity*100:.1f}%, Specificity: {optimal_specificity*100:.1f}%")
    print(f"Overall accuracy: {optimal_accuracy*100:.1f}%")
    
    # Also generate the traditional ROC curve for comparison
    generate_traditional_roc_curve(results_df, output_dir, level)

def generate_traditional_roc_curve(results_df, output_dir, level):
    """
    Generates a traditional ROC curve (TPR vs FPR) for comparison.
    """
    from sklearn.metrics import roc_curve, auc
    
    y_true_binary = (results_df['True_Label'] == results_df['Predicted_Label']).astype(int)
    y_scores = results_df['Confidence']

    if len(y_true_binary.unique()) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Set axis limits and ticks to clearly show 0.0 points
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.05)
    
    # Set ticks to explicitly include 0.0
    x_ticks = np.arange(0.0, 1.1, 0.2)
    y_ticks = np.arange(0.0, 1.1, 0.2)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve: Model Confidence vs. Classification Correctness ({level.capitalize()})', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    
    # Add grid that includes the 0.0 lines
    plt.grid(True, alpha=0.3, which='major')
    plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
    plt.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'roc_curve_traditional.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Traditional ROC curve saved to {plot_path}") 