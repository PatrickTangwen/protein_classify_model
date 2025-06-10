import argparse
import os
import numpy as np
import sys

# Add refactored_scripts to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'refactored_scripts'))

import config
from data_loader import load_protein_data, load_superfamily_map
from feature_engineering import build_features
from data_splitting import custom_split_dataset_with_negatives
from models import MODELS
from training import train_and_evaluate_model
from evaluation import get_predictions, evaluate_model_detailed, save_reports, generate_roc_curve
from plot import generate_benchmark_plot

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for protein classification.")
    parser.add_argument('--level', type=str, required=True, choices=['subfamily', 'family'],
                        help="The classification level to run.")
    parser.add_argument('--model', type=str, default='all',
                        help="The model to run. Use 'all' or a specific model key from models.py.")

    args = parser.parse_args()
    
    print("="*80)
    print(f"AutoML Protein Classification - {args.level.capitalize()} Level")
    print("="*80)
    
    # --- 1. Load Data ---
    print("\n=== Step 1: Loading Data ===")
    df = load_protein_data(config.PROTEIN_DATA_PATH, level=args.level)
    superfamily_map = load_superfamily_map(config.SUPERFAMILY_MAP_PATH)
    print(f"Data loaded for {args.level}-level classification.")
    print(f"Dataset shape: {df.shape}")

    # --- 2. Feature Engineering ---
    print("\n=== Step 2: Building Features ===")
    X, y, label_encoder, domain_vocab, feature_stats = build_features(df, level=args.level, max_domains=config.MAX_DOMAINS)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    # --- 3. Data Splitting ---
    print("\n=== Step 3: Data Splitting and Negative Control Generation ===")
    train_indices, val_indices, is_negative_control, target_test_mapping = \
        custom_split_dataset_with_negatives(df, superfamily_map, level=args.level)
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)} (includes negative controls)")
    print(f"Number of negative controls: {sum(is_negative_control.values())}")

    # --- 4. Model Training & Evaluation ---
    print("\n=== Step 4: Model Training and Evaluation ===")
    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    
    if args.model != 'all' and args.model not in MODELS:
        print(f"Error: Model '{args.model}' not found in available models: {list(MODELS.keys())}")
        return
    
    print(f"Models to run: {models_to_run}")
    
    for model_name in models_to_run:
        print("\n" + "="*50)
        print(f"Processing model: {model_name}")
        print("="*50)
        
        # Create model-specific output directory
        model_output_dir = os.path.join(config.RESULTS_DIR, args.level, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"Output directory: {model_output_dir}")
        
        # Get model configuration
        model_info = MODELS[model_name].copy()
        
        # Clone sklearn models to avoid state issues
        if model_info['type'] == 'sklearn':
            from sklearn.base import clone
            model_info['model'] = clone(model_info['model'])
        
        # Prepare data split information
        data_split = {
            'X_train': X_train, 
            'y_train': y_train, 
            'X_val': X_val, 
            'y_val': y_val,
            'num_classes': len(label_encoder.classes_),
            'input_dim': X.shape[1]
        }
        
        # Train model
        print(f"Training {model_name}...")
        trained_model = train_and_evaluate_model(model_name, model_info, data_split, model_output_dir)
        
        # Get predictions
        print(f"Generating predictions...")
        predictions, confidences = get_predictions(trained_model, model_info['type'], X_val)
        
        # Detailed evaluation
        print(f"Performing detailed evaluation...")
        report, results_df = evaluate_model_detailed(
            df, predictions, confidences, label_encoder, val_indices,
            train_indices, is_negative_control, target_test_mapping,
            args.level, superfamily_map
        )
        
        # Save all reports
        print(f"Saving reports...")
        save_reports(df, report, results_df, target_test_mapping, train_indices, model_output_dir, args.level)
        
        # Generate ROC curve
        print(f"Generating ROC curve...")
        generate_roc_curve(results_df, model_output_dir, args.level)
        
        # Print summary
        total_test = sum(m['test_count'] for m in report.values())
        total_correct = sum(m['correct'] for m in report.values())
        overall_accuracy = (total_correct / total_test * 100) if total_test > 0 else 0
        print(f"Model {model_name} completed!")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

    # --- 5. Benchmarking ---
    if len(models_to_run) > 1:
        print("\n" + "="*50)
        print("Step 5: Generating Performance Benchmark")
        print("="*50)
        print(f"Generating performance benchmark plot for {args.level} level...")
        benchmark_plot_path = generate_benchmark_plot(args.level)
        print(f"Benchmark plot saved: {benchmark_plot_path}")

    print("\n" + "="*80)
    print("Benchmark Pipeline Completed Successfully!")
    print("="*80)
    print(f"Results saved to: {os.path.join(config.RESULTS_DIR, args.level)}")
    print("Files generated for each model:")
    print("  - classification_report.txt")
    print("  - classification_stats.txt") 
    print("  - detailed_classification_results.csv")
    print("  - binary_classification_metrics.csv")
    print("  - summary_metrics.json")
    print("  - roc_curve.png")

if __name__ == "__main__":
    main() 