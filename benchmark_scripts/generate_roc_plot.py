#!/usr/bin/env python3
"""
Standalone script to generate sensitivity/specificity curves from existing benchmark results.
This script reads the detailed_classification_results.csv files and generates the plots
without re-running the entire benchmark pipeline.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the benchmark_scripts directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def generate_sensitivity_specificity_plot(results_file, output_dir, model_name, level):
    """
    Generate sensitivity/specificity plot from results CSV file.
    
    Args:
        results_file (str): Path to detailed_classification_results.csv
        output_dir (str): Directory to save the plot
        model_name (str): Name of the model (for plot title)
        level (str): Classification level ('family' or 'subfamily')
    """
    try:
        # Read the results file
        results_df = pd.read_csv(results_file)
        
        # Check required columns
        required_cols = ['Confidence']
        level_col = f'True_{level.capitalize()}'
        pred_col = f'Predicted_{level.capitalize()}'
        
        if level_col not in results_df.columns or pred_col not in results_df.columns:
            print(f"Error: Required columns not found in {results_file}")
            print(f"Expected: {level_col}, {pred_col}, Confidence")
            print(f"Found: {list(results_df.columns)}")
            return False
            
        if 'Confidence' not in results_df.columns:
            print(f"Warning: 'Confidence' column not found in {results_file}")
            return False

        # Prepare the data for sensitivity/specificity calculation
        y_true_binary = (results_df[level_col] == results_df[pred_col]).astype(int)
        y_scores = results_df['Confidence']

        if len(y_true_binary.unique()) < 2:
            print(f"Warning: Only one class present in {results_file}. Cannot generate curve.")
            return False

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
        
        # Find optimal threshold
        optimal_idx = np.argmax(sensitivity_values + specificity_values)
        optimal_threshold = thresholds[optimal_idx]
        optimal_sensitivity = sensitivity_values[optimal_idx]
        optimal_specificity = specificity_values[optimal_idx]
        
        # Calculate overall accuracy at optimal threshold
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        optimal_accuracy = np.mean(y_true_binary == y_pred_optimal)
        
        # Create the plot
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
        
        # Styling
        plt.xlabel('Probability cutoff', fontsize=12, fontweight='bold')
        plt.ylabel('Sensitivity/Specificity', fontsize=12, fontweight='bold')
        plt.title(f'Sensitivity vs Specificity: {model_name.replace("_", " ").title()} ({level.capitalize()}-Level)', 
                  fontsize=14, fontweight='bold', pad=20)
        
        # Set axis limits and ticks to clearly show 0.0 points
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        
        # Set ticks to explicitly include 0.0
        x_ticks = np.arange(0.0, 1.1, 0.25)
        y_ticks = np.arange(0.0, 1.1, 0.25)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        
        # Add grid that includes the 0.0 lines
        plt.grid(True, alpha=0.3, which='major')
        plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
        plt.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)
        
        # Add legend
        plt.legend(loc='center right', fontsize=11)
        
        # Add performance metrics text at the bottom
        metrics_text = f"Sensitivity {optimal_sensitivity*100:.1f}% Specificity {optimal_specificity*100:.1f}%\n"
        metrics_text += f"Correctly Classified {optimal_accuracy*100:.0f}%"
        
        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f'sensitivity_specificity_{model_name}_{level}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved: {plot_path}")
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  Sensitivity: {optimal_sensitivity*100:.1f}%, Specificity: {optimal_specificity*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error processing {results_file}: {str(e)}")
        return False

def find_results_files(base_dir, level, models=None):
    """
    Find all detailed_classification_results files for the specified level.
    
    Args:
        base_dir (str): Base results directory
        level (str): Classification level ('family' or 'subfamily')
        models (list): List of specific models to process (None for all)
    
    Returns:
        list: List of tuples (model_name, results_file_path)
    """
    results_files = []
    level_dir = os.path.join(base_dir, level)
    
    if not os.path.exists(level_dir):
        print(f"Error: Level directory not found: {level_dir}")
        return results_files
    
    for model_name in os.listdir(level_dir):
        if models and model_name not in models:
            continue
            
        model_dir = os.path.join(level_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
            
        # Look for the results file
        if level == 'family':
            results_file = os.path.join(model_dir, 'detailed_classification_results_family.csv')
        else:
            results_file = os.path.join(model_dir, 'detailed_classification_results.csv')
            
        if os.path.exists(results_file):
            results_files.append((model_name, results_file))
        else:
            print(f"Warning: Results file not found for {model_name}: {results_file}")
    
    return results_files

def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity/specificity curves from existing benchmark results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--level', 
        type=str, 
        required=True, 
        choices=['subfamily', 'family'],
        help="Classification level to process"
    )
    
    parser.add_argument(
        '--models', 
        type=str, 
        nargs='*',
        help="Specific models to process (default: all available models)"
    )
    
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default=None,
        help=f"Input directory containing results (default: {config.RESULTS_DIR})"
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help="Output directory for plots (default: <input-dir>/sensitivity_specificity_plots)"
    )
    
    args = parser.parse_args()
    
    # Set default directories
    input_dir = args.input_dir or config.RESULTS_DIR
    output_dir = args.output_dir or os.path.join(input_dir, 'sensitivity_specificity_plots')
    
    print("="*70)
    print(f"Generating Sensitivity/Specificity Plots - {args.level.capitalize()} Level")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all results files
    results_files = find_results_files(input_dir, args.level, args.models)
    
    if not results_files:
        print(f"\nNo results files found for {args.level} level.")
        print("Make sure you have run the benchmark pipeline first:")
        print(f"  python benchmark_scripts/run_benchmark.py --level {args.level} --model all")
        return
    
    print(f"\nFound {len(results_files)} model(s) to process:")
    for model_name, _ in results_files:
        print(f"  - {model_name}")
    
    # Generate plots
    print(f"\nGenerating plots...")
    success_count = 0
    
    for model_name, results_file in results_files:
        print(f"\nProcessing {model_name}...")
        if generate_sensitivity_specificity_plot(results_file, output_dir, model_name, args.level):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"Completed: {success_count}/{len(results_files)} plots generated successfully")
    print(f"Plots saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()