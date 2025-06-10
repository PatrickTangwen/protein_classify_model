import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add refactored_scripts to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'refactored_scripts'))
from config import RESULTS_DIR


def generate_benchmark_plot(level, output_dir=None):
    """
    Generates a vertical bar chart comparing the performance of all trained models.

    Args:
        level (str): The classification level ('family' or 'subfamily').
        output_dir (str): Custom output directory (optional).
    
    Returns:
        str: Path to the saved plot file, or None if failed.
    """
    results_level_dir = os.path.join(RESULTS_DIR, level)
    
    performance_data = []

    if not os.path.exists(results_level_dir):
        print(f"Error: Results directory for level '{level}' not found at '{results_level_dir}'.")
        print("Please run the `run_automl.py` script first.")
        return None

    # Scan for model result directories
    for model_name in os.listdir(results_level_dir):
        model_dir = os.path.join(results_level_dir, model_name)
        summary_file = os.path.join(model_dir, 'summary_metrics.json')
        
        if os.path.isfile(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    metrics = json.load(f)
                    accuracy = metrics.get('overall_accuracy_original_set')
                    if accuracy is not None:
                        performance_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Accuracy': accuracy
                        })
                        print(f"  - Found {model_name}: {accuracy:.2f}% accuracy")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read or parse {summary_file}. Error: {e}")

    if not performance_data:
        print("Error: No valid model performance data found. Cannot generate plot.")
        print(f"Please ensure `summary_metrics.json` files exist in subdirectories of: {results_level_dir}")
        return None

    # Create DataFrame and sort by accuracy for consistent plotting order
    perf_df = pd.DataFrame(performance_data).sort_values('Accuracy', ascending=False)
    
    # --- Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 7))
    
    barplot = sns.barplot(x='Model', y='Accuracy', data=perf_df, palette='viridis', width=0.6)
    
    plt.title(f'Model Performance Benchmark ({level.capitalize()}-Level Classification)', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Accuracy (Original Test Set, %)', fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha='center')

    # Add accuracy labels to the top of the bars
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', 
                           xytext=(0, 9), 
                           textcoords='offset points',
                           fontweight='bold',
                           fontsize=11)
    
    plt.tight_layout()

    # --- Save the plot ---
    if output_dir is None:
        benchmark_dir = os.path.join(RESULTS_DIR, 'benchmark')
    else:
        benchmark_dir = output_dir
    
    os.makedirs(benchmark_dir, exist_ok=True)
    plot_path = os.path.join(benchmark_dir, f'{level}_comparison.png')
    
    plt.savefig(plot_path, dpi=300)
    print(f"\nBenchmark plot saved to: {plot_path}")
    
    plt.close()
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(
        description="Generate vertical benchmark plots for AutoML results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--level', 
        type=str, 
        required=True, 
        choices=['subfamily', 'family'],
        help="The classification level to plot."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help="Custom output directory for the plot (optional)."
    )

    args = parser.parse_args()
    
    print("="*60)
    print(f"Generating Benchmark Plot for {args.level.capitalize()}-Level")
    print("="*60)
    
    plot_path = generate_benchmark_plot(
        level=args.level,
        output_dir=args.output
    )
    
    if plot_path:
        print("\n" + "="*60)
        print("Benchmark plot generation completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Failed to generate benchmark plot.")
        print("="*60)


if __name__ == "__main__":
    main() 