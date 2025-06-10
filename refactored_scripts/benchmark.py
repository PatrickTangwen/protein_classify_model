import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

def generate_benchmark_plot(level):
    """
    Generates a vertical bar chart comparing the performance of all trained models.

    Args:
        level (str): The classification level ('family' or 'subfamily').
    """
    results_level_dir = os.path.join(config.RESULTS_DIR, level)
    
    performance_data = []

    if not os.path.exists(results_level_dir):
        print(f"Results directory for level '{level}' not found. Cannot generate benchmark plot.")
        return

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
                    else:
                        print(f"Warning: 'overall_accuracy_original_set' not found in {summary_file}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read or parse {summary_file}. Error: {e}")

    if not performance_data:
        print("No valid model performance data found. Cannot generate plot.")
        return

    # Create DataFrame and sort by accuracy
    perf_df = pd.DataFrame(performance_data).sort_values('Accuracy', ascending=False)
    
    # Generate plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 7))
    barplot = sns.barplot(x='Model', y='Accuracy', data=perf_df, palette='viridis', width=0.6)
    
    plt.title(f'Model Performance Benchmark ({level.capitalize()}-Level)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Accuracy (Original Test Set, %)', fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha='center')
    
    # Add accuracy labels to the bars
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', 
                           xytext=(0, 9), 
                           textcoords='offset points',
                           fontweight='bold',
                           fontsize=11)

    # Save the plot
    benchmark_dir = os.path.join(config.RESULTS_DIR, 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    plot_path = os.path.join(benchmark_dir, f'{level}_comparison.png')
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nBenchmark plot saved to: {plot_path}")
    print(perf_df.to_string(index=False)) 