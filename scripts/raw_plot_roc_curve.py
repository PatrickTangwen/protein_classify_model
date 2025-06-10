import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# --- Path Configuration ---
# This script can work with either subfamily-level or family-level results.
# It will automatically detect and process both types if available.

def find_all_results_files():
    """
    Find all available results files.
    Returns a list of (file_path, results_type) tuples
    """
    possible_locations = [
        # Subfamily results (default location)
        (os.path.join('..', 'model_results', 'detailed_classification_results.csv'), 'subfamily'),
        # Family results
        (os.path.join('..', 'model_results_family', 'detailed_classification_results_family.csv'), 'family'),
        # Alternative: if script is run from project root
        (os.path.join('model_results', 'detailed_classification_results.csv'), 'subfamily'),
        (os.path.join('model_results_family', 'detailed_classification_results_family.csv'), 'family'),
    ]
    
    found_files = []
    for file_path, results_type in possible_locations:
        if os.path.exists(file_path):
            found_files.append((file_path, results_type))
    
    return found_files

def process_results_file(file_path, results_type):
    """
    Process a single results file and generate ROC plot.
    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Processing {results_type}-level results")
    print(f"{'='*60}")
    
    # Load the dataset
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from: {file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return False
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return False

    # Determine which columns to use based on results type and available columns
    true_col = None
    pred_col = None

    if results_type == 'subfamily':
        # For subfamily-level results
        if 'True_Subfamily' in df.columns and 'Predicted_Subfamily' in df.columns:
            true_col = 'True_Subfamily'
            pred_col = 'Predicted_Subfamily'
            classification_level = 'Subfamily'
        elif 'True_Family' in df.columns and 'Predicted_Family' in df.columns:
            # Fallback to family-level comparison if subfamily columns not available
            true_col = 'True_Family'
            pred_col = 'Predicted_Family'
            classification_level = 'Family'
            print("Warning: Subfamily columns not found, using Family-level comparison")
    else:
        # For family-level results
        if 'True_Family' in df.columns and 'Predicted_Family' in df.columns:
            true_col = 'True_Family'
            pred_col = 'Predicted_Family'
            classification_level = 'Family'
        else:
            print("Error: Expected columns 'True_Family' and 'Predicted_Family' not found in family results")
            print(f"Available columns: {list(df.columns)}")
            return False

    if true_col is None or pred_col is None:
        print(f"Error: Could not determine appropriate columns for {results_type}-level analysis")
        print(f"Available columns: {list(df.columns)}")
        return False

    print(f"Using {classification_level}-level comparison: {true_col} vs {pred_col}")

    # Create a binary target variable: 1 if prediction is correct, 0 otherwise.
    # Ensure string comparisons are robust (e.g., handle potential whitespace or type differences)
    df[true_col] = df[true_col].astype(str).str.strip()
    df[pred_col] = df[pred_col].astype(str).str.strip()
    df['is_correct'] = (df[true_col] == df[pred_col]).astype(int)

    # Use the 'Confidence' column as the prediction score.
    # Ensure 'Confidence' is numeric and handle potential missing values or non-numeric entries.
    if 'Confidence' not in df.columns:
        print("Error: 'Confidence' column not found in the dataset")
        print(f"Available columns: {list(df.columns)}")
        return False

    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=['Confidence', 'is_correct'], inplace=True)
    final_rows = len(df)

    if initial_rows != final_rows:
        print(f"Dropped {initial_rows - final_rows} rows due to missing confidence or correctness values")

    y_true_binary = df['is_correct']
    y_scores_confidence = df['Confidence']

    print(f"Final dataset for ROC analysis: {len(df)} samples")
    print(f"Correct predictions: {y_true_binary.sum()} ({y_true_binary.mean():.2%})")
    print(f"Incorrect predictions: {len(y_true_binary) - y_true_binary.sum()} ({1 - y_true_binary.mean():.2%})")

    # Check if there are samples in both classes for ROC calculation
    if len(y_true_binary.unique()) < 2:
        if len(y_true_binary) == 0:
            print("ROC AUC not defined: No data left after filtering for 'Confidence' and 'is_correct'.")
        else:
            print(f"ROC AUC not defined: Only one class present in 'is_correct' ({y_true_binary.unique()[0]}) after filtering.")
            print(f"Total valid samples for ROC: {len(y_true_binary)}")
        return False
    else:
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores_confidence)
        roc_auc = auc(fpr, tpr)

        print(f"AUC for ({classification_level} Classification Correctness vs. Confidence): {roc_auc:.3f}")

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC: {classification_level}-Level Classification Correctness vs. Confidence Score')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Saving the plot
        # Determine output directory based on input file location
        output_dir = os.path.dirname(file_path)
        
        # Ensure the directory exists before saving
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            except OSError as e:
                print(f"Error creating directory {output_dir}: {e}")
                # Fallback to current directory if creation fails
                output_dir = "."

        plot_filename = f'roc_confidence_correctness_{results_type}.png'
        full_plot_path = os.path.join(output_dir, plot_filename)
        
        try:
            plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
            print(f"ROC plot saved to: {full_plot_path}")
        except Exception as e:
            print(f"Error saving ROC plot to {full_plot_path}: {e}")
            return False
        
        plt.close()  # Close the figure to free memory
        
        # plt.show() # Uncomment if running in an environment that supports showing plots interactively
        
        return True

# Main execution
print("ROC Curve Generator for Protein Classification Results")
print("="*60)

# Try to find all available results files
available_files = find_all_results_files()

if not available_files:
    print("Error: Could not find any results files in expected locations:")
    print("- ../model_results/detailed_classification_results.csv (subfamily results)")
    print("- ../model_results_family/detailed_classification_results_family.csv (family results)")
    print("- model_results/detailed_classification_results.csv (subfamily results)")
    print("- model_results_family/detailed_classification_results_family.csv (family results)")
    print("\nPlease ensure you have run either the subfamily or family neural network script first.")
    exit()

print(f"Found {len(available_files)} results file(s):")
for file_path, results_type in available_files:
    print(f"  - {results_type}-level: {file_path}")

# Process each available results file
successful_processes = 0
total_processes = len(available_files)

for file_path, results_type in available_files:
    success = process_results_file(file_path, results_type)
    if success:
        successful_processes += 1

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Total files processed: {total_processes}")
print(f"Successful ROC plots generated: {successful_processes}")
print(f"Failed processes: {total_processes - successful_processes}")

if successful_processes > 0:
    print(f"\nSuccessfully generated ROC plots for:")
    for file_path, results_type in available_files:
        output_dir = os.path.dirname(file_path)
        plot_filename = f'roc_confidence_correctness_{results_type}.png'
        full_plot_path = os.path.join(output_dir, plot_filename)
        if os.path.exists(full_plot_path):
            print(f"  - {results_type}-level: {full_plot_path}")

print("\nScript finished.")