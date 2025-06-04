import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# --- Path Configuration ---
# Adjust this relative path if your script's location differs.
# This assumes the script is in a subdirectory (e.g., 'scripts'),
# and 'model_results' is in the parent directory (the project root).
# If the script is in the project root, change path_to_model_results_dir to:
# path_to_model_results_dir = 'model_results'
path_to_model_results_dir = os.path.join('..', 'model_results') # Default for script in a subfolder

csv_filename = 'detailed_classification_results.csv'
full_csv_path = os.path.join(path_to_model_results_dir, csv_filename)
# --- End Path Configuration ---

# 1. Load the dataset
print(f"Attempting to load data from: {full_csv_path}")
try:
    df = pd.read_csv(full_csv_path)
    print(f"Successfully loaded data from: {full_csv_path}")
except FileNotFoundError:
    print(f"Error: File not found at {full_csv_path}")
    print("Please ensure the 'path_to_model_results_dir' and 'csv_filename' are correct,")
    print("and that the script is run from the intended directory relative to your data.")
    exit() # Exit if the crucial input file is not found
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# 2. Create a binary target variable: 1 if prediction is correct, 0 otherwise.
# Ensure string comparisons are robust (e.g., handle potential whitespace or type differences)
df['True_Subfamily'] = df['True_Subfamily'].astype(str).str.strip()
df['Predicted_Subfamily'] = df['Predicted_Subfamily'].astype(str).str.strip()
df['is_correct'] = (df['True_Subfamily'] == df['Predicted_Subfamily']).astype(int)

# 3. Use the 'Confidence' column as the prediction score.
# Ensure 'Confidence' is numeric and handle potential missing values or non-numeric entries.
df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
df.dropna(subset=['Confidence', 'is_correct'], inplace=True) # Drop rows where confidence or target is NaN

y_true_binary = df['is_correct']
y_scores_confidence = df['Confidence']

# Check if there are samples in both classes for ROC calculation
if len(y_true_binary.unique()) < 2:
    if len(y_true_binary) == 0:
        print("ROC AUC not defined: No data left after filtering for 'Confidence' and 'is_correct'.")
    else:
        print(f"ROC AUC not defined: Only one class present in 'is_correct' ({y_true_binary.unique()[0]}) after filtering.")
        print(f"Total valid samples for ROC: {len(y_true_binary)}")

else:
    # 4. Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores_confidence)
    roc_auc = auc(fpr, tpr)

    print(f"AUC for (Correctness of Classification vs. Confidence): {roc_auc:.2f}")

    # 5. Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC: Classification Correctness vs. Confidence Score')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # --- Saving the plot ---
    # Ensure the directory exists before saving
    if not os.path.exists(path_to_model_results_dir):
        try:
            os.makedirs(path_to_model_results_dir)
            print(f"Created directory: {path_to_model_results_dir}")
        except OSError as e:
            print(f"Error creating directory {path_to_model_results_dir}: {e}")
            # Fallback to current directory if creation fails
            path_to_model_results_dir = "." 


    plot_filename = 'roc_confidence_correctness.png'
    full_plot_path = os.path.join(path_to_model_results_dir, plot_filename)
    
    try:
        plt.savefig(full_plot_path)
        print(f"ROC plot saved to: {full_plot_path}")
    except Exception as e:
        print(f"Error saving ROC plot to {full_plot_path}: {e}")
    
    # plt.show() # Uncomment if running in an environment that supports showing plots interactively
    # --- End Saving the plot ---

print("Script finished.")