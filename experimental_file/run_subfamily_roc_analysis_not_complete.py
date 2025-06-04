# File: scripts/neural_network_subfamily.py (or a new file like run_subfamily_roc_analysis.py)
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from collections import Counter
import logging

# --- Existing Logger Class (from your script structure) ---
class Logger:
    def __init__(self, filename="logfile.log", classification_report_file=None, verbose=True):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding='utf-8')
        self.classification_report_file = None
        if classification_report_file:
            self.classification_report_file = open(classification_report_file, "w", encoding='utf-8')
        self.verbose = verbose

    def write(self, message, is_classification_report=False):
        if self.verbose:
            self.terminal.write(message)
            self.terminal.flush()

        self.log_file.write(message)
        self.log_file.flush()

        if is_classification_report and self.classification_report_file:
            self.classification_report_file.write(message)
            self.classification_report_file.flush()

    def flush(self):
        if self.verbose:
            self.terminal.flush()
        self.log_file.flush()
        if self.classification_report_file:
            self.classification_report_file.flush()

    def close(self):
        self.log_file.close()
        if self.classification_report_file:
            self.classification_report_file.close()

# --- Existing ProteinDataset Class (from your script structure) ---
class ProteinDataset(Dataset):
    def __init__(self, df, label_encoder, max_domains=50, is_predicting=False):
        self.df = df
        self.max_domains = max_domains
        self.is_predicting = is_predicting
        self.label_encoder = label_encoder

        # Precompute features and labels if not predicting
        if not self.is_predicting:
            self.labels = self.label_encoder.transform(self.df['subfamily'].astype(str))
        else:
            # For prediction, we might not have 'subfamily'. If it exists, we can encode it.
            if 'subfamily' in self.df.columns:
                try:
                    self.labels = self.label_encoder.transform(self.df['subfamily'].astype(str))
                except ValueError as e: # Handle unseen labels during prediction if needed
                    # print(f"Warning: Unseen labels in prediction data: {e}")
                    # Create dummy labels or handle as appropriate
                    self.labels = np.full(len(self.df), -1) # Or some other placeholder
            else:
                 self.labels = np.full(len(self.df), -1) # Placeholder if no labels column

        # Feature extraction logic (simplified, ensure it matches your original)
        # This example assumes 'domains', 'domain_starts', 'domain_ends' are list-like strings
        all_features = []
        for _, row in df.iterrows():
            num_domains = row.get('num_domains', 0)
            normalized_starts = np.zeros(self.max_domains)
            normalized_ends = np.zeros(self.max_domains)
            protein_length = row.get('length', 1) # Avoid division by zero if length is missing or 0

            if protein_length == 0: protein_length = 1 # Safeguard

            domain_starts_str = row.get('domain_starts', '[]')
            domain_ends_str = row.get('domain_ends', '[]')

            try:
                # Convert string representations of lists to actual lists of numbers
                # Ensure robust parsing for various string formats (e.g., "[]", "[1, 2]", "[1.0, 2.0]")
                actual_domain_starts = [float(x) for x in domain_starts_str.strip('[]').split(',') if x.strip()] if domain_starts_str.strip('[]') else []
                actual_domain_ends = [float(x) for x in domain_ends_str.strip('[]').split(',') if x.strip()] if domain_ends_str.strip('[]') else []

            except ValueError: # Handle cases where conversion might fail
                actual_domain_starts = []
                actual_domain_ends = []


            for i in range(min(int(num_domains), self.max_domains)):
                if i < len(actual_domain_starts):
                    normalized_starts[i] = actual_domain_starts[i] / protein_length
                if i < len(actual_domain_ends):
                    normalized_ends[i] = actual_domain_ends[i] / protein_length

            features = np.concatenate(([num_domains / self.max_domains], normalized_starts, normalized_ends))
            all_features.append(features)
        
        self.features = np.array(all_features, dtype=np.float32)
        self.input_dim = self.features.shape[1]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current_features = self.features[idx]
        if self.is_predicting:
            # For prediction, we might only return features, or features and a placeholder label
             return torch.tensor(current_features, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            current_label = self.labels[idx]
            return torch.tensor(current_features, dtype=torch.float32), torch.tensor(current_label, dtype=torch.long)

# --- Existing ImprovedProteinClassifier Class (from your script structure) ---
class ImprovedProteinClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout_rate=0.5):
        super(ImprovedProteinClassifier, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights(self)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)

# --- Existing custom_split_dataset_with_negatives (from your context) ---
def custom_split_dataset_with_negatives(df, test_size=0.2, val_size=0.1, random_state=42, group_col='protein_id'):
    """
    Splits the dataset into training, validation, and test sets, ensuring that
    proteins from the same group (e.g., same protein_id) are not split across sets.
    Handles negative controls by ensuring they are distributed.
    Returns dataframes and a fitted label encoder.
    """
    df_copy = df.copy()
    df_copy['subfamily'] = df_copy['subfamily'].astype(str) # Ensure subfamily is string for label encoding

    # Separate definite positives and definite negatives
    positive_df = df_copy[df_copy['is_negative_control'] == False]
    negative_df = df_copy[df_copy['is_negative_control'] == True]

    # Initialize LabelEncoder and fit on all potential subfamilies from positive_df
    # This ensures the encoder knows all "true" classes.
    # Negative controls will be mapped; their 'subfamily' names might be varied.
    label_encoder = LabelEncoder()
    all_known_subfamilies = positive_df['subfamily'].unique().tolist()
    # Add any subfamily names from negative controls that might not be in positives, if necessary
    # For simplicity, assuming negative control 'subfamily' names are either among positives or distinct placeholders
    for neg_subfam in negative_df['subfamily'].unique():
        if neg_subfam not in all_known_subfamilies:
            all_known_subfamilies.append(neg_subfam)
    label_encoder.fit(all_known_subfamilies)


    # Split positive samples
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx_pos, test_idx_pos = next(splitter_test.split(positive_df, groups=positive_df[group_col]))
    train_val_df_pos = positive_df.iloc[train_val_idx_pos]
    test_df_pos = positive_df.iloc[test_idx_pos]

    # Adjust val_size relative to the size of train_val_df_pos
    val_size_adjusted = val_size / (1 - test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    train_idx_pos, val_idx_pos = next(splitter_val.split(train_val_df_pos, groups=train_val_df_pos[group_col]))
    train_df_pos = train_val_df_pos.iloc[train_idx_pos]
    val_df_pos = train_val_df_pos.iloc[val_idx_pos]

    # Split negative samples (if any)
    if not negative_df.empty:
        # Ensure negative_df has group_col, if not, use index or a default group
        if group_col not in negative_df.columns:
             negative_df[group_col] = negative_df.index # Simple grouping for negatives if no protein_id

        splitter_test_neg = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx_neg, test_idx_neg = next(splitter_test_neg.split(negative_df, groups=negative_df[group_col]))
        train_val_df_neg = negative_df.iloc[train_val_idx_neg]
        test_df_neg = negative_df.iloc[test_idx_neg]

        val_size_adjusted_neg = val_size / (1-test_size)
        if not train_val_df_neg.empty:
            splitter_val_neg = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted_neg, random_state=random_state)
            train_idx_neg, val_idx_neg = next(splitter_val_neg.split(train_val_df_neg, groups=train_val_df_neg[group_col]))
            train_df_neg = train_val_df_neg.iloc[train_idx_neg]
            val_df_neg = train_val_df_neg.iloc[val_idx_neg]
        else: # Handle case where train_val_df_neg might be empty after first split
            train_df_neg = pd.DataFrame(columns=df_copy.columns)
            val_df_neg = pd.DataFrame(columns=df_copy.columns)


        # Combine positive and negative splits
        train_df = pd.concat([train_df_pos, train_df_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_df = pd.concat([val_df_pos, val_df_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df = pd.concat([test_df_pos, test_df_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    else: # No negative controls
        train_df = train_df_pos.reset_index(drop=True)
        val_df = val_df_pos.reset_index(drop=True)
        test_df = test_df_pos.reset_index(drop=True)

    # Store original indices for later reference if needed (primarily for test set here)
    test_indices = test_df.index.tolist() # These are indices from the combined test_df

    return train_df, val_df, test_df, label_encoder, test_indices


# --- Function to Plot Multi-Class ROC Curves ---
def plot_multiclass_roc_from_data(y_true_encoded, y_score, class_names, output_plot_file):
    """
    Generates and saves a multi-class ROC plot from provided data.
    y_true_encoded: 1D array of true integer labels.
    y_score: 2D array of probability scores (n_samples, n_classes).
    class_names: List/array of class names.
    output_plot_file: Path to save the plot.
    """
    n_classes = len(class_names)

    y_true_binarized = label_binarize(y_true_encoded, classes=np.arange(n_classes))
    
    # Handle the case where y_true_binarized might become 1D if only one class is present
    # or if y_true_encoded only contains a single unique value.
    if y_true_binarized.ndim == 1 and n_classes > 1:
        # This can happen if y_true_encoded contains only one class value.
        # We need to reshape it to (n_samples, n_classes).
        temp_binarized = np.zeros((len(y_true_encoded), n_classes))
        unique_labels = np.unique(y_true_encoded)
        if len(unique_labels) == 1:
            class_idx = unique_labels[0]
            if 0 <= class_idx < n_classes:
                temp_binarized[:, class_idx] = 1
                y_true_binarized = temp_binarized
            else:
                print(f"Error: The single unique class index {class_idx} is out of bounds for n_classes {n_classes}.")
                return
        else: # Should not happen if label_binarize was used correctly with classes=np.arange(n_classes)
            print("Warning: y_true_binarized is 1D despite multiple classes. Check label encoding.")
            # Attempt to reshape assuming it's a collapsed version (less likely for ROC context)
            if len(y_true_encoded) == n_classes and y_true_binarized.shape[0] == len(y_true_encoded): # Unlikely
                print("Cannot safely reshape y_true_binarized.")
                return
            # If still problematic, manual inspection of y_true_encoded and n_classes is needed.


    if y_true_binarized.shape[1] != n_classes and n_classes == 1 : # Special case for binary effectively
         y_true_binarized = y_true_binarized.reshape(-1,1) # Ensure it's 2D

    if y_score.shape[1] != n_classes:
        print(f"Critical Error: Mismatch in number of classes between scores ({y_score.shape[1]}) and class_names ({n_classes}).")
        print("This likely indicates an issue with model output dimensions or class name list.")
        return


    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        if np.sum(y_true_binarized[:, i]) > 0 and np.sum(y_true_binarized[:, i]) < len(y_true_binarized):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], roc_auc[i] = None, None, float('nan')
            print(f"Warning: ROC AUC not defined for class '{class_names[i]}' (index {i}) due to insufficient class representation.")

    # Micro-average
    if np.any(np.sum(y_true_binarized, axis=0) > 0) and np.any(np.sum(y_true_binarized, axis=0) < y_true_binarized.shape[0]):
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    else:
        fpr["micro"], tpr["micro"], roc_auc["micro"] = None, None, float('nan')

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if fpr[i] is not None]))
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes_for_macro = 0
    for i in range(n_classes):
        if fpr[i] is not None and tpr[i] is not None:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            valid_classes_for_macro +=1
    
    if valid_classes_for_macro > 0:
        mean_tpr /= valid_classes_for_macro
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        fpr["macro"], tpr["macro"], roc_auc["macro"] = None, None, float('nan')

    plt.figure(figsize=(12, 10)) # Increased figure size for better legend readability
    if fpr["micro"] is not None:
        plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})", color="deeppink", linestyle=":", linewidth=4)
    if fpr["macro"] is not None:
        plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-average ROC (AUC = {roc_auc['macro']:.2f})", color="navy", linestyle=":", linewidth=4)

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"] +
                   ["#FFC300", "#C70039", "#900C3F", "#581845"]) # Added more colors
    for i, color in zip(range(n_classes), colors):
        if fpr[i] is not None:
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"ROC of {class_names[i]} (AUC = {roc_auc[i]:.2f})")
        elif not np.isnan(roc_auc[i]):
             plt.plot([], [], color=color, lw=2, label=f"ROC of {class_names[i]} (AUC = N/A)")


    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Chance level (AUC = 0.5)")
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Multi-class Receiver Operating Characteristic (Subfamily)", fontsize=16)
    plt.legend(loc="lower right", fontsize='medium') # Adjusted font size
    plt.grid(True)
    
    try:
        plt.savefig(output_plot_file, bbox_inches='tight') # Added bbox_inches for tight layout
        print(f"Multi-class ROC plot saved to: {output_plot_file}")
    except Exception as e:
        print(f"Error saving ROC plot: {e}")
    plt.close() # Close the plot to free memory

# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this script (e.g., /c:/Users/11944/Desktop/protein_cal/scripts)
    
    # Path to the main CSV data file
    # UPDATE THIS LINE:
    # CSV_DATA_FILE = os.path.join(BASE_DIR, '..', 'data_source', 'YOUR_ACTUAL_DATA_FILENAME.csv')
    # For example, if 'data_new.csv' is the correct file:
    CSV_DATA_FILE = os.path.join(BASE_DIR, '..', 'data_source', 'data_new.csv')
    
    # Path to the pre-trained model (assuming model_results is a sibling of data_source)
    MODEL_FILENAME = 'best_protein_classifier.pth'
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'model_results', MODEL_FILENAME) # Model in ../model_results/

    # Output directory for saved .npy files and the ROC plot
    RESULTS_DIR = os.path.join(BASE_DIR, '..', 'model_results', 'subfamily_roc_data')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    PLOT_OUTPUT_FILE = os.path.join(RESULTS_DIR, 'subfamily_multiclass_roc_curve.png')
    TRUE_LABELS_FILE = os.path.join(RESULTS_DIR, 'subfamily_roc_true_labels.npy')
    PROBABILITIES_FILE = os.path.join(RESULTS_DIR, 'subfamily_roc_probabilities.npy')
    CLASS_NAMES_FILE = os.path.join(RESULTS_DIR, 'subfamily_roc_class_names.npy')

    MAX_DOMAINS = 50 # Should match training
    BATCH_SIZE = 64  # Or any appropriate batch size for evaluation
    
    # --- End Configuration ---

    print(f"Using data file: {CSV_DATA_FILE}")
    print(f"Attempting to load model from: {MODEL_PATH}")
    print(f"Results will be saved in: {RESULTS_DIR}")

    if not os.path.exists(CSV_DATA_FILE):
        print(f"Error: Data CSV file not found at {CSV_DATA_FILE}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    # 1. Load Data and Prepare Test Set
    try:
        main_df = pd.read_csv(CSV_DATA_FILE)
        # Ensure 'protein_id' and 'is_negative_control' columns exist for the split function
        if 'protein_id' not in main_df.columns:
            print("Warning: 'protein_id' column not found. Using index for grouping.")
            main_df['protein_id'] = main_df.index
        if 'is_negative_control' not in main_df.columns:
            print("Warning: 'is_negative_control' column not found. Assuming all are positive controls.")
            main_df['is_negative_control'] = False

    except FileNotFoundError:
        print(f"Error: Main data CSV file not found at {CSV_DATA_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading main_df: {e}")
        sys.exit(1)

    _, _, test_df, label_encoder, _ = custom_split_dataset_with_negatives(
        main_df,
        test_size=0.2, # This should match how your model was originally evaluated
        val_size=0.1,  # Ensure consistency with original split if comparing results
        random_state=42,
        group_col='protein_id'
    )
    
    if test_df.empty:
        print("Error: Test dataset is empty after split. Cannot proceed.")
        sys.exit(1)

    test_dataset = ProteinDataset(df=test_df, label_encoder=label_encoder, max_domains=MAX_DOMAINS, is_predicting=False) # False to get labels
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = test_dataset.input_dim
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_

    print(f"Test set size: {len(test_df)}")
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    # print(f"Class names: {class_names}")


    # 2. Load Pre-trained Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedProteinClassifier(input_dim=input_dim, num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()

    # 3. Perform Inference and Collect Data
    all_true_labels_encoded = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels_encoded_batch in test_loader:
            inputs = inputs.to(device)
            # labels_encoded_batch are already on CPU from ProteinDataset
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_true_labels_encoded.append(labels_encoded_batch.cpu().numpy())
            all_probabilities.append(probs.cpu().numpy())

    if not all_true_labels_encoded or not all_probabilities:
        print("Error: No data collected from test_loader. Check test set and model.")
        sys.exit(1)
        
    true_labels_for_roc = np.concatenate(all_true_labels_encoded)
    probabilities_for_roc = np.concatenate(all_probabilities, axis=0)

    # 4. Save Data
    try:
        np.save(TRUE_LABELS_FILE, true_labels_for_roc)
        np.save(PROBABILITIES_FILE, probabilities_for_roc)
        np.save(CLASS_NAMES_FILE, class_names) # allow_pickle=True is default for object arrays
        print(f"Saved true labels for ROC to {TRUE_LABELS_FILE}")
        print(f"Saved probability scores for ROC to {PROBABILITIES_FILE}")
        print(f"Saved class names for ROC to {CLASS_NAMES_FILE}")
    except Exception as e:
        print(f"Error saving .npy files: {e}")
        sys.exit(1)

    # 5. Generate and Save Plot
    if true_labels_for_roc.size > 0 and probabilities_for_roc.size > 0 and class_names.size > 0:
        print("Generating ROC plot...")
        plot_multiclass_roc_from_data(
            true_labels_for_roc,
            probabilities_for_roc,
            class_names,
            PLOT_OUTPUT_FILE
        )
    else:
        print("ROC data arrays are empty. Skipping plot generation.")
        
    print("ROC analysis script finished.")
