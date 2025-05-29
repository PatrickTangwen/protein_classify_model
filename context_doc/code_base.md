# Protein Classification Scripts Analysis Documentation

## Overview

This documentation describes a suite of Python scripts designed for protein classification tasks using deep learning. The primary scripts are `neural_network_subfamily.py` for subfamily-level classification and `neural_network_family.py` for family-level classification. A utility script, `run.py`, allows users to choose which classification level to execute. The system processes protein domain data to predict protein classifications using a neural network, with detailed evaluation and reporting.

## Execution

The primary way to run the classification is via `run.py`:

-   **Navigate to the `scripts` directory.**
-   **To run subfamily-level classification:**
    ```bash
    python run.py -subfamily
    ```
    This will execute `neural_network_subfamily.py`.
-   **To run family-level classification:**
    ```bash
    python run.py -family
    ```
    This will execute `neural_network_family.py`.

## Core Scripts

### 1. `neural_network_subfamily.py`
   - **Purpose**: Implements subfamily-level protein classification.
   - **Output Directory**: `../model_results/`
   - **Key Report Files**:
     - `../model_results/classification_report.txt` (Detailed per-subfamily breakdown)
     - `../model_results/confidence_threshold_analysis.txt` (Confidence analysis, overall statistics, binary metrics, misclassification summary)
     - `../model_results/detailed_classification_results.csv`
     - `../model_results/binary_classification_metrics.csv`
     - `../model_results/training_log.txt`
     - `../model_results/training_history.png`
     - `../model_results/best_protein_classifier.pth`

### 2. `neural_network_family.py`
   - **Purpose**: Implements family-level protein classification.
   - **Output Directory**: `../model_results_family/`
   - **Key Report Files** (note the `_family` suffix):
     - `../model_results_family/classification_report_family.txt` (Detailed per-family breakdown)
     - `../model_results_family/confidence_threshold_analysis_family.txt` (Confidence analysis, overall statistics, binary metrics, misclassification summary)
     - `../model_results_family/detailed_classification_results_family.csv`
     - `../model_results_family/binary_classification_metrics_family.csv`
     - `../model_results_family/training_log_family.txt`
     - `../model_results_family/training_history_family.png`
     - `../model_results_family/best_protein_classifier_family.pth`

## Data Input and Structure (Common for both classification levels)

### Input Data Source
- **File**: `data_new.csv` from the `../data_source/` directory (relative to the script's location).
- **Superfamily Mapping File**: `../data_source/fam2supefamily.csv` used for superfamily-based analysis (e.g., negative control selection, misclassification analysis in family-level script).
- **Expected Columns in `data_new.csv`**:
  - `Domains`: String representation of lists containing domain information.
  - `Seperators`: String representation of lists containing separator region information.
  - `Subfamily`: Target classification labels (used directly by `neural_network_subfamily.py` and to derive `Family` labels in `neural_network_family.py`).
  - `Length`: Length of the protein sequence (previously `Protein length`).
  - `Accession`: Protein identifier (previously `Protein`).

### Data Format Details
- **Domains**: Each domain contains `[accession, start_pos, end_pos, score]`
  - Accession format examples: `pfamXXXXX`, `CDD:XXXXX`
  - Positions are absolute coordinates in the protein.
  - Scores represent domain confidence/quality.
- **Separators**: Each separator contains `[type, start_pos, end_pos]`
  - Represents regions between domains.
  - Positions are absolute coordinates.

## Core Classes and Functions (Largely shared between `neural_network_subfamily.py` and `neural_network_family.py`)

### 1. `ProteinDataset` Class

**Purpose**: Custom PyTorch Dataset class that processes protein domain data into machine learning features.

**Key Features**:
- Converts string representations of domain and separator lists to actual Python lists using `ast.literal_eval()`.
- Creates comprehensive feature vectors from domain and separator information.
- Implements label encoding:
    - For `Subfamily` in `neural_network_subfamily.py`.
    - For `Family` (derived from `Subfamily`) in `neural_network_family.py`.
- Normalizes features for better model performance.
- Optimized tensor creation by first converting list of NumPy arrays to a single NumPy array.

**Feature Engineering Process**:

#### Domain Features
1. **Domain Presence Vector**: Binary vector indicating which domains are present from a learned vocabulary.
2. **Domain Position Features**: Normalized start/end positions (divided by protein length).
3. **Domain Scores**: Log-transformed domain confidence scores (`np.log1p(score)`).
4. **Domain Order Features**: Sequential order of domains in the protein, up to `max_domains`.
5. **Domain Count**: Normalized count of domains per protein (`count / max_domains`).

#### Separator Features
1. **Position Features**: Normalized start/end positions of separator regions.
2. **Length Features**: Normalized length of separator regions.
3. **Padding**: Fixed-size feature vector (padded/truncated to 60 features, corresponding to max 20 separators × 3 features each).

#### Feature Normalization
- Z-score normalization: `(features - mean) / std`.
- Handles zero standard deviation by setting std to 1 to avoid division by zero errors.
- Applied to the entire feature matrix for consistent scaling.

**Methods**:
- `__init__(df, max_domains=50)`: Initializes dataset with feature processing.
- `__len__()`: Returns dataset size.
- `__getitem__(idx)`: Returns feature-label pairs for training.

### 2. `ImprovedProteinClassifier` Class

**Purpose**: Deep neural network for protein classification.

**Architecture**:
- **Input Layer**: Variable size based on feature dimensions from `ProteinDataset`.
- **Hidden Layers**: Configurable (default: [512, 256, 128]).
- **Activation**: ReLU activation functions.
- **Regularization**:
  - Batch normalization (`nn.BatchNorm1d`) after each hidden layer.
  - Dropout (`nn.Dropout(0.4)`) for preventing overfitting.
- **Output Layer**: Linear layer with number of output units matching the number of classes (subfamilies or families). Softmax is applied during evaluation, not part of the model directly, as `nn.CrossEntropyLoss` combines LogSoftmax and NLLLoss.

**Key Features**:
- **Weight Initialization**: Kaiming normal initialization (`nn.init.kaiming_normal_`) for ReLU networks.
- **Batch Normalization**: Improves training stability and convergence.
- **Dropout**: Reduces overfitting during training.

**Methods**:
- `__init__(input_dim, num_classes, hidden_dims)`: Network architecture setup.
- `_init_weights(module)`: Custom weight initialization for Linear and BatchNorm1d layers.
- `forward(x)`: Forward pass through the network.

### 3. Data Splitting Strategy

**Function**: `custom_split_dataset(df)`

**Purpose**: Implements a data splitting strategy based on class size (subfamily or family).

**Splitting Rules (applied to the target class: Subfamily or Family)**:
1. **Single Member Classes**: Protein appears in both training and test sets.
2. **Two Member Classes**: 1:1 split (one for training, one for testing).
3. **More Than Two Member Classes**: 80:20 train-test split.

**Benefits**:
- Attempts to handle class imbalance.
- Ensures all classes are represented in evaluation where possible.

### 4. Negative Control Generation

**Function**: `generate_negative_controls(df, test_indices, train_indices)`

**Purpose**: For each target class (subfamily/family) in the test set, generate negative control proteins.

**Rules**:
- Negative controls are selected from *other superfamilies*.
- If a target class's family is not assigned to any superfamily, its negatives are selected from families that *are* assigned to superfamilies.
- Negative proteins must not belong to the same target class being tested.
- **Size**: The number of negative controls is `max(n_test_for_class, 5)`.

### 5. Combined Data Splitting with Negatives

**Function**: `custom_split_dataset_with_negatives(df)`

**Purpose**: Integrates `custom_split_dataset` and `generate_negative_controls` to produce final training and validation sets. The validation set includes original test samples and their corresponding negative controls.

**Returns**:
- `train_indices`
- `test_indices_with_negatives`
- `is_negative_control` (dictionary mapping index to boolean)
- `*_test_mapping` (dictionary mapping target class to its positive and negative test indices)

### 6. Evaluation and Analysis Functions

#### `analyze_misclassification_type(true_class, pred_class)`
- **In `neural_network_subfamily.py`**:
    - **Purpose**: Categorizes prediction errors based on whether the predicted subfamily belongs to the same parent family as the true subfamily.
    - **Returns**: 'same_family' or 'different_family'.
- **In `neural_network_family.py`**:
    - **Purpose**: Categorizes prediction errors based on whether the predicted family belongs to the same parent superfamily as the true family.
    - **Returns**: 'same_superfamily' or 'different_superfamily'.

#### `evaluate_model_detailed(...)`

**Purpose**: Comprehensive model evaluation with detailed per-class (subfamily/family) analysis and binary classification metrics (TP, FP, TN, FN) considering negative controls.

**Evaluation Metrics (per class)**:
  - Total size of the class.
  - Training sample count.
  - Test sample count (excluding negative controls for this specific class).
  - Negative control count for this class's evaluation.
  - Correct predictions (on original test samples).
  - Accuracy (on original test samples).
  - True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) for binary view.
  - Precision, Recall, Specificity, F1-Score for binary view.
  - Detailed misclassification analysis (listing misclassified proteins, their predicted class, confidence, and error type like same_family/different_family or same_superfamily/different_superfamily).

**Output**:
- A detailed report dictionary (e.g., `subfamily_report` or `family_report`).
- A `results_df` Pandas DataFrame with all predictions, true labels, and confidences for samples in the validation loader.

## Training Pipeline (Common for both scripts)

### 1. Data Preparation
- Load data using Pandas.
- Create `Family` column if `neural_network_family.py` is run.
- Instantiate `ProteinDataset`.
- Use `custom_split_dataset_with_negatives` to get train/validation indices and mappings.
- Create `torch.utils.data.Subset` for train and validation sets.
- Create `DataLoader` for train and validation.

### 2. Class Imbalance Handling
- **Weighted Loss Function**: `nn.CrossEntropyLoss` with `weight` parameter.
- **Class Weights Calculation**: Inverse frequency weighting for classes in the entire dataset, then normalized.

### 3. Model Configuration
- **Optimizer**: AdamW (`optim.AdamW`) with weight decay (0.01).
- **Learning Rate**: Initial 0.001 with `ReduceLROnPlateau` scheduler (monitors validation accuracy, factor 0.5, patience 5).
- **Loss Function**: Weighted `nn.CrossEntropyLoss`.
- **Additional Regularization**: L2 regularization manually added to the loss (lambda=0.001).

### 4. Training Loop Features
- **Early Stopping**: Patience of 15 epochs based on validation accuracy.
- **Model Checkpointing**: Saves the state dictionary of the best model (based on validation accuracy) in memory; one final save to disk at the end of training.
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_` with max norm of 1.0.
- **Progress Tracking**: Per-epoch training/validation loss, accuracy, and duration.

### 5. Logging
- A custom `Logger` class redirects `sys.stdout` to both the console and a log file (e.g., `training_log.txt`).
- It also handles switching output to a separate classification report file (e.g., `classification_report.txt`) when the detailed report section begins, to keep the terminal less cluttered during this verbose output.

## Evaluation and Reporting

### 1. Training Visualization
- **Loss Curves**: Training and validation loss over epochs.
- **Accuracy Curves**: Training and validation accuracy over epochs.
- Plots saved as PNG files (e.g., `training_history.png`).

### 2. Detailed Classification Report (e.g., `classification_report.txt` or `classification_report_family.txt`)

#### Per-Class Analysis (Subfamily or Family)
For each class, the system reports:
- **Total Size**: Number of members in the original dataset.
- **Data Split Information**: How the class was divided for training/testing based on its size.
- **Training Set Statistics**: Number of training proteins from this class and their accessions/labels.
- **Testing Set Statistics**:
    - Number of original test proteins from this class and their accessions/labels.
    - Number of negative controls used for this class's binary evaluation and their accessions/labels.
    - Correct predictions and accuracy on original test proteins.
- **Binary Classification Metrics (with negative controls)**: TP, FP, TN, FN, Precision, Recall, Specificity, F1-Score.
- **Misclassification Analysis**:
    - Total misclassifications for original test proteins.
    - Counts of same-level (family/superfamily) vs. different-level errors.
    - Details for each misclassified protein: Accession, Predicted Class, Confidence, Error Type.

### 3. Confidence Threshold Analysis File (e.g., `confidence_threshold_analysis.txt`)
This file consolidates:
- **Confidence Threshold Table**:
    - Shows how many test proteins (original + negatives) are retained at various confidence thresholds (0.0 to 0.9).
    - Reports the percentage of the total test set retained.
    - Reports family-level accuracy and subfamily-level accuracy (for `neural_network_subfamily.py`) or just family-level accuracy (for `neural_network_family.py`) calculated *only* on the retained proteins at each threshold.
    - Includes detailed column definitions.
- **Overall Classification Statistics**:
    - Total test proteins (original).
    - Total correct predictions (original).
    - Overall accuracy (original).
- **Overall Binary Classification Metrics with Negative Controls**:
    - Aggregated TP, FP, TN, FN over all classes.
    - Overall Precision, Recall, Specificity, F1-Score, Accuracy for the binary setup.
- **Overall Misclassification Statistics**:
    - Total misclassifications.
    - Percentage of same-level (family/superfamily) vs. different-level errors.

### 4. CSV Output Files
- **Detailed Classification Results (e.g., `detailed_classification_results.csv`)**:
    - Protein Accession, True Class, Predicted Class, Confidence, Original Index for every sample in the validation set.
- **Binary Classification Metrics (e.g., `binary_classification_metrics.csv`)**:
    - Per-class TP, FP, TN, FN, Precision, Recall, Specificity, F1-Score, Test Samples, Negative Controls.

## Key Strengths of the Implementation

### 1. Robust Feature Engineering
- Comprehensive feature representation (presence, position, scores, order, count, separators).
- Normalization and scaling.
- Handling of variable-length domain sequences through padding/truncation and vocabulary.

### 2. Sophisticated Data Handling
- Custom splitting strategy for imbalanced data.
- Generation and use of negative controls for more robust binary evaluation per class.
- Hierarchical error analysis.

### 3. Advanced Training Techniques
- Class-weighted loss.
- Multiple regularization methods (Dropout, L2, Batch Normalization, Weight Decay in AdamW).
- Early stopping and best model saving.
- Learning rate scheduling.

### 4. Comprehensive and Segregated Reporting
- Detailed per-class reports.
- Separate, focused confidence analysis with aggregated statistics.
- Multiple CSV outputs for easy data export and further analysis.
- Robust logging for traceability.

## Potential Areas for Improvement

### 1. Feature Engineering
- Incorporate sequence-based features (e.g., embeddings from pre-trained protein language models).
- Explore domain-domain interaction features.
- Utilize evolutionary conservation scores.

### 2. Model Architecture
- Experiment with attention mechanisms to weigh important domains/features.
- Try ensemble methods.
- Investigate graph neural networks if domain interactions are to be modeled explicitly.

### 3. Evaluation Metrics
- Implement macro/micro averaging for metrics like F1-score across all classes.
- Add ROC/AUC curves, especially for the binary per-class evaluations.
- Employ cross-validation for more robust performance estimation.

## Dependencies
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib/Seaborn
- Tabulate

This suite of scripts provides a flexible and thorough framework for protein classification, adaptable for both subfamily and family-level tasks with detailed performance insights. 