# AutoML Protein Classification Pipeline: Code Base Documentation

## 1. Overview

This document provides a comprehensive analysis of the benchmark of protein classification pipeline. This pipeline is a modular, refactored version of the original protein classification scripts, designed to facilitate the training, evaluation, and benchmarking of multiple machine learning models (Random Forest, SVM, Neural Network and other models) on a consistent basis.

The core objectives of this refactored pipeline are:
- **Modularity**: Code is broken down into logical, reusable scripts for data handling, feature engineering, model training, and evaluation.
- **Extensibility**: New models can be easily added to the benchmarking suite.
- **Consistency**: All models are trained and evaluated on the exact same data splits, including identical true negative control sets, ensuring fair comparisons.
- **Automation**: A single command can orchestrate the entire process of training and evaluating all specified models for a given classification level.
- **Flexible Benchmarking**: A standalone script allows for the regeneration and customization of performance comparison plots without re-running the entire pipeline.

## 2. Execution

The pipeline is primarily controlled by two scripts: `run_benchmark.py` for executing the full workflow and `plot.py` for visualizing the results.

### Running the Full Pipeline (`run_benchmark.py`)

This is the main orchestrator script.

**Usage:**
```bash
# General format
python run_benchmark.py --level [family|subfamily] --model [all|random_forest|svm|neural_network]

# Example: Run all models for subfamily-level classification
python run_benchmark.py --level subfamily --model all

# Example: Run only the Random Forest model for family-level classification
python run_benchmark.py --level family --model random_forest
```

### Generating Benchmark Plots Separately (`plot.py`)

This script allows for on-demand generation and customization of the model comparison plots. It reads the `summary_metrics.json` file from each model's output directory.

**Usage:**
```bash
# General format
python plot.py --level [family|subfamily] [options]

# Example: Generate a standard plot for subfamily results
python plot.py --level subfamily
```
See the script's `argparse` section for a full list of customization options.

## 3. Project Structure

### Script Directory (`refactored_scripts/`)
The core logic is organized into the following modules:
- **`config.py`**: Central configuration for paths, model parameters, and global constants.
- **`data_loader.py`**: Handles loading and initial preparation of protein data and superfamily maps.
- **`feature_engineering.py`**: Contains model-agnostic logic for creating the feature matrix `X` and label vector `y`.
- **`data_splitting.py`**: Implements the project's specific data splitting and true negative control generation rules.
- **`models.py`**: Defines all models to be benchmarked (Neural Network, Random Forest, SVM).
- **`training.py`**: Provides a generic training framework that can handle both PyTorch and Scikit-learn models.
- **`evaluation.py`**: Contains generalized functions for detailed evaluation and generating all report files in the original format.
- **`plot.py`**: The script for generating comparison plots

### Output Directory (`benchmark/`)

All outputs from this pipeline are saved to a new, structured directory to avoid conflicts with the original scripts.

```
model_results_automl/
├── family/
│   ├── neural_network/
│   │   ├── classification_report_family.txt
│   │   ├── classification_stats_family.txt
│   │   ├── detailed_classification_results_family.csv
│   │   ├── binary_classification_metrics_family.csv
│   │   ├── summary_metrics.json
│   │   ├── model.pth
│   │   ├── training_history.png
│   │   └── training_log.txt
│   ├── random_forest/
│   │   ├── classification_report_family.txt
│   │   └── ... (all other reports)
│   └── svm/
│       └── ...
├── subfamily/
│   ├── neural_network/
│   │   ├── classification_report.txt
│   │   └── ...
│   └── ...
└── benchmark/
    ├── family_comparison.png
    └── subfamily_comparison.png
```

## 4. Pipeline Workflow

The `run_benchmark.py` script executes the following steps in sequence:

1.  **Load Data**: Uses `data_loader.py` to load `data_new.csv` and `fam2supefamily.csv`.
2.  **Build Features**: Uses `feature_engineering.py` to convert the raw data into a numerical feature matrix `X` and encoded labels `y`.
3.  **Split Data**: Uses `data_splitting.py` to create a single, definitive train/validation split, including negative controls. This split is used for all models.
4.  **Train & Evaluate Models**: Loops through the selected models:
    - Calls `training.py` to train the model.
    - Calls `evaluation.py` to get predictions, perform detailed analysis, and save all standardized report files.

## 5. Core Modules and Logic

### `data_loader.py`
- **`load_protein_data()`**: Reads `data_new.csv`. It uses `ast.literal_eval` to parse the string-formatted `Domains` and `Seperators` columns. When `level='family'`, it derives the `Family` column from the `Subfamily` column.
- **`load_superfamily_map()`**: Reads `fam2supefamily.csv` and returns a clean dictionary mapping family IDs to superfamily labels.

### `feature_engineering.py`
- **`build_features()`**: This function is now fully decoupled from any specific ML framework. It generates a comprehensive feature vector for each protein, including:
    - **Domain Features**: Presence (one-hot), normalized positions, log-transformed scores, sequential order, and normalized count.
    - **Separator Features**: Normalized positions and lengths, padded to a fixed size.
- It returns a tuple containing the scaled feature matrix (`X`), encoded labels (`y`), the `LabelEncoder` instance, the domain vocabulary, and the scaling statistics (mean/std).

### `data_splitting.py`
This module precisely implements the data splitting and negative control logic from the original scripts.

- **`custom_split_dataset()`**: Splits the data based on the number of members in each class (subfamily or family):
    - **1 member**: Used for both training and testing.
    - **2 members**: Split 1:1 into training and testing.
    - **>2 members**: Split 80:20 into training and testing.

- **`generate_negative_controls()`**: For each class in the test set, it generates a set of true negative controls based on the following rules:
    - **Source**: Negative controls are selected from *other superfamilies*.
    - **Exclusion**: Proteins from the same class being tested are never selected as negatives.
    - **Size**: The number of negative controls is `max(n_test_for_class, 5)`.
    - **Special Case**: If a target class has no superfamily assignment, its negatives are drawn only from classes that *do* have a superfamily assignment.

- **`custom_split_dataset_with_negatives()`**: Orchestrates the splitting process by first calling `custom_split_dataset()` and then `generate_negative_controls()` to produce the final `train_indices` and `validation_indices` (which includes positives and their corresponding negatives).

### `models.py`
- **`MODELS` Dictionary**: This dictionary serves as the central model registry for the pipeline. It contains configuration for each model, including the model object (or class) and its type ('sklearn' or 'pytorch').
- **`ImprovedProteinClassifier`**: The original PyTorch neural network architecture. It features:
    - **Hidden Layers**: [512, 256, 128]
    - **Activation**: ReLU
    - **Regularization**: Batch Normalization and Dropout (0.4)
    - **Weight Initialization**: Kaiming Normal

### `training.py`
This module provides a generic training interface.
- **`train_sklearn_model()`**: Handles training for Scikit-learn models using the `.fit()` method and saves the final model using `joblib`.
- **`train_pytorch_model()`**: Encapsulates the original PyTorch training loop, including:
    - **Optimizer**: AdamW with weight decay.
    - **Scheduler**: `ReduceLROnPlateau` on validation accuracy.
    - **Early Stopping**: Halts training if validation accuracy doesn't improve for a set number of epochs.
    - **Artifacts**: Saves the best model state (`model.pth`), a log of the training process (`training_log.txt`), and a plot of loss/accuracy curves (`training_history.png`).

### `evaluation.py`
This module is responsible for generating all analysis and report files in the exact format of the original scripts.
- **`evaluate_model_detailed()`**: The main evaluation engine. It takes predictions and calculates a comprehensive set of metrics for each class.
- **`generate_verbose_report_text()`**: Produces the content for `classification_report.txt`, a human-readable file with a detailed per-class breakdown.
- **`generate_classification_stats()`**: Produces the content for `classification_stats.txt`, which contains four `tabulate`-formatted sections:
    1.  Confidence Threshold Analysis
    2.  Overall Classification Statistics (Original Test Set)
    3.  Overall Binary Classification Metrics (incl. Negative Controls)
    4.  Overall Misclassification Statistics (Original Test Set)
- **`save_reports()`**: Orchestrates the saving of all report files with the correct, level-dependent names (e.g., `_family` suffix). The files generated are:
    - `classification_report.txt`
    - `classification_stats.txt`
    - `detailed_classification_results.csv`
    - `binary_classification_metrics.csv`
    - `summary_metrics.json` (used by the benchmark plotter)
- **`generate_roc_curve()`**: Creates and saves the ROC curve plot.

### `plot.py`
A standalone utility for creating customizable model comparison plots. It reads the `summary_metrics.json` files from the output directories and generates a bar chart comparing the "Overall Accuracy (Original Test Set)" for each model. It includes numerous command-line flags for customizing the plot's appearance.

## 6. Dependencies
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Tabulate
- Joblib 