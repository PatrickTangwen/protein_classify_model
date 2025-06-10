# AutoML Integration and Benchmarking Workflow

This document outlines the plan to integrate Automated Machine Learning (AutoML) capabilities into the protein classification project. The goal is to create a modular, extensible framework for training, evaluating, and comparing multiple machine learning models.

## 1. Objective

The primary objectives of this task are:
1.  **Modularize the Codebase**: Refactor the existing scripts (`neural_network_family.py`, `neural_network_subfamily.py`) into reusable modules for data processing, splitting, training, and evaluation.
2.  **Integrate Conventional ML Models**: Incorporate standard, non-deep-learning models (e.g., Random Forest, SVM) into the pipeline to benchmark against the current neural network.
3.  **Ensure Consistent Evaluation**: Use the exact same data splits and true negative control sets for every model to ensure a fair and consistent comparison.
4.  **Automate Execution**: Enable the training and evaluation of all models for a specific classification level (family or subfamily) through a single command.
5.  **Centralize Benchmarking**: Generate a final performance comparison plot that visualizes the accuracy of all evaluated models, providing a clear benchmark.

## 2. Proposed Project Structure

To achieve modularity, we will refactor the logic into a new set of scripts inside the `refactored_scripts/` directory. The output for this new pipeline will be stored in a new `model_results_automl/` directory, with subdirectories for each model.

### New Script Structure (`refactored_scripts/`)

refactored_scripts/
├── __init__.py
├── config.py                 # Central configuration for paths, models, and parameters.
├── data_loader.py            # Functions for loading and initial preparation of data.
├── feature_engineering.py    # Feature creation logic, decoupled from PyTorch.
├── data_splitting.py         # Custom data splitting and negative control generation.
├── models.py                 # Definition of all models to be trained (NN, RF, SVM, etc.).
├── training.py               # A generic trainer to handle both PyTorch and Scikit-learn models.
├── evaluation.py             # Generalized evaluation, reporting, and ROC plotting functions.
└── benchmark.py              # Script to generate the final model comparison plot.


### New Output Directory Structure

model_results_automl/
├── family/
│   ├── neural_network/
│   │   ├── classification_report.txt
│   │   ├── detailed_classification_results.csv
│   │   └── roc_curve.png
│   │   └── ...
│   ├── random_forest/
│   │   └── ...
│   └── svm/
│       └── ...
├── subfamily/
│   ├── neural_network/
│   │   └── ...
│   └── ...
└── benchmark/
    ├── family_comparison.png
    └── subfamily_comparison.png

## 3. Implementation Plan (Workflow Stages)

The implementation will proceed in the following stages:

### Stage 1: Foundational Code Refactoring

The first step is to deconstruct the existing scripts into modular, reusable functions.

1.  **`data_loader.py`**:
    *   Create a function `load_protein_data(path)` to read `data_new.csv`.
    *   Create a function `load_superfamily_map(path)` to read `fam2supefamily.csv`.
    *   This module will handle the initial loading and basic preprocessing (like adding the 'Family' column).

2.  **`feature_engineering.py`**:
    *   Create a function `build_features(df, max_domains)`.
    *   This function will encapsulate the entire feature creation logic currently inside the `ProteinDataset` class.
    *   **Crucially, it will return model-agnostic outputs**: a NumPy array `X` (feature matrix), a NumPy array `y` (encoded labels), the `LabelEncoder` instance, and the feature vocabulary. This decouples feature generation from PyTorch, making it usable for any model.

3.  **`data_splitting.py`**:
    *   Move the existing `custom_split_dataset`, `generate_negative_controls`, and `custom_split_dataset_with_negatives` functions into this module.
    *   These functions will operate on the DataFrame and return indices and mappings, ensuring they remain independent of any specific model or feature set.

### Stage 2: Model Definition and Training

Next, we will define the models and create a generic training pipeline.

1.  **`models.py`**:
    *   Define a dictionary or list of all models to be benchmarked.
    *   This will include Scikit-learn models and the existing PyTorch model.
    *   Example:
        ```python
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        # from our existing code
        from scripts.neural_network_family import ImprovedProteinClassifier

        MODELS = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'neural_network': ImprovedProteinClassifier, # The class, not an instance
        }
        ```

2.  **`training.py`**:
    *   Implement a generic `train_and_evaluate_model` function.
    *   This function will accept a model object, training data, validation data, and output directory.
    *   It will feature a conditional block to handle the different training paradigms:
        *   **For Scikit-learn models**: Use the `.fit()` and `.predict_proba()` API.
        *   **For the PyTorch model**: Encapsulate the existing training loop (including optimizer, scheduler, early stopping, etc.).
    *   After training, it will call evaluation functions and save the trained model artifact and all reports to the specified output directory.

### Stage 3: Orchestration with `run.py`

The existing `run.py` will be modified to orchestrate the new, modular pipeline.

*   **Command Line Arguments**: Enhance `run.py` to accept arguments for classification level and model selection.
    *   `python run.py --level [subfamily|family]` (Replaces `-subfamily` and `-family`)
    *   `python run.py --model [all|random_forest|svm|neural_network]` (To run one or all models)
*   **Execution Flow**:
    1.  Parse command-line arguments.
    2.  Load data and create features using the new modules (`data_loader`, `feature_engineering`).
    3.  Generate the single, definitive train/test split using `data_splitting`. This split will be reused for all models.
    4.  Identify which models to run based on the `--model` argument.
    5.  Loop through the selected models. For each model:
        a. Create the model-specific output directory (e.g., `model_results_automl/family/random_forest/`).
        b. Call `training.train_and_evaluate_model` with the model, data splits, and output path.

### Stage 4: Benchmarking

The final step is to compare the results from all models.

1.  **`evaluation.py`**:
    *   Generalize the `evaluate_model_detailed` function to work with predictions from any model.
    *   Generalize the ROC curve plotting logic from `raw_plot_roc_curve.py` into a function that can be called for each model after evaluation.

2.  **`benchmark.py`**:
    *   Create a function `generate_benchmark_plot(results_base_dir, level)`.
    *   This function will:
        a. Scan the subdirectories within the results path (e.g., `model_results_automl/family/`).
        b. For each model directory, read the `classification_stats_...txt` file to extract the "Overall Accuracy (Original Test Set)" metric.
        c. Generate a bar chart comparing the accuracy of all models.
        d. Save the plot to the `model_results_automl/benchmark/` directory.
    *   The `run.py` script can automatically call this function after all models have been evaluated.

### Stage 5: Aligning Report Generation

**Objective**: Modify the AutoML pipeline's output to exactly match the format, structure, and naming conventions of the reports generated by the original `neural_network_family.py` and `neural_network_subfamily.py` scripts.

1.  **Replicate `classification_report.txt`**:
    *   **Task**: Create a new function in `evaluation.py` named `generate_verbose_report_text`.
    *   **Details**: This function will take the detailed evaluation results and produce a long-form, human-readable text file. It will iterate through each class (family or subfamily) and print a comprehensive breakdown, including:
        *   Total size and data split information.
        *   Lists of training and testing protein accessions.
        *   Lists of negative controls used for that class.
        *   Accuracy on the original test set.
        *   Binary classification metrics (TP, FP, TN, FN, Precision, etc.).
        *   A detailed list of every misclassified protein and its error type.
    *   This will replicate the main, verbose output that was previously sent to `classification_report.txt`.

2.  **Replicate `classification_stats.txt`**:
    *   **Task**: Enhance the `save_reports` function in `evaluation.py`.
    *   **Details**: The function will be updated to construct a text file that precisely matches the original `classification_stats_...txt` format. This includes four distinct sections with `tabulate`-formatted tables:
        *   Confidence Threshold Analysis.
        *   Overall Classification Statistics (Original Test Set).
        *   Overall Binary Classification Metrics with Negative Controls.
        *   Overall Misclassification Statistics (Original Test Set).
    *   The `summary_metrics.json` file will be kept, as it's vital for the automated benchmarking plot.

3.  **Replicate PyTorch-Specific Outputs**:
    *   **Task**: Update the `train_pytorch_model` function in `training.py`.
    *   **Details**:
        *   **Training History Plot**: The function will be modified to capture the training/validation loss and accuracy at each epoch. After training, it will use `matplotlib` to generate and save a `training_history.png` file, identical to the original.
        *   **Training Log**: A simple logger will be added to the PyTorch training loop to capture epoch-by-epoch progress (loss, accuracy, time) and save it to `training_log.txt`. This will be specific to the neural network model.

4.  **Standardize File Naming**:
    *   **Task**: Adjust file-saving logic in `evaluation.py` and `training.py`.
    *   **Details**: All output files will be saved with standardized names (e.g., `classification_report.txt`, `training_log.txt`) directly inside their model-specific output directory (e.g., `model_results_automl/family/neural_network/`). This maintains the original file names while using the new directory structure for organization.

By implementing these changes, the new pipeline will produce a set of reports for each model that is directly and easily comparable to the output of the original scripts.

