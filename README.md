# Protein Classification Pipeline

A comprehensive AutoML pipeline for protein classification using multiple machine learning models with rigorous evaluation methodology.

## Overview

This pipeline provides an automated system for training and benchmarking multiple machine learning models on protein classification tasks. It supports both **subfamily-level** and **family-level** classification with a sophisticated evaluation system that includes negative controls and one-vs-all binary classification analysis.

### Key Features

- **Multiple Models**: Supports Neural Networks, Random Forest, SVM, Logistic Regression, Extra Trees, KNN, Naive Bayes, and XGBoost
- **Evaluation**: One-vs-all binary classification approach with negative controls from other superfamilies
- **Automated Benchmarking**: Compare all models on identical data splits

## Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn tabulate joblib
# Optional: XGBoost for additional model
pip install xgboost
```

### Data Requirements

Ensure you have the following files in the `data_source/` directory:
- `data_new.csv`: Protein data with domains and separators
- `fam2supefamily.csv`: Family to superfamily mapping

### Basic Usage

#### Run All Models for Subfamily Classification

```bash
python benchmark_scripts/run_benchmark.py --level subfamily --model all
```

#### Run All Models for Family Classification

```bash
python benchmark_scripts/run_benchmark.py --level family --model all
```

#### Run a Specific Model

```bash
# Run only Random Forest for subfamily classification
python benchmark_scripts/run_benchmark.py --level subfamily --model random_forest

# Run only Neural Network for family classification
python benchmark_scripts/run_benchmark.py --level family --model neural_network
```

#### Available Models

- `neural_network`: Deep Neural Network with batch normalization and dropout
- `random_forest`: Random Forest Classifier
- `svm`: Support Vector Machine with RBF kernel
- `logistic_regression`: Logistic Regression
- `extra_trees`: Extra Trees Classifier
- `knn`: K-Nearest Neighbors
- `naive_bayes`: Gaussian Naive Bayes
- `xgboost`: XGBoost Classifier (if installed)

## Output Structure

Results are saved in the `benchmark_results/` directory:

```
benchmark_results/
├── family/                          # Family-level classification results
│   ├── neural_network/
│   │   ├── classification_report_family.txt      # Detailed per-class analysis
│   │   ├── classification_stats_family.txt       # Statistical analysis with thresholds
│   │   ├── detailed_classification_results_family.csv  # Per-protein predictions
│   │   ├── binary_classification_metrics_family.csv    # Per-class binary metrics
│   │   ├── summary_metrics.json                  # Overall performance metrics
│   │   ├── sensitivity_specificity_curve.png     # Threshold analysis plot
│   │   ├── roc_curve_traditional.png            # Traditional ROC curve
│   │   ├── model.pth (PyTorch) / model.joblib (sklearn)  # Trained model
│   │   ├── training_history.png                  # Training curves (Neural Network)
│   │   └── training_log.txt                      # Training log (Neural Network)
│   ├── random_forest/
│   └── ... (other models)
├── subfamily/                       # Subfamily-level classification results
│   └── ... (same structure as family)
└── plots/
    ├── family_comparison.png        # Model comparison for family-level
    └── subfamily_comparison.png     # Model comparison for subfamily-level
```

## Understanding the Evaluation

### One-vs-All Binary Classification

The pipeline uses a sophisticated evaluation approach:

1. **For each subfamily/family**: Creates a separate binary classification problem
2. **Positive samples**: Test proteins that belong to the target class
3. **Negative samples**: Carefully selected proteins from other superfamilies
4. **Metrics**: Calculated using the formula `(TP + TN) / (TP + TN + FP + FN)`

### Key Metrics Explained

- **TP (True Positive)**: Correctly predicted as target class
- **TN (True Negative)**: Correctly rejected negative controls
- **FP (False Positive)**: Incorrectly accepted negative controls
- **FN (False Negative)**: Incorrectly predicted as different class

### Confidence Threshold Analysis

The system analyzes performance at different confidence thresholds (0.0 to 0.9), showing:
- How many original test proteins are retained at each threshold
- One-vs-all accuracy including negative controls
- Optimal operating points for the classifier

## Data Splitting Strategy

The pipeline uses a specialized splitting strategy based on class size:

- **1 member**: Used for both training and testing
- **2 members**: Split 1:1 for training/testing  
- **>2 members**: Split 80:20 for training/testing

### Negative Controls

For each test class, negative controls are selected:
- **Source**: Other superfamilies (never same subfamily/family)
- **Size**: `max(test_set_size, 5)` proteins
- **Special case**: Classes without superfamily assignment use only proteins with superfamily assignments as negatives

## Advanced Usage

### Generate Benchmark Comparison Plots

If you've already run the models and want to regenerate the main comparison bar plots:

```bash
python benchmark_scripts/generate_benchmark_plot.py --level subfamily
python benchmark_scripts/generate_benchmark_plot.py --level family
```

### Custom Output Directory

```bash
python benchmark_scripts/generate_benchmark_plot.py --level subfamily --output /path/to/custom/output
```

### Generate Sensitivity/Specificity Curves

To regenerate the sensitivity vs. specificity plots for each model without re-running the entire pipeline:

```bash
# Generate plots for all models at subfamily level
python benchmark_scripts/generate_roc_plot.py --level subfamily

# Generate plots for specific models at family level
python benchmark_scripts/generate_roc_plot.py --level family --models neural_network svm
```


## Configuration

Key parameters can be modified in `benchmark_scripts/config.py`:

```python
# Training configuration
PYTORCH_EPOCHS = 100          # Maximum epochs for neural network
PYTORCH_PATIENCE = 15         # Early stopping patience
PYTORCH_BATCH_SIZE = 32       # Batch size for training
PYTORCH_LR = 0.001           # Learning rate

# Feature configuration  
MAX_DOMAINS = 50             # Maximum domains for order features
```

## Output Files Explained

### classification_report.txt
Detailed per-class breakdown including:
- Training/test set composition
- Binary classification metrics (TP/TN/FP/FN)
- Misclassification analysis
- Negative control details

### classification_stats.txt
Statistical analysis with four sections:
1. **Confidence Threshold Analysis**: Performance at different confidence levels
2. **Overall Statistics**: Aggregated performance including negative controls
3. **Binary Classification Metrics**: Precision, recall, specificity, F1-score
4. **Misclassification Statistics**: Error pattern analysis

### detailed_classification_results.csv
Per-protein prediction results with:
- Protein accession
- True and predicted labels
- Confidence scores
- Negative control flags

### binary_classification_metrics.csv
Per-class binary metrics in tabular format for easy analysis

### Visualization Files
- **sensitivity_specificity_curve.png**: Shows optimal threshold selection by plotting sensitivity and specificity against probability cutoffs.
- **roc_curve_traditional.png**: Traditional ROC curve analysis (TPR vs FPR).
- **training_history.png**: Loss/accuracy curves (Neural Network only)

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure `data_new.csv` and `fam2supefamily.csv` are in `data_source/`
2. **XGBoost not found**: Install with `pip install xgboost` or remove from models list
3. **GPU not detected**: PyTorch will automatically fall back to CPU
4. **Memory issues**: Reduce batch size in `config.py` for large datasets

### Performance Tips

- Use GPU for neural network training when available
- Consider reducing the number of models for faster iteration
- Monitor memory usage with large datasets

## Contributing

To add a new model:

1. Add model configuration to `MODELS` dictionary in `benchmark_scripts/models.py`
2. Specify model type ('sklearn' or 'pytorch')
3. The pipeline will automatically handle training and evaluation

## Citation

If you use this pipeline in your research, please cite the original protein classification methodology and this implementation.
