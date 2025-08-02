The pipeline is primarily controlled by two scripts: `run_benchmark.py` for executing the full workflow and `plot.py` for visualizing the results.

### Running the Full Pipeline (`run_benchmark.py`) 
This is the original data splitting strategy. In which the true negative set is generated for each subfamily or family by selecting negative control proteins from *other* superfamilies.

**Usage:**
```bash
# General format
python run_benchmark.py --level [family|subfamily] --model [all|random_forest|svm|neural_network]

# Example: Run all models for subfamily-level classification
python run_benchmark.py --level subfamily --model all

# Example: Run only the Random Forest model for family-level classification
python run_benchmark.py --level family --model random_forest
```

### Running the Full Pipeline with New Data Splitting (`run_benchmark_same_sup.py`) 

This script is the same as `run_benchmark.py`, but uses the new data splitting strategy.The new data splitting strategy is defined in `data_splitting_same_sup.py`. In which the true negative set is generated for each subfamily or family by selecting negative control proteins from *same* superfamilies.

**Usage:**
```bash
python run_benchmark_same_sup.py --level [family|subfamily] --model [all|random_forest|svm|neural_network]
```




Here’s the integrated and cleaned-up version of your content under a unified **README**-style structure. It combines all details into a cohesive **"Basic Usage"** section while clearly explaining the difference between the two scripts (`run_benchmark.py` and `run_benchmark_same_sup.py`).

---


## Advanced Usage


### Generate Sensitivity/Specificity Curves

To regenerate the sensitivity vs. specificity plots for each model without re-running the entire pipeline:

```bash
# Generate plots for all models at subfamily level
python benchmark_scripts/generate_roc_plot.py --level subfamily

# Generate plots for specific models at family level
python benchmark_scripts/generate_roc_plot.py --level family --models neural_network svm
```
