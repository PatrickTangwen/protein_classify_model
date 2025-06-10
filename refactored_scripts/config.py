import os

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SOURCE_DIR = os.path.join(BASE_DIR, 'data_source')
RESULTS_DIR = os.path.join(BASE_DIR, 'model_results_automl')

# Input data files
PROTEIN_DATA_PATH = os.path.join(DATA_SOURCE_DIR, 'data_new.csv')
SUPERFAMILY_MAP_PATH = os.path.join(DATA_SOURCE_DIR, 'fam2supefamily.csv')

# --- Model & Feature Configuration ---
MAX_DOMAINS = 50

# --- Training Configuration ---
# (Could be expanded for PyTorch specific settings)
PYTORCH_EPOCHS = 100
PYTORCH_PATIENCE = 15
PYTORCH_BATCH_SIZE = 32
PYTORCH_LR = 0.001 