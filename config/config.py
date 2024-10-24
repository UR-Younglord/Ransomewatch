import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set the base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for dataset and reports
DATASET_DIR = os.path.join(BASE_DIR, '../dataset')
REPORTS_DIR = os.path.join(DATASET_DIR, 'reports')
RANSOMWARE_DATASET_DIR = os.path.join(DATASET_DIR, 'ransomware_dataset')

# Load model parameters from a JSON file
def load_model_params(filepath):
    """Load model parameters from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading model parameters: {e}")
        return {}

MODEL_PARAMS = load_model_params(os.path.join(BASE_DIR, 'model_params.json'))

# Validate directory paths
def validate_paths():
    """Validate the existence of critical directory paths."""
    paths = [DATASET_DIR, REPORTS_DIR, RANSOMWARE_DATASET_DIR]
    for path in paths:
        if not os.path.exists(path):
            logging.warning(f"Directory does not exist: {path}")
            os.makedirs(path)  # Create the directory if it doesn't exist
            logging.info(f"Created directory: {path}")

validate_paths()

# Additional configuration parameters can be added as needed
