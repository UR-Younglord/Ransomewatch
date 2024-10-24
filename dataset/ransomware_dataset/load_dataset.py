import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the dataset path from the config
dataset_path = r"C:\projects\ransomware_detection\dataset\ransomware_dataset\Ransomware PE Header Feature Dataset\Ransomware_headers.csv"

# Function to load and preview the dataset
def load_dataset(path):
    """Load the dataset and return a DataFrame."""
    try:
        if os.path.exists(path):
            data = pd.read_csv(path)
            logging.info("Dataset loaded successfully.")
            return data
        else:
            logging.error(f"File not found: {path}")
            return None
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

# Load the dataset
data = load_dataset(dataset_path)

# Check if data was loaded successfully
if data is not None:
    # Preview the first few rows
    print(data.head())

    # Additional exploration
    print(f"Shape of dataset: {data.shape}")
    print(data.describe())
else:
    logging.warning("No data to preview.")
