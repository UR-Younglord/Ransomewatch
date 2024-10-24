import kagglehub
import os

def download_dataset(dataset_name):
    """Download the latest version of the specified dataset."""
    try:
        # Download dataset
        path = kagglehub.dataset_download(dataset_name)
        
        # Check if the dataset path exists
        if os.path.exists(path):
            print("Dataset downloaded successfully.")
            print("Path to dataset files:", path)
        else:
            print("Failed to download the dataset or path does not exist.")
            
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")

# Example usage
download_dataset("solarmainframe/ids-intrusion-csv")
