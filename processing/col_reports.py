import os
import shutil

def collect_reports(source_dir='C:/projects/ransomware_detection/dataset', 
                    target_dir='dataset/reports', 
                    report_types=['.json', '.txt'], 
                    verbose=True):
    """
    Collects report files from source_dir and copies them to target_dir.
    
    :param source_dir: Directory where reports are stored.
    :param target_dir: Directory where reports will be copied.
    :param report_types: List of report file extensions to copy.
    :param verbose: Whether to print messages about copied files.
    """
    # Ensure target directory is relative to the current working directory
    target_dir = os.path.join(os.getcwd(), target_dir)

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if any(filename.endswith(ext) for ext in report_types):
            try:
                shutil.copy(os.path.join(source_dir, filename), target_dir)
                if verbose:
                    print(f"Copied: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")

if __name__ == "__main__":
    collect_reports()  # Call the function for testing
