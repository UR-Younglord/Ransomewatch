import pandas as pd
import os
import json
from typing import List, Dict

def generate_features(report_dir: str = 'dataset/reports', 
                      csv_file: str = 'C:/projects/ransomware_detection/dataset/ransomware_dataset/Ransomware PE Header Feature Dataset/Ransomware_headers.csv') -> pd.DataFrame:
    """
    Generates raw features from report files in report_dir and includes features from the CSV file.

    :param report_dir: Directory containing report files.
    :param csv_file: Path to the CSV file containing ransomware dataset.
    :return: DataFrame containing combined raw features.
    """
    feature_list = []

    # Process JSON report files
    for filename in os.listdir(report_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(report_dir, filename), 'r') as f:
                    report = json.load(f)
                    features = extract_features(report)  # Extract features from JSON
                    feature_list.append(features)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {filename}: {e}")

    # Process CSV file for additional features
    if os.path.exists(csv_file):
        csv_features = pd.read_csv(csv_file)
        csv_feature_data = extract_csv_features(csv_features)  # Extract relevant features from the CSV
        feature_list.extend(csv_feature_data)
    else:
        print(f"CSV file not found: {csv_file}")

    return pd.DataFrame(feature_list)

def extract_features(report: Dict) -> Dict:
    """Extract relevant features from the report."""
    # Replace this with actual feature extraction logic
    return {
        'feature1': report.get('key1', 0),
        'feature2': report.get('key2', 0),
        # Add more features as needed
    }

def extract_csv_features(csv_data: pd.DataFrame) -> List[Dict]:
    """Extract relevant features from the CSV data."""
    csv_features = []
    for index, row in csv_data.iterrows():
        csv_features.append({
            'csv_feature1': row.get('filename', ''),
            'csv_feature2': row.get('GR', 0),
            'csv_feature3': row.get('family', ''),
            # Add more features based on your dataset structure
        })
    return csv_features

if __name__ == "__main__":
    features_df = generate_features()
    print(features_df)
