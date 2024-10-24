import os
import pandas as pd
from processing.col_reports import collect_reports
from processing.feature_gen import generate_features, extract_features  # Adjusted import for extract_features
from processing.feature_dict import get_feature_dict
from processing.feature_fin import generate_dataset
from classifiers.ensemble_classifier import EnsembleClassifier
from analysis.performance_analysis import evaluate_model

def main():
    # Step 1: Collect reports from the specified directory
    print("Collecting reports...")
    collect_reports() 

    # Load the dataset
    dataset_path = r"C:\projects\ransomware_detection\dataset\ransomware_dataset\Ransomware PE Header Feature Dataset\Ransomware_headers.csv"
    if not os.path.isfile(dataset_path):
        print("Dataset file does not exist at the specified path:", dataset_path)
        return

    data = pd.read_csv(dataset_path) 
    print("Loaded dataset:")
    print(data.head())  # Display the first few rows

    # Check for missing values
    print("Missing values in dataset:")
    print(data.isnull().sum())

    # Fill missing values or drop rows/columns
    data.fillna(method='ffill', inplace=True)  

    # Check data types
    print("Data types in dataset:")
    print(data.dtypes)

    # Convert categorical features to numerical if necessary
    if 'family' in data.columns:
        data['family'] = data['family'].astype('category').cat.codes

    # Check the dataset shape
    print("Dataset shape:", data.shape)

    # Step 2: Generate raw features from collected reports
    print("Generating raw features...")
    raw_features = generate_features() 

    # Step 3: Extract features using the extract_features function
    extracted_features = extract_features(raw_features)  # Call the extract_features function
    print("Extracted features shape:", extracted_features.shape)

    # Extract features and labels
    X = data.iloc[:, :-1]  # Adjust column slicing as necessary
    y = data.iloc[:, -1]
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)

    # Step 4: Get total unique feature list from extracted features
    print("Extracting unique features...")
    unique_features = get_feature_dict(extracted_features)  # Use extracted features here

    # Step 5: Generate dataset based on occurrences of unique features
    print("Creating final dataset...")
    dataset = generate_dataset(unique_features) 
    print("Final dataset shape:", dataset.shape)
    print("Final dataset columns:", dataset.columns.tolist())

    # Ensure there is data before training
    if 'label' not in dataset.columns:
        print("No 'label' column found in the dataset!")
        return

    X, y = dataset.drop('label', axis=1), dataset['label'] 
    if X.empty or y.empty:
        print("No data to train on!")
        return

    # Step 6: Train the ensemble model and evaluate performance
    print("Training the model...")
    model = EnsembleClassifier()
    model.train(X, y)
    y_pred = model.predict(X)

    print("Evaluating model performance...")
    metrics = evaluate_model(y, y_pred)

    # Save metrics to CSV
    metrics_path = "classifier_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
