# feature_importance.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the trained model."""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()

if __name__ == "__main__":
    # Example data loading and model training
    dataset_path = r"C:\projects\ransomware_detection\dataset\ransomware_dataset\Ransomware_headers.csv"  # Adjust the path
    data = pd.read_csv(dataset_path)

    # Extract features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Train the RandomForest model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Plot feature importance
    plot_feature_importance(model, X.columns)
