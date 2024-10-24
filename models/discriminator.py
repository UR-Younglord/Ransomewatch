import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

class RansomwareDiscriminator:
    def __init__(self, n_estimators=100, max_depth=None):
        """Initialize the RandomForestClassifier with given parameters."""
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def train(self, df):
        """Train the discriminator model with cross-validation."""
        required_columns = ['File Size', 'File Extension', 'Last Modified', 'Creation Time', 'Permissions', 'Entropy', 'Label']
        
        # Check if all required columns are in the DataFrame
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f'Missing required column: {col}')

        # Prepare the features and labels
        X = df[required_columns[:-1]]
        y = df['Label']  # Ensure you have a 'Label' column in your DataFrame
        
        # Convert categorical features if necessary
        X = pd.get_dummies(X, drop_first=True)
        
        # Perform cross-validation
        scores = cross_val_score(self.model, X, y, cv=5)  # 5-Fold Cross Validation
        print(f'Cross-Validation Accuracy: {scores.mean()}')
        
        # Fit the model
        self.model.fit(X, y)

    def predict(self, df):
        """Predicts ransomware files based on input DataFrame."""
        if not all(col in df.columns for col in ['File Size', 'File Extension', 'Last Modified', 'Creation Time', 'Permissions', 'Entropy']):
            raise ValueError("Input DataFrame must contain the required columns.")

        # Prepare the features for prediction
        X_pred = df[['File Size', 'File Extension', 'Last Modified', 'Creation Time', 'Permissions', 'Entropy']]
        X_pred = pd.get_dummies(X_pred, drop_first=True)

        return self.model.predict(X_pred)

# Example usage
# if __name__ == "__main__":
#     df = pd.read_csv('your_dataset.csv')  # Load your dataset
#     discriminator = RansomwareDiscriminator(n_estimators=200, max_depth=10)
#     discriminator.train(df)
#     predictions = discriminator.predict(df)
#     print(classification_report(df['Label'], predictions))
