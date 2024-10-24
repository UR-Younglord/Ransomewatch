import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO)

class EnsembleClassifier:
    def __init__(self, n_estimators1=100, n_estimators2=200, random_state=42):
        """Initialize the ensemble classifier with two Random Forest classifiers."""
        self.classifier1 = RandomForestClassifier(n_estimators=n_estimators1, random_state=random_state)
        self.classifier2 = RandomForestClassifier(n_estimators=n_estimators2, random_state=random_state)

        # Use VotingClassifier for ensemble method
        self.model = VotingClassifier(estimators=[
            ('rf1', self.classifier1),
            ('rf2', self.classifier2)
        ], voting='hard')

    def clean_data(self, X):
        """Clean the data by replacing empty strings, converting to numeric, and dropping NaNs."""
        # Replace empty strings with NaN
        X.replace('', pd.NA, inplace=True)
        
        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values
        X_cleaned = X.dropna()
        logging.info("Shape of X after cleaning: %s", X_cleaned.shape)
        
        if X_cleaned.shape[0] == 0:
            raise ValueError("X contains no valid samples after cleaning.")
        
        return X_cleaned

    def train(self, X, y):
        """Train the ensemble model with cleaned data."""
        # Convert X to DataFrame if it's not already
        X = pd.DataFrame(X)
        logging.info("Initial X shape: %s", X.shape)

        X_cleaned = self.clean_data(X)

        # Ensure y is numeric
        y = pd.Series(y)
        y = pd.to_numeric(y, errors='coerce')
        y_cleaned = y.dropna()

        if y_cleaned.shape[0] == 0:
            raise ValueError("y contains no valid labels after cleaning.")
        
        # Check if X and y have the same length
        if len(X_cleaned) != len(y_cleaned):
            raise ValueError("The number of samples in X and y do not match.")
        
        # Fit the model
        self.model.fit(X_cleaned, y_cleaned)

    def predict(self, X):
        """Make predictions using the trained model."""
        X_cleaned = self.clean_data(pd.DataFrame(X))  # Clean the input data
        return self.model.predict(X_cleaned)

    def score(self, X, y):
        """Evaluate the model's accuracy."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
