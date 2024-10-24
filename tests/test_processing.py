import unittest
import pandas as pd
from processing.feature_gen import generate_features
from processing.feature_dict import get_feature_dict
from processing.feature_fin import generate_dataset

class TestProcessing(unittest.TestCase):
    
    def test_generate_features(self):
        raw_features = generate_features()
        self.assertIsInstance(raw_features, pd.DataFrame)
        # Check if the DataFrame has the expected columns
        expected_columns = ['feature1', 'feature2']  # Adjust according to your expected features
        for column in expected_columns:
            self.assertIn(column, raw_features.columns)

    def test_get_feature_dict(self):
        raw_features = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [1, 2, 1]
        })
        unique_features = get_feature_dict(raw_features)
        self.assertGreater(len(unique_features), 0)
        self.assertIsInstance(unique_features, pd.DataFrame)
        self.assertIn('Feature1', unique_features.columns)
        self.assertIn('Feature2', unique_features.columns)

    def test_generate_dataset(self):
        unique_features = pd.DataFrame({'Feature1': [1], 'Feature2': [1]})
        dataset = generate_dataset(unique_features)
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertIn('count', dataset.columns)  # Assuming 'count' is generated
        self.assertIn('label', dataset.columns)  # Assuming 'label' is generated
        self.assertGreater(len(dataset), 0)

    def test_empty_features(self):
        empty_df = pd.DataFrame(columns=['Feature1', 'Feature2'])
        unique_features = get_feature_dict(empty_df)
        self.assertEqual(len(unique_features), 0)

if __name__ == '__main__':
    unittest.main()
