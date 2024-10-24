import unittest
from processing.feature_dict import extract_features
import pandas as pd

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'File Path': ['file1.exe', 'file2.dll'],
            'File Size': [12345, 67890],
            'File Extension': ['exe', 'dll']
        })

    def test_extract_features(self):
        features = extract_features(self.sample_data)
        
        # Check that expected features are in the DataFrame
        expected_features = ['feature1', 'feature2', 'feature3']  # Update with actual expected features
        for feature in expected_features:
            self.assertIn(feature, features.columns)

        # Validate that features have expected properties
        self.assertGreater(len(features), 0)  # Ensure there are features extracted
        self.assertEqual(features.shape[0], self.sample_data.shape[0])  # Ensure the number of rows is the same

    def test_extract_features_empty_data(self):
        empty_data = pd.DataFrame(columns=['File Path', 'File Size', 'File Extension'])
        features = extract_features(empty_data)
        self.assertTrue(features.empty, "Expected features DataFrame to be empty for empty input data.")

    def test_extract_features_with_missing_values(self):
        data_with_missing = pd.DataFrame({
            'File Path': ['file1.exe', None],
            'File Size': [12345, 67890],
            'File Extension': ['exe', 'dll']
        })
        features = extract_features(data_with_missing)
        
        # Check that the feature extraction still works with missing values
        self.assertIn('feature1', features.columns)
        self.assertEqual(features.shape[0], 1)  # Adjust based on how missing values are handled

if __name__ == '__main__':
    unittest.main()
