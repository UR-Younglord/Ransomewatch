import unittest
import pandas as pd
from models.generator import generate_synthetic_data

class TestDataGenerator(unittest.TestCase):
    
    def test_generate_synthetic_data(self):
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(num_samples=100)
        
        # Check the number of samples
        self.assertEqual(len(synthetic_data), 100)

        # Check for expected columns
        expected_columns = ['File Path', 'File Size', 'File Extension', 'Last Modified', 'Creation Time', 'Permissions', 'Entropy']
        for column in expected_columns:
            self.assertIn(column, synthetic_data.columns)

        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(synthetic_data['File Size']))
        self.assertTrue(pd.api.types.is_numeric_dtype(synthetic_data['Entropy']))
        self.assertTrue(synthetic_data['File Extension'].isin(['.exe', '.dll', '.locked', '.encrypted']).all())

    def test_generate_zero_samples(self):
        synthetic_data = generate_synthetic_data(num_samples=0)
        self.assertEqual(len(synthetic_data), 0)

    def test_generate_large_samples(self):
        synthetic_data = generate_synthetic_data(num_samples=10000)
        self.assertEqual(len(synthetic_data), 10000)

if __name__ == '__main__':
    unittest.main()
