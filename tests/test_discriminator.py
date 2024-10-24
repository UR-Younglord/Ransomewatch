import unittest
import pandas as pd
from models.discriminator import RansomwareDiscriminator

class TestRansomwareDiscriminator(unittest.TestCase):
    
    def setUp(self):
        self.model = RansomwareDiscriminator()
        self.sample_data = pd.DataFrame({
            'File Size': [1000, 2000, 3000],
            'File Extension': ['.exe', '.dll', '.exe'],
            'Last Modified': [1610000000, 1610000500, 1610001000],
            'Creation Time': [1609999000, 1609999500, 1610000000],
            'Permissions': ['read', 'write', 'read'],
            'Entropy': [3.5, 4.5, 3.0],
            'Label': [1, 0, 1]
        })
    
    def test_training(self):
        # Train the model
        self.model.train(self.sample_data)
        # Check that the model has been fitted
        self.assertTrue(hasattr(self.model.model, "estimators_"))
        self.assertGreater(len(self.model.model.estimators_), 0)

    def test_prediction(self):
        self.model.train(self.sample_data)
        predictions = self.model.predict(self.sample_data)
        self.assertEqual(len(predictions), len(self.sample_data))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))  # Assuming binary classification

    def test_train_empty_data(self):
        empty_data = pd.DataFrame(columns=['File Size', 'File Extension', 'Last Modified', 'Creation Time', 'Permissions', 'Entropy', 'Label'])
        with self.assertRaises(ValueError):
            self.model.train(empty_data)

    def test_predict_untrained_model(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.sample_data)

if __name__ == '__main__':
    unittest.main()
