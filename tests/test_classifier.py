import unittest
import pandas as pd
from classifiers.ensemble_classifier import EnsembleClassifier

class TestEnsembleClassifier(unittest.TestCase):
    def setUp(self):
        self.model = EnsembleClassifier()
        self.X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        self.y = pd.Series([0, 1, 0])

    def test_train(self):
        self.model.train(self.X, self.y)
        # Check that the model has been fitted (e.g., classifier's attribute changes)
        self.assertTrue(hasattr(self.model.model, "estimators_"))
        self.assertGreater(len(self.model.model.estimators_), 0)

    def test_predict(self):
        self.model.train(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))  # Assuming binary classification

    def test_train_empty_data(self):
        empty_X = pd.DataFrame(columns=['feature1', 'feature2'])
        empty_y = pd.Series(dtype=int)
        with self.assertRaises(ValueError):
            self.model.train(empty_X, empty_y)

    def test_predict_untrained_model(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.X)

if __name__ == '__main__':
    unittest.main()
