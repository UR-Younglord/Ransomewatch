import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

class ClassifierAnalysis:
    def __init__(self, metrics_file):
        """Initialize the ClassifierAnalysis with the path to the metrics file."""
        if not os.path.isfile(metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        self.metrics_file = metrics_file
        self.metrics_df = pd.read_csv(metrics_file)

    def plot_accuracy(self, output_file='classifier_accuracy_comparison.png'):
        """Plot accuracy of classifiers."""
        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.metrics_df, x='Classifier', y='Accuracy', palette='viridis')
        plt.title('Classifier Accuracy Comparison')
        plt.xlabel('Classifier')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

    def plot_f1_score(self, output_file='classifier_f1_score_comparison.png'):
        """Plot F1 Score of classifiers."""
        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.metrics_df, x='Classifier', y='F1_Score', palette='plasma')
        plt.title('Classifier F1 Score Comparison')
        plt.xlabel('Classifier')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

    def plot_precision_recall(self, output_file='classifier_precision_recall_comparison.png'):
        """Plot Precision and Recall of classifiers."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.metrics_df.melt(id_vars=['Classifier'], value_vars=['Precision', 'Recall']),
                     x='Classifier', y='value', hue='variable', marker='o')
        plt.title('Classifier Precision and Recall Comparison')
        plt.xlabel('Classifier')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

    def display_metrics(self):
        """Display the performance metrics of classifiers."""
        print("Classifier Performance Metrics:")
        print(self.metrics_df)

    def plot_confusion_matrix(self, y_true, y_pred, output_file='confusion_matrix.png'):
        """Plot confusion matrix for model evaluation."""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(output_file)
        plt.show()

if __name__ == "__main__":
    analysis = ClassifierAnalysis('classifier_metrics.csv')  # Assuming metrics are stored in this file
    analysis.display_metrics()
    analysis.plot_accuracy()
    analysis.plot_f1_score()
    analysis.plot_precision_recall()

    # Example usage for confusion matrix plotting
    # Replace these with actual true labels and predictions
    y_true = [0, 1, 0, 1, 0]  # Sample true labels
    y_pred = [0, 0, 0, 1, 1]  # Sample predictions
    analysis.plot_confusion_matrix(y_true, y_pred)
