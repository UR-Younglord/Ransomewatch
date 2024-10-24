import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, output_file='classifier_metrics.csv', plot_cm=True):
    """Evaluate the model performance and print metrics.
    
    Parameters:
        y_true (list): Actual labels.
        y_pred (list): Predicted labels.
        output_file (str): Path to save metrics CSV file.
        plot_cm (bool): Whether to plot confusion matrix.
    """
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Create a DataFrame for metrics
    metrics = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    metrics_df = pd.DataFrame(metrics)
    
    # Print classification report
    print("Classification Report:\n", classification_report(y_true, y_pred))
    
    # Save metrics to CSV
    metrics_df.to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")

    # Confusion Matrix
    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Example usage
# if __name__ == "__main__":
#     # y_true = [...]  # Replace with actual labels
#     # y_pred = [...]  # Replace with predicted labels
#     # evaluate_model(y_true, y_pred)
