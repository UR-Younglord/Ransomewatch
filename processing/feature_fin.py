import pandas as pd

def generate_dataset(features_df):
    """
    Generates the final dataset based on occurrences of unique features.

    :param features_df: DataFrame containing unique features.
    :return: Final dataset DataFrame with counts and labels.
    """
    if features_df.empty:
        raise ValueError("Input DataFrame is empty. Please provide a valid DataFrame.")

    # Group by unique features and count occurrences
    final_dataset = features_df.groupby(list(features_df.columns)).size().reset_index(name='count')

    # Example labeling logic: Change this based on your criteria
    final_dataset['label'] = final_dataset['count'].apply(lambda x: 1 if x > 1 else 0)  # Adjust threshold as needed

    return final_dataset

if __name__ == "__main__":
    # Example DataFrame for testing
    example_df = pd.DataFrame({'feature1': [1, 1, 2], 'feature2': [3, 4, 3]})
    final_dataset = generate_dataset(example_df)
    print(final_dataset)
