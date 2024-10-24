import pandas as pd

def get_feature_dict(features_df, return_as='DataFrame'):
    """
    Extracts unique features from the DataFrame.

    :param features_df: DataFrame containing raw features.
    :param return_as: Specify the format for the output - 'DataFrame' or 'list'.
    :return: Unique features as a DataFrame or a list.
    """
    if features_df.empty:
        print("Warning: The input DataFrame is empty.")
        return pd.DataFrame() if return_as == 'DataFrame' else []

    unique_features = features_df.drop_duplicates()

    if return_as == 'list':
        return unique_features.values.flatten().tolist()  # Flatten to a list if needed

    return unique_features

if __name__ == "__main__":
    # Example DataFrame for testing
    example_df = pd.DataFrame({'feature1': [1, 2, 1], 'feature2': [3, 4, 3]})
    unique_features_df = get_feature_dict(example_df)
    print("Unique Features as DataFrame:\n", unique_features_df)

    unique_features_list = get_feature_dict(example_df, return_as='list')
    print("Unique Features as List:\n", unique_features_list)
