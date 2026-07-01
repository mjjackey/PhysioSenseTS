import pandas as pd
import os

def save_features(feature_df, data_path, subject_id):
    """
    Save the features to a file.
    Args: feature_df (pandas DataFrame): The features DataFrame.
    data_path (str): The path to save the features.
    """
    if not isinstance(feature_df, pd.DataFrame):
        print("Error: feature_df is not a pandas DataFrame.")
        return
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    feature_df.to_parquet(f"{data_path}/{subject_id}_features.parquet")
    print(f"Features saved to {data_path}/{subject_id}_features.parquet")

def load_features(data_path, subject_id):
    """
    Load the features from a file.
    Args: data_path (str): The path to load the features.
    subject_id (str): The subject ID.
    Returns: pandas DataFrame: The features DataFrame.
    """
    if not os.path.exists(f"{data_path}/{subject_id}_features.parquet"):
        print("Error: features file not found.")
        return None
    feature_df = pd.read_parquet(f"{data_path}/{subject_id}_features.parquet")
    return feature_df