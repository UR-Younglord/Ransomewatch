import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(num_samples: int, 
                             min_file_size: int = 1000, 
                             max_file_size: int = 100000, 
                             min_entropy: float = 0.5, 
                             max_entropy: float = 7.0,
                             random_state: int = None) -> pd.DataFrame:
    """
    Generates realistic synthetic data to augment training.

    Parameters:
        num_samples (int): Number of synthetic samples to generate.
        min_file_size (int): Minimum file size in bytes.
        max_file_size (int): Maximum file size in bytes.
        min_entropy (float): Minimum entropy value.
        max_entropy (float): Maximum entropy value.
        random_state (int): Seed for random number generator for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic data with additional features.
    """
    if random_state is not None:
        np.random.seed(random_state)

    synthetic_data = []
    for i in range(num_samples):
        # Generate file size from log-normal distribution for realism
        file_size = int(np.random.lognormal(mean=np.log((max_file_size + min_file_size) / 2), 
                                             sigma=0.5))
        file_size = min(max(file_size, min_file_size), max_file_size)  # Ensure within bounds
        
        file_extension = np.random.choice(['.exe', '.dll', '.locked', '.encrypted'])
        
        # Generate last modified and creation time with finer granularity
        last_modified = datetime.now() - timedelta(days=np.random.randint(1, 365),
                                                    hours=np.random.randint(0, 24),
                                                    minutes=np.random.randint(0, 60))
        creation_time = last_modified - timedelta(days=np.random.randint(1, 30),
                                                   hours=np.random.randint(0, 24),
                                                   minutes=np.random.randint(0, 60))
        
        permissions = np.random.choice(['read', 'write', 'execute'])
        
        # Conditional entropy based on file extension
        entropy = np.random.uniform(min_entropy, max_entropy) if file_extension in ['.exe', '.dll'] else np.random.uniform(1, max_entropy)

        file_path = f'synthetic_path_{i+1}{file_extension}'  # Unique file paths
        
        # Adding more realistic attributes
        owner = np.random.choice(['user1', 'user2', 'user3'])
        last_accessed = last_modified - timedelta(days=np.random.randint(0, 30), hours=np.random.randint(0, 24))

        synthetic_data.append([file_path, file_size, file_extension, last_modified.timestamp(), 
                               creation_time.timestamp(), last_accessed.timestamp(), 
                               permissions, owner, entropy])
    
    return pd.DataFrame(synthetic_data, columns=['File Path', 'File Size', 'File Extension', 
                                                 'Last Modified', 'Creation Time', 'Last Accessed',
                                                 'Permissions', 'Owner', 'Entropy'])

# Example usage
# synthetic_df = generate_synthetic_data(100, random_state=42)
# print(synthetic_df.head())
