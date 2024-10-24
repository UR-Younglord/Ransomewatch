import os
import pandas as pd
import math
from collections import Counter

def scan_directory(directory):
    """Scans a directory for files and extracts their characteristics."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            try:
                file_size = os.path.getsize(file_path)
                file_extension = os.path.splitext(filename)[1]
                last_modified = os.path.getmtime(file_path)
                creation_time = os.path.getctime(file_path)
                permissions = get_permission_string(os.stat(file_path).st_mode)
                entropy = calculate_entropy(file_path)
                files.append([file_path, file_size, file_extension, last_modified, creation_time, permissions, entropy])
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return pd.DataFrame(files, columns=['File Path', 'File Size', 'File Extension', 'Last Modified', 'Creation Time', 'Permissions', 'Entropy'])

def get_permission_string(mode):
    """Converts numeric permission mode to a readable string."""
    is_readable = 'r' if mode & 0o400 else '-'
    is_writable = 'w' if mode & 0o200 else '-'
    is_executable = 'x' if mode & 0o100 else '-'
    return is_readable + is_writable + is_executable

def calculate_entropy(file_path):
    """Calculates the Shannon entropy of the file to assess randomness."""
    with open(file_path, 'rb') as file:
        data = file.read()

    if len(data) == 0:
        return 0  # Return zero entropy for empty files

    byte_counts = Counter(data)
    total_bytes = len(data)

    # Calculate the entropy
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_counts.values() if count > 0)
    return entropy
