import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("solarmainframe/ids-intrusion-csv")

print("Path to dataset files:", path)
