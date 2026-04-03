import kagglehub

# Download latest version
path = kagglehub.dataset_download("andreasmoegelmose/multiview-traffic-intersection-dataset")

print("Path to dataset files:", path)