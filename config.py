CONFIG = {
    "data_path": "/path/to/dataset",           # Root path to the dataset
    "pretraining_label":"no_sub",              # Options: "cdm", "axion"
    "pretrain": False,                         # True for pretraining, False for downstream tasks
    "downstream": "super-resolution",          # Options: "classifier", "super-resolution", or None
    "batch_size": 256,                         # Batch size for DataLoader
    "split_ratio": 0.9,                        # Train/test split ratio
}
