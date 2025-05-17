from pathlib import Path

CONFIG = {
    "data_path": Path("data/"),           # Root folder for your dataset
    "pretrain": False,                    # Set True for self-supervised pretraining mode
    "pretraining_label": "pretrain",      # Folder name under data_path for pretraining .npy files
    "downstream": "classifier",           # Options: 'classifier', 'super-resolution'
    "mode": "train",                      # Options: 'train', 'test'
    "batch_size": 32,                     # Batch size for dataloaders
    "split_ratio": 0.8,                   # Train/Val split ratio
    "num_workers": 4,                     # DataLoader workers
    "pin_memory": True,                   # DataLoader pin memory for GPU acceleration
}