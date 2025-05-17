import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor, Resize
from torch.utils.data import DataLoader, random_split
from config import CONFIG as config 

class CustomDataset(Dataset):
    def __init__(self, config: dict, transform=None, target_transform=None):
        """
        Custom dataset loader for different training modes (pretraining, classification, super-resolution).

        Args:
            config (dict): Configuration dictionary with keys:
                - data_path (str or Path): Root directory for data.
                - pretrain (bool): If True, load data for self-supervised pretraining.
                - downstream (str): Mode for downstream task. Options: 'classifier', 'super-resolution'.
            transform (callable, optional): Transformation to apply on input data.
            target_transform (callable, optional): Transformation to apply on target (label or HR image).
        """
        self.data_path = Path(config["data_path"])
        self.transform = transform
        self.target_transform = target_transform
        self.pretrain = config.get("pretrain", False)
        self.downstream = config.get("downstream", None)

        # Handle different modes
        if self.pretrain:
            # Load all .npy files recursively for self-supervised learning
            image = Path(self.data_path/config["pretraining_label"])
            self.file_paths = list(image_path.rglob("*.npy"))

        elif self.downstream == "classifier":
            # Load .npy files organized as: /class_name/*.npy
            self.file_paths = list(self.data_path.rglob("*/*.npy"))
            self.labels = [path.parent.name for path in self.file_paths]
            self.classes = sorted(set(self.labels))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        elif self.downstream == "super-resolution":
            # Load LR and HR pairs from subfolders
            self.file_paths = list((self.data_path / "LR").glob("*.npy"))

        else:
            raise ValueError("Either 'pretrain' must be True or valid 'downstream' must be specified.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        if self.pretrain:
            # Axion-style datasets may wrap array in a list/array
            if file_path.parent.name == "axion":
                img = np.load(file_path, allow_pickle=True)[0]
            else:
                img = np.load(file_path)
            
            if self.transform:
                img = self.transform(img)
            return img

        elif self.downstream == "classifier":
            img = np.load(file_path)
            if self.transform:
                img = self.transform(img)

            label = self.labels[idx]
            label_idx = self.class_to_idx[label]
            return img, label_idx

        elif self.downstream == "super-resolution":
            LR = np.load(file_path)
            HR_path = self.data_path / "HR" / file_path.name
            HR = np.load(HR_path)

            if self.transform:
                LR = self.transform(LR)
            if self.target_transform:
                HR = self.target_transform(HR)
            return LR, HR

        else:
            raise ValueError("Invalid data loading configuration.")

def train_dataset(config):
    """
    Builds a dataset and dataloader pipeline using settings from config.
    
    Returns:
        train_loader (DataLoader): Dataloader for training set.
        test_loader (DataLoader): Dataloader for test set.
    """
    # Define appropriate transforms based on the task
    if config["pretrain"] or config["downstream"] == "classifier":
        transform = Compose([
            ToTensor(),  
            Lambda(lambda x: x.float())
        ])
        target_transform = None

    elif config["downstream"] == "super-resolution":
        transform = Compose([
            Lambda(lambda x: torch.from_numpy(x).float()),
            Resize((64, 64))  # Resize only LR images if needed
        ])
        target_transform = Compose([
            Lambda(lambda x: torch.from_numpy(x).float())
        ])
    else:
        transform = None
        target_transform = None

    # Initialize dataset based on the training mode
    dataset = CustomDataset(
        config=config,
        transform=transform,
        target_transform=target_transform
    )

    # Train/Test split
    train_size = int(config["split_ratio"] * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_set, 
                              batch_size=config["batch_size"], 
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(test_set, 
                             batch_size=config["batch_size"], 
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=True)

    return train_loader, val_loader



        