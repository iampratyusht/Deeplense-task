# Multi-Class Image Classification using ResNet18

## Overview

This task implements a multi-class classification model classifying the images into lenses using **ResNet18**, fine-tuned for a custom dataset containing **.npy** image files. The model is trained, validated, and tested using PyTorch and `torchvision`. Performance is evaluated using accuracy, loss plots, confusion matrices, and ROC AUC curves.

## Dataset

- The dataset consists of `.npy` images categorized into **3 classes** i.e. strong lensing images with **no substructure**, **subhalo substructure**, and **vortex substructure**.
- `Train Data` is splitted into 90:10 `train-test` split, and model evaluated on `val`.
- Custom `NumpyDataset` class is used to load `.npy` files and apply necessary transformations.

## Model Architecture

- **Base Model**: ResNet18
- **Modifications**:
  - Replaced the final fully connected layer with a custom classifier:
    ```python
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(64, 3)
    )
    ```
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW (learning rate = `1e-3`)

## Training

- **Dataset split**: 90% training, 10% test.
- **Batch size**: 128
- **Epochs**: 20 epochs

## Performance Metrics

1. **Loss & Accuracy Curves**&#x20;
```python
y_true, y_pred = compute_predictions(model, test_dataloader, num_classes=3, device="cuda")
```
![Epoch vs Loss Curve](./Epoch%20vs%20Loss.png)![Epoch vs Accuracy Curve](./Epoch%20vs%20Accuracy.png)
2. **Confusion Matrix** (for 3 classes)&#x20;
```python
plot_confusion_matrices(y_true, y_pred, class_labels=data.classes)
```
![Confusion Matrix](./Confusion%20matrix.png)

3. **ROC AUC Curves**&#x20;
```python
plot_roc_curves(y_true, y_pred, num_classes=3)
```
![ROC Curve](./ROC%20curve.png)

## Results & Discussion

- Model weights are saved as checkpoints named `common_task_wt.pth`

*Developed using PyTorch & Torchvision.*
