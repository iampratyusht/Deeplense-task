# Super-Resolution of Strong Lensing Images using Masked Autoencoder (MAE)

## Overview
This project fine-tunes a pre-trained Masked Autoencoder (MAE) model to perform super-resolution on strong lensing images. The model is trained to upscale low-resolution (LR) images using high-resolution (HR) samples as ground truths.

## Dataset
The dataset consists of `.npy` strong lensing images at multiple resolutions:
- **High-Resolution (HR)**
- **Low-Resolution (LR)**


## Approach
- Utilize a **pre-trained ViT based Masked Autoencoder (MAE)** as the backbone.
- Fine-tune the model to reconstruct HR images from LR inputs.
- Loss function: **Mean Squared Error (MSE)**.
- Evaluation metrics: **MSE, Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR)**.
- **Batch size**: 256
- **Optimizer**: AdamW with a learning rate scheduler.


## Implementation
The implementation is based on **PyTorch** and follows these key steps:
1. Load the pre-trained MAE model.
2. Modify the decoder to generate HR images from LR inputs.
    ```python
    reconstructed_img = interpolate(upsample_blocks(encoded_features))
    ```

## Performance Metrics
After fine-tuning, the model achieves the following results on the validation set:
- **MSE**: `0.0003`
- **SSIM**: `0.9549`
- **PSNR**: `35.8112 dB`
![Metrics Plot](.Evaluations/Epoch%20vs%20Loss,%20SSIM%20and%20PSNR.png)

## Results & Discussion
- The model successfully reconstructs fine details in strong lensing images.
![Upscaled Images](.Evaluations/Upscaled%20Images.png)
- Model weights are saved as checkpoints named `SR_wt.pth`
- Higher SSIM and PSNR indicate improved structural preservation and less noise.

---
This task is part of Task VI.A for fine-tuning an MAE model for super-resolution.

