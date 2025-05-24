import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from mae_vit import MAE_ViT
from data import test_dataset  # Loads the test dataloader from your data.py


class PretrainEvaluator:
    def __init__(self, config, model_path, train_losses, val_losses):
        """
        Initialize the evaluator.

        Args:
            config (dict): Configuration dictionary.
            model_path (str): Path to the saved model weights.
            train_losses (list): Training loss history.
            val_losses (list): Validation loss history.
        """
        self.config = config
        self.model_path = model_path
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MAE_ViT(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            emb_dim=config["emb_dim"],
            encoder_layer=config["encoder_layer"],
            encoder_head=config["encoder_head"],
            decoder_layer=config["decoder_layer"],
            decoder_head=config["decoder_head"],
            mask_ratio=config["mask_ratio"]
        ).to(self.device)

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print(f"Loaded model from {self.model_path}")

    def plot_loss_curve(self, save_path=None):
        epochs = list(range(1, len(self.train_losses) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_losses, 'bo-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'ro-', label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Epoch vs Training & Validation Loss")
        plt.legend()
        plt.grid()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def visualize_reconstructions(self, num_images=5, output_dir="reconstructions"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

        _, test_loader = test_dataset(self.config)

        with torch.no_grad():
            for batch in test_loader:
                images = batch.to(self.device)
                break

        reconstructed_images, masks = self.model(images[:num_images])

        images = images[:num_images].cpu().numpy().transpose(0, 2, 3, 1)
        reconstructed_images = reconstructed_images.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(num_images):
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            mask = masks[i] > 0.5
            masked_image = images[i] * (1 - mask)
            visible_image = masked_image + reconstructed_images[i] * mask

            axes[0].imshow(images[i], cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(masked_image, cmap='gray')
            axes[1].set_title("Masked")
            axes[1].axis("off")

            axes[2].imshow(reconstructed_images[i], cmap='gray')
            axes[2].set_title("Reconstructed")
            axes[2].axis("off")

            axes[3].imshow(visible_image, cmap='gray')
            axes[3].set_title("Visible")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"reconstruction_{i + 1}.png"))
            plt.close(fig)

    def evaluate(self, loss_plot_path=None, num_images=5):
        """
        Evaluate the model: plot losses and save reconstruction images.
        """
        self.plot_loss_curve(save_path=loss_plot_path)
        self.visualize_reconstructions(num_images=num_images)
        print("Pretraining evaluation completed.")
