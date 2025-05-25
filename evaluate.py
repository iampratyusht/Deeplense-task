import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from model import MAE_ViT
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



class ClassifierEvaluator:
    def __init__(self, model, test_loader, class_labels, save_path="plots", device=None):
        """
        Args:
            model (nn.Module): Trained classification model.
            test_loader (DataLoader): DataLoader for the test dataset.
            class_labels (list): List of class labels (e.g., ["class0", "class1", "class2"]).
            save_path (str): Directory to save all plots.
            device (str): Device string, defaults to CUDA if available.
        """
        self.model = model
        self.test_loader = test_loader
        self.class_labels = class_labels
        self.save_path = save_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.save_path, exist_ok=True)

    def accuracy_fn(self, y_true, y_pred):
        """
        Computes and prints overall classification accuracy.

        Args:
            y_true (np.ndarray): One-hot encoded true labels.
            y_pred (np.ndarray): One-hot encoded predicted labels.
        """
        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def compute_predictions(self, num_classes=3):
        self.model.to(self.device)
        self.model.eval()
        outputs = torch.empty(len(self.test_loader.dataset), num_classes, device=self.device)
        y_pred = torch.zeros_like(outputs)
        y_true = torch.zeros_like(outputs)
        with torch.no_grad():
            start_idx = 0
            for X, y in tqdm(self.test_loader):
                X, y = X.to(self.device), y.to(self.device)
                batch_size = X.shape[0]
                batch_outputs = self.model(X)
                outputs[start_idx: start_idx + batch_size] = batch_outputs
                y_true[torch.arange(start_idx, start_idx + batch_size), y] = 1
                start_idx += batch_size
        y_pred[torch.arange(outputs.shape[0]), torch.argmax(outputs, dim=1)] = 1
        y_pred = y_pred.to(torch.int)
        return y_true.cpu().numpy(), y_pred.cpu().numpy(), outputs.softmax(dim=1).cpu().numpy()

    def plot_training_metrics(self, train_loss, test_loss, train_acc, test_acc, lr_history=None):
        epochs = list(range(1, len(train_loss) + 1))

        # Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label="Train Loss", marker='o', linestyle='-')
        plt.plot(epochs, test_loss, label="Test Loss", marker='s', linestyle='--', color='r')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Epoch vs Loss")
        plt.legend()
        plt.grid(True)
        loss_plot = os.path.join(self.save_path, "loss_curve.png")
        plt.savefig(loss_plot)
        plt.close()

        # Accuracy Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc, label="Train Accuracy", marker='o', linestyle='-')
        plt.plot(epochs, test_acc, label="Test Accuracy", marker='s', linestyle='--', color='g')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Epoch vs Accuracy")
        plt.legend()
        plt.grid(True)
        acc_plot = os.path.join(self.save_path, "accuracy_curve.png")
        plt.savefig(acc_plot)
        plt.close()

        # Learning Rate Plot
        if lr_history is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, lr_history, label="Lr History", marker='x', linestyle='-')
            plt.xlabel("Epochs")
            plt.ylabel("Learning Rate")
            plt.title("Epoch vs Learning rate")
            plt.legend()
            plt.grid(True)
            lr_plot = os.path.join(self.save_path, "lr_curve.png")
            plt.savefig(lr_plot)
            plt.close()

    def plot_confusion_matrices(self, y_true, y_pred):
        fig, axes = plt.subplots(1, len(self.class_labels), figsize=(5 * len(self.class_labels), 4))

        for i, (label, ax) in enumerate(zip(self.class_labels, axes)):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{label} (Class {i})")

        plt.tight_layout()
        cm_plot = os.path.join(self.save_path, "confusion_matrices.png")
        plt.savefig(cm_plot)
        plt.close()

    def plot_roc_curves(self, y_true_np, output_np):
        plt.figure(figsize=(12, 6))

        for i in range(len(self.class_labels)):
            fpr, tpr, _ = roc_curve(y_true_np[:, i], output_np[:, i])
            roc_auc = auc(fpr, tpr)
            print(f"Class {i} : AUC = {roc_auc:.2f}")
            plt.plot(fpr, tpr, label=f'{self.class_labels[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend(loc='lower right')
        roc_plot = os.path.join(self.save_path, "roc_curves.png")
        plt.savefig(roc_plot)
        plt.close()

def compute_psnr(img1, img2, max_value=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr

class SREvaluator:
    def __init__(self, model, dataset, device, save_dir):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def compare_images(self, n=5):
        self.model.eval()
        indices = np.random.choice(len(self.dataset), n, replace=False)

        for i, idx in enumerate(indices):
            lr_img, hr_img = self.dataset[idx]
            lr_img_tensor = lr_img.unsqueeze(0).to(self.device)
            hr_img_np = hr_img.numpy()

            with torch.no_grad():
                sr_img = self.model(lr_img_tensor).cpu().numpy()[0, 0]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(lr_img.cpu().numpy()[0, 0], cmap='gray')
            axes[0].set_title("Low-Resolution Image")
            axes[0].axis("off")

            axes[1].imshow(sr_img, cmap='gray')
            axes[1].set_title("Super-Resolved Image")
            axes[1].axis("off")

            axes[2].imshow(hr_img_np.squeeze(), cmap='gray')
            axes[2].set_title("Ground Truth High-Resolution Image")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"comparison_{i+1}.png")
            plt.savefig(save_path)
            plt.close(fig)

    def plot_metrics(self, train_losses, val_losses, ssim_scores, psnr_scores):
        epochs = range(1, len(train_losses) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(epochs, train_losses, label="Train Loss")
        axes[0].plot(epochs, val_losses, label="Val Loss")
        axes[0].set_title("Loss Curve")
        axes[0].legend()

        axes[1].plot(epochs, ssim_scores, label="SSIM", color='orange')
        axes[1].set_title("SSIM Curve")
        axes[1].legend()

        axes[2].plot(epochs, psnr_scores, label="PSNR", color='green')
        axes[2].set_title("PSNR Curve")
        axes[2].legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "metrics_plot.png")
        plt.savefig(save_path)
        plt.close(fig)
