import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchinfo import summary
from model import MAE_ViT, MAE_SR
from evaluate import compute_psnr, SREvaluator
from skimage.metrics import structural_similarity as ssim

def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epochs, lr_scheduler, model_path):
    train_losses, val_losses, ssim_scores, psnr_scores = [], [], [], []
    best_psnr = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for lr_imgs, hr_imgs in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            loss = loss_fn(sr_imgs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss, mse_vals, ssim_vals, psnr_vals = 0, [], [], []

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_dataloader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = model(lr_imgs)
                loss = loss_fn(sr_imgs, hr_imgs)
                total_val_loss += loss.item()
                mse_vals.append(F.mse_loss(sr_imgs, hr_imgs).item())
                ssim_vals.append(ssim(hr_imgs.cpu().numpy()[0, 0], sr_imgs.cpu().numpy()[0, 0], data_range=1.0))
                psnr_vals.append(compute_psnr(hr_imgs.cpu().numpy(), sr_imgs.cpu().numpy()))

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_ssim = np.mean(ssim_vals)
        avg_psnr = np.mean(psnr_vals)

        val_losses.append(avg_val_loss)
        ssim_scores.append(avg_ssim)
        psnr_scores.append(avg_psnr)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), model_path)

        lr_scheduler.step()
    evaluator = SREvaluator(model, val_dataloader.dataset, device, save_dir=args.output_dir)
    evaluator.compare_images(n=5)
    evaluator.plot_metrics(train_losses, val_losses, ssim_scores, psnr_scores)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mae_model = MAE_ViT(
        image_size=64,
        patch_size=16,
        emb_dim=768,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75
    ).to(device)

    mae_model.load_state_dict(torch.load(args.mae_weights_path, map_location=device), strict=False)

    model_sr = MAE_SR(mae_model.encoder).to(device)

    summary(model_sr, input_size=(1, 1, 64, 64), verbose=0,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20, row_settings=["var_names"])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model_sr.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.02)
    lr_func = lambda epoch: min((epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / 50 * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    # Import or define your own DataLoader functions
    from data import train_dataset, test_dataset
    train_loader, val_loader = train_dataset(args.batch_size)
    

    train(model_sr, train_loader, val_loader, loss_fn, optimizer, device,
          epochs=args.epochs, lr_scheduler=lr_scheduler, model_path=args.save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Super-Resolution using MAE encoder")
    parser.add_argument("--mae_weights_path", type=str, required=True, help="Path to pre-trained MAE encoder weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--save_model_path", type=str, default="SR_wt.pth", help="Path to save the best model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='sr_outputs', help='Directory to save evaluation plots and images')

    args = parser.parse_args()
    main(args)
