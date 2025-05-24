import argparse
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchinfo import summary

from data import train_dataset, test_dataset
from vit.mae import MAE_ViT  
from evaluate import PretrainEvaluator

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    update_steps,
    mask_ratio,
    save_path
):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        step = 0
        optimizer.zero_grad()

        for img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True):
            step += 1
            img = img.to(device)
            predicted, mask = model(img)
            loss = torch.mean((predicted - img) ** 2 * mask) / mask_ratio
            loss.backward()

            if step % update_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img in val_loader:
                img = img.to(device)
                predicted, mask = model(img)
                loss = torch.mean((predicted - img) ** 2 * mask) / mask_ratio
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        tqdm.write(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

    return train_losses, val_losses

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader = train_dataset(config=args.__dict__)
    print(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")

    # Initialize model
    model = MAE_ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        encoder_layer=args.encoder_layer,
        encoder_head=args.encoder_head,
        decoder_layer=args.decoder_layer,
        decoder_head=args.decoder_head,
        mask_ratio=args.mask_ratio
    ).to(device)

    # Print model summary
    print(summary(model, input_size=(args.batch_size, 1, args.image_size, args.image_size)))

    # Optimizer and LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / 50 * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    # Train
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        update_steps=args.steps_per_update,
        mask_ratio=args.mask_ratio,
        save_path=args.save_path
    )

    evaluator = PretrainEvaluator(config, model_path="model.pth", train_losses=train_losses, val_losses=val_losses)
    evaluator.evaluate(loss_plot_path="loss_curve.png", num_images=5)

    print("Training complete. Best model saved at:", args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE Pretraining Script")

    # Model and training arguments
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--encoder_layer', type=int, default=12)
    parser.add_argument('--encoder_head', type=int, default=3)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--decoder_head', type=int, default=3)
    parser.add_argument('--mask_ratio', type=float, default=0.75)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--steps_per_update', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="vit-mae-wt.pth")

    # Dataset and dataloader config
    parser.add_argument('--data_path', type=str, required=True, help="Path to the root directory of dataset")
    parser.add_argument('--pretrain', action="store_true", help="Enable pretraining mode")
    parser.add_argument('--downstream', type=str, default=None, choices=['classifier', 'super-resolution'])
    parser.add_argument('--pretraining_label', type=str, default="no_sub", help="Subdirectory for pretraining data")
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default="train")

    args = parser.parse_args()
    main(args)
