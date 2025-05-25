import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import train_dataset
from evaluate import accuracy_fn
from torch.utils.tensorboard import SummaryWriter
from downstream_classifier import get_classifier_model
import os
import math
import argparse


def train(model, args):
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    # Load dataloaders
    train_dataloader, val_dataloader = train_dataset(vars(args))

    device = args.device
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Select scheduler
    scheduler = None
    if args.use_scheduler:
        if args.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch)
        elif args.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler_type == "lambda":
            lr_func = lambda epoch: min((epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / 50 * math.pi) + 1))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    best_val_acc = 0
    train_loss, test_loss, train_acc, test_acc, lr_history = [], [], [], [], []

    for epoch in range(args.total_epoch):
        model.train()
        total_train_loss, total_train_acc = 0.0, 0.0
        optimizer.zero_grad()

        for step, (img, label) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.total_epoch}")):
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = accuracy_fn(logits, label)
            loss.backward()

            if (step + 1) % args.steps_per_update == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()
            total_train_acc += acc.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(train_dataloader)

        model.eval()
        total_val_loss, total_val_acc = 0.0, 0.0

        with torch.no_grad():
            for img, label in val_dataloader:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = accuracy_fn(logits, label)
                total_val_loss += loss.item()
                total_val_acc += acc.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_acc = total_val_acc / len(val_dataloader)

        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            lr_history.append(lr)
            writer.add_scalar("Learning Rate", lr, epoch)

        tqdm.write(f"Epoch {epoch+1}/{args.total_epoch} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.3f} | "
                   f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.3f}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", avg_val_acc, epoch)

        train_loss.append(avg_train_loss)
        test_loss.append(avg_val_loss)
        train_acc.append(avg_train_acc)
        test_acc.append(avg_val_acc)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f"Saving best model with acc {best_val_acc:.4f} at {epoch+1} epoch!")
            torch.save(model.state_dict(), args.model_weight_path)

    writer.close()
    return train_loss, test_loss, train_acc, test_acc, lr_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ViT classifier model with fine-tuning options")
    parser.add_argument("--strategy", type=str, choices=["frozen", "partial", "full"], required=True,
                        help="Finetuning strategy: 'frozen', 'partial', or 'full'")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to pre-trained MAE weight file (.pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of output classes")
    parser.add_argument("--tensorboard_log_dir", type=str, default="runs/exp",
                        help="TensorBoard log directory")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay")
    parser.add_argument("--steps_per_update", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--scheduler_type", type=str, choices=["cosine", "step", "lambda"],
                        help="Scheduler type")
    parser.add_argument("--total_epoch", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR")
    parser.add_argument("--model_weight_path", type=str, default="best_model.pth",
                        help="Path to save best model weights")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--downstream", type=str, default="classifier", help="Downstream task")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")

    args = parser.parse_args()

    model = get_classifier_model(strategy=args.strategy,
                                 weight_path=args.weight_path,
                                 device=args.device,
                                 num_classes=args.num_classes)

    train(model, args)