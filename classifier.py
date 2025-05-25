import torch
import argparse
from mae_vit import MAE_ViT, MAE_Encoder
from einops import rearrange


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=3) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_token = self.encoder.cls_token
        self.pos_embedding = self.encoder.pos_embedding
        self.patchify = self.encoder.patchify
        self.transformer = self.encoder.transformer
        self.layer_norm = self.encoder.layer_norm
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.pos_embedding.shape[-1], 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


def get_classifier_model(strategy: str, weight_path: str, device: str, num_classes: int = 3) -> torch.nn.Module:
    """
    Create a ViT_Classifier using MAE encoder based on specified finetuning strategy.

    Args:
        strategy (str): Finetuning strategy. Options: "frozen", "partial", "full".
        weight_path (str): Path to pre-trained MAE weights.
        device (str): Torch device ("cuda" or "cpu").
        num_classes (int): Number of output classes. Default is 3.

    Returns:
        torch.nn.Module: Configured classifier model.
    """
    model_mae = MAE_ViT(
        image_size=64,
        patch_size=16,
        emb_dim=768,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75
    )
    model_mae.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

    model_cls = ViT_Classifier(model_mae.encoder, num_classes=num_classes).to(device)

    # Strategy-based finetuning
    if strategy == "frozen":
        for param in model_cls.encoder.parameters():
            param.requires_grad = False

    elif strategy == "partial":
        for param in model_cls.encoder.parameters():
            param.requires_grad = False
        for name, param in model_cls.encoder.named_parameters():
            if "norm" in name or name.startswith("layer_norm"):
                param.requires_grad = True

    elif strategy == "full":
        for param in model_cls.encoder.parameters():
            param.requires_grad = True

    else:
        raise ValueError("Invalid strategy. Choose from 'frozen', 'partial', or 'full'.")

    return model_cls



