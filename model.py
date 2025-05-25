import torch
import numpy as np
import torch
from einops import rearrange, repeat
from timm.models.vision_transformer import Block
from torch.nn.init import trunc_normal_

# Helper function: Generate random forward and backward indices for shuffling patches
def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

# Helper function: Gather patches by index across a batch
def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

# Module for randomly shuffling and dropping image patches
class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio):
        """
        Args:
            ratio (float): Ratio of patches to be masked.
        """
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))  # Number of visible patches after masking

        # Generate random indexes for each sample in the batch
        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        # Shuffle and keep only visible patches
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]
        return patches, forward_indexes, backward_indexes

# MAE Encoder
class MAE_Encoder(torch.nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, num_layer, num_head, mask_ratio):
        """
        Encoder module for MAE.
        - Converts image to patches
        - Randomly masks them
        - Runs a ViT encoder on visible patches

        Args:
            image_size (int): Size of input image (assumes square).
            patch_size (int): Size of each patch.
            emb_dim (int): Embedding dimension.
            num_layer (int): Number of transformer layers.
            num_head (int): Number of attention heads.
            mask_ratio (float): Fraction of patches to be masked.
        """
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        # Patchify: Convert image into non-overlapping patches
        self.patchify = torch.nn.Conv2d(1, emb_dim, patch_size, patch_size)

        # Vision Transformer Encoder using timm
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        # Convert image to patch embeddings
        patches = self.patchify(img)  # (B, C, H, W)
        patches = rearrange(patches, 'b c h w -> (h w) b c')  # (T, B, C)

        # Add positional embedding
        patches = patches + self.pos_embedding

        # Shuffle and mask patches
        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        # Prepend [CLS] token
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)

        # ViT expects (B, T, C)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.transformer(patches)

        features = self.layer_norm(features)
        features = rearrange(features, 'b t c -> t b c')
        return features, backward_indexes

# MAE Decoder
class MAE_Decoder(torch.nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, num_layer, num_head):
        """
        Decoder module for MAE.
        - Restores full set of tokens using mask token
        - Reconstructs image patches from the full token set
        """
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        # Transformer decoder using timm
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        # Map tokens back to pixel space
        self.head = torch.nn.Linear(emb_dim, patch_size ** 2)

        # Patch -> Image reconstruction
        self.patch2img = rearrange(
            '(h w) b (c p1 p2) -> b c (h p1) (w p2)',
            p1=patch_size, p2=patch_size,
            h=image_size // patch_size
        )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]  # T_visible
        B = features.shape[1]

        # Recover original patch positions (add 1 for cls token offset)
        backward_indexes = torch.cat([
            torch.zeros(1, B).to(backward_indexes),  # cls token index
            backward_indexes + 1
        ], dim=0)

        # Append mask tokens for masked patches
        full_T = backward_indexes.shape[0]
        features = torch.cat([
            features,
            self.mask_token.expand(full_T - T, B, -1)
        ], dim=0)

        # Reorder to original patch order
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        # Transformer decoder
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')[1:]  # Remove CLS

        # Predict pixel patches
        patches = self.head(features)

        # Reconstruct binary mask for loss computation (optional)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)

        # Convert patches back to image
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

# Full MAE-ViT Model
class MAE_ViT(torch.nn.Module):
    def __init__(
        self,
        image_size=64,
        patch_size=16,
        emb_dim=256,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75
    ):
        """
        Args:
            image_size (int): Size of input image.
            patch_size (int): Size of each square patch.
            emb_dim (int): Embedding dimension.
            encoder_layer (int): Encoder transformer layers.
            encoder_head (int): Attention heads in encoder.
            decoder_layer (int): Decoder transformer layers.
            decoder_head (int): Attention heads in decoder.
            mask_ratio (float): Ratio of image patches to mask.
        """
        super().__init__()
        self.encoder = MAE_Encoder(
            image_size, patch_size, emb_dim,
            encoder_layer, encoder_head, mask_ratio
        )
        self.decoder = MAE_Decoder(
            image_size, patch_size, emb_dim,
            decoder_layer, decoder_head
        )

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask


class MAE_Classifier(torch.nn.Module):
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

    model_cls = MAE_Classifier(model_mae.encoder, num_classes=num_classes).to(device)

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

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.act(x)

# Super-resolution model using pretrained MAE encoder
class MAE_SR(nn.Module):
    def __init__(
        self, 
        mae_encoder, 
        input_dim=768, 
        emb_dim=256, 
        patch_size=16, 
        num_layers=2, 
        num_heads=8, 
        num_upsample_blocks=3
    ):
        super().__init__() 
        self.encoder = mae_encoder
        self.cls_token = self.encoder.cls_token
        self.pos_embedding = self.encoder.pos_embedding
        self.patchify = self.encoder.patchify
        self.transformer = self.encoder.transformer
        self.layer_norm = self.encoder.layer_norm

        # Project high-dimensional encoder output to fixed channel size
        self.feature_proj = nn.Conv1d(input_dim, emb_dim, kernel_size=1)

        # Stack of upsampling blocks
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(emb_dim, emb_dim) for _ in range(num_upsample_blocks)],
            nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1),  # Output 1-channel image
            nn.Tanh()  # Normalize output between -1 and 1
        )

        # Optional resizing
        self.interpolate = lambda x: F.interpolate(x, size=(150, 150), mode='bilinear', align_corners=False)

    def forward(self, x):
        patches = self.patchify(x)
        patches = rearrange(patches, 'b c h w -> (h w) b c')        
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)        
        patches = rearrange(patches, 't b c -> b t c')

        features = self.layer_norm(self.transformer(patches))        
        encoded_features = rearrange(features, 'b t c -> b c t')[:, :, 1:]  # Remove CLS token

        encoded_features = self.feature_proj(encoded_features)  # Project to emb_dim channels
        encoded_features = encoded_features.view(-1, emb_dim, 4, 4)  # Reshape to spatial 4x4

        reconstructed_img = self.upsample_blocks(encoded_features)
        reconstructed_img = self.interpolate(reconstructed_img)

        return reconstructed_img

