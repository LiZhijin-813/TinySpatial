import torch
import torch.nn as nn
from typing import Tuple


class JointPatchEmbedding(nn.Module):
    """Joint patch embedding for BUS (1ch) + SWE (3ch).

    Concatenates both modalities along channel dim to form 4-channel input,
    then projects via Conv2d to patch tokens.
    Output: (B, N, embed_dim) where N = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, embed_dim: int = 192, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels=4,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, bus: torch.Tensor, swe: torch.Tensor) -> torch.Tensor:
        # bus: (B, 1, H, W), swe: (B, 3, H, W)
        x = torch.cat([bus, swe], dim=1)  # (B, 4, H, W)
        x = self.projection(x)            # (B, embed_dim, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)


class MaskingEngine(nn.Module):
    """Generates random masks for ACM-MIM pretraining.

    Masks 75% of patch positions. The mask applies ONLY to BUS channel
    reconstruction targets — SWE tokens are always retained in the encoder.

    Returns:
        mask: (B, N) bool tensor — True = masked position
        ids_restore: (B, N) — indices to restore original ordering
    """

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, num_patches: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        len_keep = int(num_patches * (1 - self.mask_ratio))

        # Random ordering per sample
        ids_shuffle = torch.stack([
            torch.randperm(num_patches, device=device) for _ in range(batch_size)
        ])
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # First len_keep tokens are visible, rest are masked
        mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
        mask[:, :len_keep] = False

        # Reorder mask to match original patch positions
        mask = mask.gather(1, ids_restore)

        return mask, ids_restore


class LightweightDecoder(nn.Module):
    """Lightweight decoder for masked patch reconstruction.

    Two-layer Transformer + linear projection back to 16x16 pixel space.
    Only processes masked tokens (plus CLS for context).
    """

    def __init__(self, embed_dim: int = 192, decoder_dim: int = 96, depth: int = 2, num_heads: int = 6, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size

        # Project encoder output to decoder dimension
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_dim)

        # Mask token (learnable) inserted at masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Two Transformer blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder_blocks = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=decoder_dim, nhead=num_heads, dim_feedforward=decoder_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        ) for _ in range(depth)])

        self.norm = nn.LayerNorm(decoder_dim)

        # Project back to pixel space: patch_size x patch_size x 1 (BUS grayscale)
        self.pixel_proj = nn.Linear(decoder_dim, patch_size * patch_size * 1)

    def forward(
        self,
        visible_tokens: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visible_tokens: (B, N_vis, embed_dim) — unmasked encoder outputs
            mask: (B, N) bool — True = masked position
            ids_restore: (B, N) — original position indices

        Returns:
            pred_pixels: (B, N, patch_size*patch_size*1) — predicted BUS pixels for all patches
        """
        B, N_vis, _ = visible_tokens.shape
        N = mask.shape[1]

        # Project to decoder dim
        x = self.encoder_to_decoder(visible_tokens)  # (B, N_vis, decoder_dim)

        # Append mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)
        x = torch.cat([x, mask_tokens], dim=1)  # (B, N, decoder_dim)

        # Restore original patch ordering
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        # Decoder transformer blocks
        for block in self.decoder_blocks:
            # Self-attention within decoder
            x = block(x, x)

        x = self.norm(x)

        # Project to pixel space
        pred_pixels = self.pixel_proj(x)  # (B, N, patch_size*patch_size*1)

        return pred_pixels


class ReconstructionLoss(nn.Module):
    """MSE loss computed only on masked positions.

    Targets are the original BUS pixel values at masked patch locations.
    """

    def __init__(self, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size

    def forward(
        self,
        pred_pixels: torch.Tensor,
        bus_img: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_pixels: (B, N, patch_size*patch_size*1) — predicted pixels
            bus_img: (B, 1, H, W) — original BUS image
            mask: (B, N) bool — True = masked position
        """
        B, _, H, W = bus_img.shape
        p = self.patch_size

        # Convert BUS image to patch pixels: (B, N, p*p*1)
        bus_patches = bus_img.reshape(B, 1, H // p, p, W // p, p)
        bus_patches = bus_patches.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, p * p * 1)

        # MSE only at masked positions
        loss_per_patch = (pred_pixels - bus_patches) ** 2  # (B, N, p*p*1)
        loss_per_patch = loss_per_patch.mean(dim=-1)        # (B, N)

        # Average over masked positions only
        mask_float = mask.float()
        num_masked = mask_float.sum(dim=1).clamp(min=1.0)   # (B,)
        loss = (loss_per_patch * mask_float).sum(dim=1) / num_masked  # (B,)
        return loss.mean()
