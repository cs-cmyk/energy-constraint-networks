"""
DINOv2 patch-level encoder for the constraint network (vision extension).

Analogous to bert_encoder.py:
- BERT: paragraph → tokenize → per-token hidden states → pool into windows
- DINOv2: image → patchify → per-patch hidden states (no pooling needed)

The constraint network receives the same shape: (batch, num_positions, dim)
But now each position is a spatial image patch, not a text window.

DINOv2 ViT-B/14 produces 16×16 = 256 patch tokens at 768 dims for 224×224 input.
We add 2D sinusoidal positional encoding before flattening to 1D, so the SSM
knows spatial position (row, column), not just sequence index.

Usage:
  encoder = DINOv2PatchEncoder()
  patches = encoder.encode_image(img_tensor)  # (256, 768)
  patches = encoder.encode_image_file("face.jpg")  # convenience

Install:
  pip install torch torchvision pillow
  (DINOv2 loaded via torch.hub — no extra install needed)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class DINOv2PatchEncoder:
    """
    Encodes images into patch-level representations using frozen DINOv2.

    Process:
    1. Resize + normalize image to 224×224
    2. Run through frozen DINOv2 ViT-B/14
    3. Extract patch tokens (discard [CLS])
    4. Add 2D sinusoidal positional encoding

    Output: (num_patches, hidden_dim) — same interface as BERTWindowEncoder
    """

    def __init__(self, model_name="dinov2_vitb14", device="cpu",
                 img_size=224, add_pos_encoding=True):
        self.device = device
        self.img_size = img_size
        self.add_pos_encoding = add_pos_encoding

        # DINOv2 ViT-B/14: patch_size=14, 224/14 = 16 patches per side
        self.patch_size = 14
        self.grid_size = img_size // self.patch_size  # 16
        self.num_patches = self.grid_size ** 2  # 256

        print(f"Loading DINOv2 encoder: {model_name}")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.dim = self.model.embed_dim  # 768 for ViT-B
        print(f"  Patch dim: {self.dim}, Grid: {self.grid_size}×{self.grid_size}, "
              f"Patches: {self.num_patches}")

        # Pre-compute 2D sinusoidal positional encoding
        if add_pos_encoding:
            self.pos_encoding = self._build_2d_sinusoidal_pos(
                self.grid_size, self.dim
            ).to(device)  # (256, 768)
            print(f"  2D positional encoding: {self.pos_encoding.shape}")

        # Standard ImageNet normalization (DINOv2 was trained with this)
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    @staticmethod
    def _build_2d_sinusoidal_pos(grid_size, dim):
        """
        2D sinusoidal positional encoding.
        Encodes (row, col) position into dim-dimensional vector.
        First dim/2 channels encode row, second dim/2 encode column.
        """
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
        half_dim = dim // 2
        quarter_dim = dim // 4

        # Frequency bands
        omega = torch.arange(quarter_dim, dtype=torch.float32) / quarter_dim
        omega = 1.0 / (10000.0 ** omega)  # (quarter_dim,)

        positions = torch.arange(grid_size, dtype=torch.float32)
        # Row encoding
        row_pos = positions.unsqueeze(1) * omega.unsqueeze(0)  # (grid, quarter)
        row_enc = torch.cat([row_pos.sin(), row_pos.cos()], dim=1)  # (grid, half)
        # Column encoding
        col_pos = positions.unsqueeze(1) * omega.unsqueeze(0)
        col_enc = torch.cat([col_pos.sin(), col_pos.cos()], dim=1)

        # Combine: each (row, col) pair gets [row_enc; col_enc]
        pos_encoding = torch.zeros(grid_size, grid_size, dim)
        for r in range(grid_size):
            for c in range(grid_size):
                pos_encoding[r, c] = torch.cat([row_enc[r], col_enc[c]])

        # Flatten to (num_patches, dim) in raster order
        return pos_encoding.reshape(grid_size * grid_size, dim)

    def preprocess(self, img_tensor):
        """
        Preprocess image tensor for DINOv2.
        Input: (3, H, W) or (B, 3, H, W) in [0, 1] range
        Output: (B, 3, 224, 224) normalized
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Resize if needed
        if img_tensor.shape[-2:] != (self.img_size, self.img_size):
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False
            )

        # Normalize
        img_tensor = img_tensor.to(self.device)
        img_tensor = (img_tensor - self.pixel_mean) / self.pixel_std
        return img_tensor

    def encode_image(self, img_tensor):
        """
        Encode a single image into patch representations.

        Args:
            img_tensor: (3, H, W) or (B, 3, H, W) in [0, 1] range

        Returns: (num_patches, dim) — 256 patches × 768 dims for ViT-B/14
        """
        img = self.preprocess(img_tensor)

        with torch.no_grad():
            # DINOv2 forward — get intermediate patch tokens
            features = self.model.forward_features(img)
            patch_tokens = features["x_norm_patchtokens"]  # (B, num_patches, dim)

        patches = patch_tokens.squeeze(0)  # (num_patches, dim)

        # Add 2D positional encoding
        if self.add_pos_encoding:
            patches = patches + self.pos_encoding

        return patches

    def encode_batch(self, img_batch):
        """
        Encode a batch of images.

        Args:
            img_batch: (B, 3, H, W) in [0, 1] range

        Returns: (B, num_patches, dim)
        """
        img = self.preprocess(img_batch)

        with torch.no_grad():
            features = self.model.forward_features(img)
            patch_tokens = features["x_norm_patchtokens"]  # (B, num_patches, dim)

        if self.add_pos_encoding:
            patch_tokens = patch_tokens + self.pos_encoding.unsqueeze(0)

        return patch_tokens

    def encode_image_file(self, path):
        """
        Convenience: load image from file, resize, encode.
        Returns: (num_patches, dim)
        """
        from torchvision import transforms
        from PIL import Image

        img = Image.open(path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),  # → [0, 1]
        ])
        img_tensor = transform(img)
        return self.encode_image(img_tensor)

    def patches_to_grid(self, per_patch_values):
        """
        Reshape per-patch values back to 2D spatial grid.
        Useful for visualizing per-patch energy as a heatmap.

        Args:
            per_patch_values: (num_patches,) tensor

        Returns: (grid_size, grid_size) tensor
        """
        return per_patch_values.reshape(self.grid_size, self.grid_size)


def precompute_vision_embeddings(image_paths, encoder, device,
                                  corruption_fn=None, corruptions_per_image=5,
                                  cache_path=None, batch_size=16):
    """
    Pre-compute all DINOv2 patch embeddings for a dataset of face images.
    Same caching strategy as precompute_bert_embeddings.

    Args:
        image_paths: list of image file paths
        encoder: DINOv2PatchEncoder instance
        corruption_fn: function(img_tensor, corruption_type) → corrupted_tensor
        corruptions_per_image: number of corruption variants per image
        cache_path: where to save/load cache
        batch_size: images to encode at once

    Returns: pos_embs, neg_embs, neg_types
    """
    import os
    import time
    from torchvision import transforms
    from PIL import Image

    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        return data["pos"], data["neg"], data["types"]

    N = len(image_paths)
    num_patches = encoder.num_patches
    dim = encoder.dim

    pos_embs = torch.zeros(N, num_patches, dim)
    neg_embs = torch.zeros(N * corruptions_per_image, num_patches, dim)
    neg_types = []

    transform = transforms.Compose([
        transforms.Resize((encoder.img_size, encoder.img_size)),
        transforms.ToTensor(),
    ])

    corruption_types = ["texture_splice", "lighting_shift"]

    print(f"  Encoding {N} images + {N * corruptions_per_image} corruptions (DINOv2)...")
    t0 = time.time()

    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)  # (3, 224, 224) in [0, 1]

        # Encode original
        with torch.no_grad():
            emb = encoder.encode_image(img_tensor).cpu()  # (256, 768)
        pos_embs[idx] = emb

        # Generate and encode corruptions
        if corruption_fn is not None:
            for k in range(corruptions_per_image):
                ctype = corruption_types[k % len(corruption_types)]

                # Corruption function returns a corrupted image tensor
                corrupted = corruption_fn(img_tensor, ctype, idx=idx, k=k)

                with torch.no_grad():
                    c_emb = encoder.encode_image(corrupted).cpu()

                neg_idx = idx * corruptions_per_image + k
                neg_embs[neg_idx] = c_emb
                neg_types.append(ctype)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (N - idx - 1) / rate
            print(f"    {idx+1}/{N} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({N/elapsed:.1f} images/s)")

    if cache_path:
        print(f"  Caching to {cache_path}")
        torch.save({"pos": pos_embs, "neg": neg_embs, "types": neg_types}, cache_path)

    return pos_embs, neg_embs, neg_types


if __name__ == "__main__":
    # Quick test — encode a single image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DINOv2PatchEncoder("dinov2_vitb14", device)

    # Create a random test image (replace with real face for actual testing)
    test_img = torch.rand(3, 224, 224)
    patches = encoder.encode_image(test_img)
    print(f"\nInput: {test_img.shape}")
    print(f"Output: {patches.shape}")  # (256, 768)
    print(f"Patch norm (mean): {patches.norm(dim=1).mean():.4f}")

    # Test that spatial position matters
    # Flip left-right: same content, different spatial arrangement
    flipped = test_img.flip(dims=[2])  # horizontal flip
    patches_orig = encoder.encode_image(test_img)
    patches_flip = encoder.encode_image(flipped)
    diff = (patches_orig - patches_flip).norm().item()
    print(f"\nEmbedding distance (original vs h-flipped): {diff:.4f}")
    print(f"  (Should be nonzero — 2D pos encoding differentiates spatial position)")

    # Test batch encoding
    batch = torch.rand(4, 3, 224, 224)
    batch_patches = encoder.encode_batch(batch)
    print(f"\nBatch input: {batch.shape}")
    print(f"Batch output: {batch_patches.shape}")  # (4, 256, 768)
