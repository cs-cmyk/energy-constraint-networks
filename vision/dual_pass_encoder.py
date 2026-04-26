"""
Dual-Pass DINOv2 Encoder.

Two passes through the same DINOv2:
  1. DINOv2(original image)      → (256, 768) structural features
  2. DINOv2(frequency heatmap)   → (256, 768) frequency features

Both outputs are in the SAME feature space — DINOv2's native
representation. Concatenated along sequence dimension:
  → (512, 768) total

The constraint network sees 512 positions: 256 structural patches
followed by 256 frequency patches. Its SSM tracks state across
both, and its attention compares structural patches with frequency
patches using the same learned operations.

No separate branch, no incompatible representations.

Usage:
  encoder = DualPassEncoder(device="cuda")
  features = encoder.encode_image(img_tensor)  # (512, 768)
  features = encoder.encode_batch(img_batch)   # (B, 512, 768)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DualPassEncoder:
    """
    Encodes images through DINOv2 twice:
      Pass 1: original RGB image → structural/semantic features
      Pass 2: frequency heatmap rendered as RGB → frequency features

    The frequency heatmap is rendered as a 224×224 RGB image:
      R channel: high-frequency power (normalized per image)
      G channel: low-frequency power (normalized per image)
      B channel: high/low ratio (the smoothing detector)

    Both passes produce (256, 768) in DINOv2's native space.
    Concatenated: (512, 768).
    """

    def __init__(self, dinov2_model="dinov2_vitb14", device="cpu",
                 img_size=224, add_pos_encoding=True):
        from vision_encoder import DINOv2PatchEncoder

        self.device = device
        self.img_size = img_size
        self.grid_size = 16
        self.patch_px = img_size // self.grid_size  # 14

        # Single DINOv2 instance — used for both passes
        self.dinov2 = DINOv2PatchEncoder(
            dinov2_model, device, img_size, add_pos_encoding)

        # Output properties
        self.dinov2_dim = self.dinov2.dim
        self.dim = self.dinov2.dim        # 768 — same dim, double positions
        self.num_patches = 512            # 256 structural + 256 frequency
        self.num_patches_per_pass = 256

        # FFT masks for frequency computation
        self._build_fft_masks()

        print(f"  Dual-pass encoder: 2 × DINOv2 → ({self.num_patches}, {self.dim})")

    def _build_fft_masks(self):
        """Build masks for low and high frequency bands."""
        h = w = self.patch_px
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        self.low_mask = torch.from_numpy(
            (r < max_r * 0.35).astype(np.float32))
        self.high_mask = torch.from_numpy(
            (r >= max_r * 0.65).astype(np.float32))

    def _compute_freq_image(self, img_tensor):
        """
        Compute frequency heatmap and render as RGB image.

        Args:
            img_tensor: (3, 224, 224) in [0, 1]

        Returns: (3, 224, 224) frequency image in [0, 1]
        """
        # Grayscale
        gray = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]

        pp = self.patch_px
        gs = self.grid_size

        # Extract all patches: (256, 14, 14)
        patches = gray[:gs * pp, :gs * pp]
        patches = patches.reshape(gs, pp, gs, pp).permute(0, 2, 1, 3)
        patches = patches.reshape(gs * gs, pp, pp)

        # Batch FFT
        fft_all = torch.fft.fft2(patches)
        fft_shifted = torch.fft.fftshift(fft_all, dim=(-2, -1))
        power = torch.abs(fft_shifted) ** 2

        # Per-patch band powers
        low_mask = self.low_mask.to(power.device)
        high_mask = self.high_mask.to(power.device)

        low_power = (power * low_mask).sum(dim=(-2, -1)) / low_mask.sum().clamp(min=1)
        high_power = (power * high_mask).sum(dim=(-2, -1)) / high_mask.sum().clamp(min=1)
        ratio = high_power / low_power.clamp(min=1e-8)

        # Reshape to grid: (16, 16)
        low_grid = low_power.reshape(gs, gs)
        high_grid = high_power.reshape(gs, gs)
        ratio_grid = ratio.reshape(gs, gs)

        # Normalize each to [0, 1]
        def norm(x):
            mn, mx = x.min(), x.max()
            if mx > mn:
                return (x - mn) / (mx - mn)
            return torch.ones_like(x) * 0.5

        r_ch = norm(high_grid)   # R: high-freq power
        g_ch = norm(low_grid)    # G: low-freq power
        b_ch = norm(ratio_grid)  # B: high/low ratio

        # Stack to RGB: (3, 16, 16)
        freq_small = torch.stack([r_ch, g_ch, b_ch])

        # Upscale to 224×224 with bilinear interpolation
        freq_img = F.interpolate(
            freq_small.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)  # (3, 224, 224)

        return freq_img.clamp(0, 1)

    def encode_image(self, img_tensor):
        """
        Encode single image: two DINOv2 passes, concatenated.

        Args:
            img_tensor: (3, H, W) in [0, 1]

        Returns: (512, 768) — 256 structural + 256 frequency patches
        """
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)

        img_tensor = img_tensor.to(self.device)

        if img_tensor.shape[-2:] != (self.img_size, self.img_size):
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        # Pass 1: original image
        struct_features = self.dinov2.encode_image(img_tensor)  # (256, 768)

        # Compute frequency image
        freq_img = self._compute_freq_image(img_tensor)

        # Pass 2: frequency image through same DINOv2
        freq_features = self.dinov2.encode_image(freq_img)  # (256, 768)

        # Concatenate along sequence dimension
        return torch.cat([struct_features, freq_features], dim=0)  # (512, 768)

    def encode_batch(self, img_batch):
        """
        Encode batch: two DINOv2 passes, concatenated.

        Args:
            img_batch: (B, 3, H, W) in [0, 1]

        Returns: (B, 512, 768)
        """
        img_batch = img_batch.to(self.device)

        if img_batch.shape[-2:] != (self.img_size, self.img_size):
            img_batch = F.interpolate(
                img_batch,
                size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False
            )

        # Pass 1: original images
        struct_features = self.dinov2.encode_batch(img_batch)  # (B, 256, 768)

        # Compute frequency images for batch
        freq_imgs = torch.zeros_like(img_batch)
        for i in range(img_batch.shape[0]):
            freq_imgs[i] = self._compute_freq_image(img_batch[i])

        # Pass 2: frequency images
        freq_features = self.dinov2.encode_batch(freq_imgs)  # (B, 256, 768)

        # Concatenate along sequence dimension
        return torch.cat([struct_features, freq_features], dim=1)  # (B, 512, 768)

    def encode_image_file(self, path):
        from torchvision import transforms
        from PIL import Image
        img = Image.open(path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        return self.encode_image(transform(img))

    def patches_to_grid(self, per_patch_values):
        """
        Reshape per-patch values back to grids.
        First 256 = structural, second 256 = frequency.
        Returns: (structural_grid, frequency_grid) each (16, 16)
        """
        struct = per_patch_values[:256].reshape(self.grid_size, self.grid_size)
        freq = per_patch_values[256:].reshape(self.grid_size, self.grid_size)
        return struct, freq


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Testing dual-pass encoder...")
    encoder = DualPassEncoder(device=device)

    # Test single image
    img = torch.rand(3, 224, 224)
    features = encoder.encode_image(img)
    print(f"  Input: {img.shape}")
    print(f"  Output: {features.shape}")  # (512, 768)
    print(f"  Structural patches: {features[:256].shape}")
    print(f"  Frequency patches:  {features[256:].shape}")

    # Test smoothing detection
    smooth = F.avg_pool2d(img.unsqueeze(0), 7, stride=1, padding=3).squeeze(0)
    f_orig = encoder.encode_image(img)
    f_smooth = encoder.encode_image(smooth)

    # Overall distance
    total_diff = (f_orig - f_smooth).norm().item()

    # Structural vs frequency patch distances
    struct_diff = (f_orig[:256] - f_smooth[:256]).norm().item()
    freq_diff = (f_orig[256:] - f_smooth[256:]).norm().item()

    print(f"\n  Smoothing detection:")
    print(f"    Total distance:      {total_diff:.4f}")
    print(f"    Structural patches:  {struct_diff:.4f}")
    print(f"    Frequency patches:   {freq_diff:.4f}")
    print(f"    Freq/Struct ratio:   {freq_diff / max(struct_diff, 1e-8):.2f}x")

    # Batch test
    batch = torch.rand(4, 3, 224, 224)
    batch_out = encoder.encode_batch(batch)
    print(f"\n  Batch input: {batch.shape}")
    print(f"  Batch output: {batch_out.shape}")  # (4, 512, 768)

    # Visualize frequency image
    freq_img = encoder._compute_freq_image(img)
    print(f"\n  Frequency image: {freq_img.shape}, "
          f"range [{freq_img.min():.4f}, {freq_img.max():.4f}]")
