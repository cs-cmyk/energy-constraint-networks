"""
Frequency Heatmap Encoder.

Instead of feeding raw pixels to a CNN, this computes a spatial map
of local frequency characteristics:

  For each 14×14 patch (matching DINOv2's grid):
    1. Compute 2D FFT of the patch
    2. Bin the power spectrum into radial frequency bands
    3. Compute local statistics (power, entropy, kurtosis)

Output: (num_bands * num_stats, 16, 16) heatmap
  → each spatial position shows the frequency profile of that patch
  → GAN-smoothed regions show as cold spots in high-frequency bands
  → manipulation boundaries show as discontinuities between patches

This converts "does this region have different texture?" from a
per-pixel question into a spatial pattern that a CNN can detect.

The heatmap can be fed to either:
  - A small trainable CNN (learns what patterns indicate fakes)
  - Directly pooled into per-patch features for the constraint network

Usage:
  encoder = FrequencyHeatmapEncoder()
  heatmap = encoder.compute_heatmap(img_tensor)  # (channels, 16, 16)
  features = encoder.encode_image(img_tensor)      # (256, freq_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyHeatmapEncoder:
    """
    Computes per-patch frequency statistics as a spatial heatmap.

    For each 14×14 patch:
      - 2D FFT → power spectrum
      - Radial binning into frequency bands (low, mid-low, mid-high, high)
      - Statistics per band: mean power, std, kurtosis
      - Cross-band ratio: high/low power (smoothing detector)

    Output: (num_features, 16, 16) heatmap
    Features per patch = 4 bands × 3 stats + 2 ratios = 14

    No trainable parameters. Pure signal processing.
    """

    def __init__(self, img_size=224, num_bands=4, device="cpu"):
        self.img_size = img_size
        self.device = device
        self.grid_size = 16
        self.patch_px = img_size // self.grid_size  # 14
        self.num_patches = self.grid_size ** 2
        self.num_bands = num_bands

        # Stats per band: mean_power, std_power, kurtosis
        self.stats_per_band = 3
        # Extra features: high/low ratio, spectral entropy
        self.extra_features = 2
        self.channels = num_bands * self.stats_per_band + self.extra_features
        self.dim = self.channels  # per-patch feature dim

        # Pre-compute radial frequency bin masks for a 14×14 FFT
        self.band_masks = self._build_band_masks()

        print(f"  Frequency heatmap encoder: {self.num_bands} bands × "
              f"{self.stats_per_band} stats + {self.extra_features} extra "
              f"= {self.channels} features per patch")

    def _build_band_masks(self):
        """
        Build radial frequency band masks for a patch_px × patch_px FFT.
        Bands divide the frequency range into equal radial zones.
        """
        h = w = self.patch_px
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        Y, X = np.ogrid[:h, :w]
        # Distance from center (after fftshift)
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        # Equal radial bands
        band_edges = np.linspace(0, max_r, self.num_bands + 1)
        masks = []
        for i in range(self.num_bands):
            mask = ((r >= band_edges[i]) & (r < band_edges[i + 1])).astype(np.float32)
            if mask.sum() == 0:
                mask[cy, cx] = 1.0  # avoid empty bands
            masks.append(torch.from_numpy(mask))

        return masks  # list of (patch_px, patch_px) tensors

    def compute_heatmap(self, img_tensor):
        """
        Compute frequency heatmap for a single image.
        Fully vectorized — batch FFT over all patches at once.

        Args:
            img_tensor: (3, H, W) in [0, 1]

        Returns: (channels, 16, 16) frequency heatmap
        """
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)

        # Convert to grayscale
        gray = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]

        if gray.shape != (self.img_size, self.img_size):
            gray = F.interpolate(
                gray.unsqueeze(0).unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False
            ).squeeze()

        pp = self.patch_px
        gs = self.grid_size

        # Extract all patches at once: (256, 14, 14)
        patches = gray[:gs * pp, :gs * pp]
        patches = patches.reshape(gs, pp, gs, pp).permute(0, 2, 1, 3)
        patches = patches.reshape(gs * gs, pp, pp)  # (256, 14, 14)

        # Batch FFT
        fft_all = torch.fft.fft2(patches)
        fft_shifted = torch.fft.fftshift(fft_all, dim=(-2, -1))
        power = torch.abs(fft_shifted) ** 2  # (256, 14, 14)

        # Apply band masks and compute stats — vectorized
        heatmap = torch.zeros(self.channels, gs * gs, device=gray.device)

        for b, mask in enumerate(self.band_masks):
            mask_d = mask.to(gray.device)
            mask_bool = mask_d > 0
            n_pixels = mask_bool.sum()

            if n_pixels == 0:
                continue

            # Extract band values for all patches: (256, n_pixels)
            band_vals = power[:, mask_bool]

            mean_p = band_vals.mean(dim=1)                    # (256,)
            std_p = band_vals.std(dim=1) if n_pixels > 1 else torch.zeros(gs * gs, device=gray.device)

            # Kurtosis
            centered = band_vals - mean_p.unsqueeze(1)
            var = centered.pow(2).mean(dim=1).clamp(min=1e-16)
            kurt = centered.pow(4).mean(dim=1) / var.pow(2) - 3.0

            idx = b * self.stats_per_band
            heatmap[idx] = mean_p
            heatmap[idx + 1] = std_p
            heatmap[idx + 2] = kurt

        # High/low ratio
        low_power = heatmap[0].clamp(min=1e-8)   # band 0 mean power
        high_power = heatmap[(self.num_bands - 1) * self.stats_per_band]  # last band mean
        heatmap[-2] = high_power / low_power

        # Spectral entropy
        band_means = torch.stack([
            heatmap[b * self.stats_per_band] for b in range(self.num_bands)
        ])  # (num_bands, 256)
        total = band_means.sum(dim=0).clamp(min=1e-8)
        probs = band_means / total
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=0)
        heatmap[-1] = entropy

        return heatmap.reshape(self.channels, gs, gs)

    def compute_heatmap_batch(self, img_batch):
        """
        Compute frequency heatmaps for a batch. Vectorized per image.

        Args:
            img_batch: (B, 3, H, W) in [0, 1]

        Returns: (B, channels, 16, 16)
        """
        B = img_batch.shape[0]
        heatmaps = torch.zeros(B, self.channels, self.grid_size, self.grid_size)
        for i in range(B):
            heatmaps[i] = self.compute_heatmap(img_batch[i])
        return heatmaps

    def encode_image(self, img_tensor):
        """
        Encode single image as per-patch frequency features.
        Returns: (256, freq_dim)
        """
        heatmap = self.compute_heatmap(img_tensor)  # (channels, 16, 16)
        # Reshape to (256, channels)
        features = heatmap.reshape(self.channels, self.num_patches).T
        return features

    def encode_batch(self, img_batch):
        """
        Encode batch as per-patch frequency features.
        Returns: (B, 256, freq_dim)
        """
        heatmaps = self.compute_heatmap_batch(img_batch)  # (B, channels, 16, 16)
        B = heatmaps.shape[0]
        features = heatmaps.reshape(B, self.channels, self.num_patches)
        features = features.permute(0, 2, 1)  # (B, 256, channels)
        return features


class HeatmapCNN(nn.Module):
    """
    Small CNN that processes frequency heatmaps.

    Takes (B, channels, 16, 16) frequency heatmap and learns
    spatial patterns that indicate manipulation.

    Since the input is already 16×16, this is a very small network
    that looks for spatial discontinuities in the frequency profile.
    """

    def __init__(self, in_channels=14, out_dim=32):
        super().__init__()
        self.out_dim = out_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
        )

        params = sum(p.numel() for p in self.parameters())
        print(f"  Heatmap CNN: {params:,} params, "
              f"{in_channels} input channels → {out_dim} output dim")

    def forward(self, heatmap):
        """
        Args:
            heatmap: (B, in_channels, 16, 16)

        Returns: (B, 256, out_dim) per-patch features
        """
        features = self.conv(heatmap)  # (B, out_dim, 16, 16)
        B = features.shape[0]
        # Reshape to per-patch: (B, 256, out_dim)
        features = features.reshape(B, self.out_dim, 256).permute(0, 2, 1)
        return features


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test heatmap encoder
    print("Testing frequency heatmap encoder...")
    encoder = FrequencyHeatmapEncoder(224, device=device)

    img = torch.rand(3, 224, 224)
    heatmap = encoder.compute_heatmap(img)
    print(f"  Input: {img.shape}")
    print(f"  Heatmap: {heatmap.shape}")  # (14, 16, 16)
    print(f"  Channels: {encoder.channels}")

    # Test smoothing detection
    smooth = F.avg_pool2d(img.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
    h_orig = encoder.compute_heatmap(img)
    h_smooth = encoder.compute_heatmap(smooth)

    # High/low ratio channel (second to last)
    ratio_orig = h_orig[-2]   # (16, 16)
    ratio_smooth = h_smooth[-2]

    print(f"\n  High/low frequency ratio:")
    print(f"    Original: mean={ratio_orig.mean():.6f}")
    print(f"    Smoothed: mean={ratio_smooth.mean():.6f}")
    print(f"    Difference: {(ratio_orig - ratio_smooth).mean():+.6f}")
    print(f"    (Negative = smoothed has less high-freq relative to low-freq)")

    # Test partial smoothing (simulate GAN-rendered region)
    partial = img.clone()
    partial[:, 80:144, 80:144] = F.avg_pool2d(
        img[:, 80:144, 80:144].unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
    h_partial = encoder.compute_heatmap(partial)

    ratio_diff = h_partial[-2] - h_orig[-2]
    print(f"\n  Partial smoothing (center region):")
    print(f"    Affected patches (rows 5-10, cols 5-10):")
    print(f"      Mean ratio diff: {ratio_diff[5:11, 5:11].mean():+.6f}")
    print(f"    Unaffected patches:")
    print(f"      Mean ratio diff: {ratio_diff[:5, :].mean():+.6f}")
    print(f"    (Affected should be more negative = spatial discontinuity)")

    # Test per-patch encoding
    features = encoder.encode_image(img)
    print(f"\n  Per-patch features: {features.shape}")  # (256, 14)

    # Test batch
    batch = torch.rand(4, 3, 224, 224)
    batch_features = encoder.encode_batch(batch)
    print(f"  Batch features: {batch_features.shape}")  # (4, 256, 14)

    # Test heatmap CNN
    print("\n\nTesting heatmap CNN...")
    cnn = HeatmapCNN(in_channels=encoder.channels, out_dim=32)
    heatmap_batch = encoder.compute_heatmap_batch(batch)
    cnn_out = cnn(heatmap_batch)
    print(f"  Heatmap input: {heatmap_batch.shape}")
    print(f"  CNN output: {cnn_out.shape}")  # (4, 256, 32)
