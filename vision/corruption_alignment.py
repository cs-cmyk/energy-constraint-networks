"""
Corruption-to-Deepfake Alignment Diagnostic

Tests whether our synthetic corruptions target the same structural properties
that actual deepfakes violate.

For each deepfake method, this script:
1. Computes per-patch embedding differences between real and fake images
2. Computes per-patch embedding differences between real and each corruption type
3. Measures the correlation between these difference patterns

If a corruption's embedding signature correlates with a deepfake method's
signature, that corruption targets the right structural property.

Usage:
  python corruption_alignment.py \
    --real_dir ff_c23_faces/original \
    --fake_dirs ff_c23_faces/NeuralTextures ff_c23_faces/Deepfakes \
    --num_images 50

Output: correlation matrix showing which corruptions align with which methods.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict


def compute_patch_signatures(encoder, real_paths, fake_paths, num_images=50):
    """
    Compute the per-patch embedding difference signature for a deepfake method.

    For matched pairs (same video, real vs fake), compute:
      signature[patch_i] = mean(||E(fake)_i - E(real)_i||)

    This tells us WHERE in the image the deepfake creates embedding changes.

    Returns: (256,) mean per-patch distance
    """
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    patch_dists = []

    for i in range(min(num_images, len(real_paths), len(fake_paths))):
        try:
            real_img = transform(Image.open(real_paths[i]).convert("RGB"))
            fake_img = transform(Image.open(fake_paths[i]).convert("RGB"))

            with torch.no_grad():
                real_patches = encoder.encode_image(real_img)  # (256, 768)
                fake_patches = encoder.encode_image(fake_img)

            # Per-patch L2 distance
            dist = (real_patches - fake_patches).norm(dim=1).cpu().numpy()  # (256,)
            patch_dists.append(dist)
        except Exception:
            continue

    if not patch_dists:
        return np.zeros(256)

    return np.mean(patch_dists, axis=0)  # (256,) mean signature


def compute_corruption_signatures(encoder, real_paths, corruption_types,
                                    corruption_fn, num_images=50):
    """
    Compute per-patch embedding difference signatures for each corruption type.

    Returns: dict of corruption_type → (256,) signature
    """
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    signatures = {}

    for ctype in corruption_types:
        patch_dists = []
        print(f"    Computing signature for {ctype}...")

        for i in range(min(num_images, len(real_paths))):
            try:
                img = Image.open(real_paths[i]).convert("RGB")
                img_tensor = transform(img)

                corrupted = corruption_fn(img_tensor, ctype, idx=i, k=0)

                with torch.no_grad():
                    real_patches = encoder.encode_image(img_tensor)
                    corr_patches = encoder.encode_image(corrupted)

                dist = (real_patches - corr_patches).norm(dim=1).cpu().numpy()
                patch_dists.append(dist)
            except Exception:
                continue

        if patch_dists:
            signatures[ctype] = np.mean(patch_dists, axis=0)
        else:
            signatures[ctype] = np.zeros(256)

    return signatures


def compute_frequency_signature(real_paths, fake_paths, num_images=50):
    """
    Compare frequency spectra of real vs fake images.
    This is a direct diagnostic for GAN artefacts.

    Returns: mean power spectrum difference across images.
    """
    import cv2

    diffs = []
    for i in range(min(num_images, len(real_paths), len(fake_paths))):
        try:
            real = cv2.imread(str(real_paths[i]), cv2.IMREAD_GRAYSCALE)
            fake = cv2.imread(str(fake_paths[i]), cv2.IMREAD_GRAYSCALE)

            if real is None or fake is None:
                continue

            # Resize to same size
            real = cv2.resize(real, (224, 224)).astype(np.float32)
            fake = cv2.resize(fake, (224, 224)).astype(np.float32)

            # 2D FFT → power spectrum
            real_fft = np.fft.fft2(real)
            fake_fft = np.fft.fft2(fake)
            real_power = np.abs(np.fft.fftshift(real_fft))
            fake_power = np.abs(np.fft.fftshift(fake_fft))

            # Azimuthal average (radial power spectrum)
            h, w = real_power.shape
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            r = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
            max_r = min(cy, cx)

            real_radial = np.zeros(max_r)
            fake_radial = np.zeros(max_r)
            for ri in range(max_r):
                ring = r == ri
                if ring.sum() > 0:
                    real_radial[ri] = real_power[ring].mean()
                    fake_radial[ri] = fake_power[ring].mean()

            # Normalize
            real_radial = real_radial / (real_radial.sum() + 1e-8)
            fake_radial = fake_radial / (fake_radial.sum() + 1e-8)

            diffs.append(fake_radial - real_radial)
        except Exception:
            continue

    if not diffs:
        return None

    mean_diff = np.mean(diffs, axis=0)
    return mean_diff


def main():
    parser = argparse.ArgumentParser(
        description="Corruption-to-deepfake alignment diagnostic")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--num_images", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from vision_encoder import DINOv2PatchEncoder
    from vision_corruptions import apply_corruption, set_donor_pool

    encoder = DINOv2PatchEncoder("dinov2_vitb14", device)

    # Collect paths
    exts = {".jpg", ".jpeg", ".png"}
    real_paths = sorted(
        p for p in Path(args.real_dir).rglob("*") if p.suffix.lower() in exts)
    print(f"\nReal images: {len(real_paths)}")

    set_donor_pool(real_paths)

    corruption_types = ["texture_splice", "lighting_shift", "colour_temperature",
                        "frequency_injection", "symmetry_break",
                        "material_inconsistency"]

    # Compute corruption signatures
    print(f"\nComputing corruption embedding signatures...")
    corr_sigs = compute_corruption_signatures(
        encoder, real_paths, corruption_types, apply_corruption,
        num_images=args.num_images)

    # Compute deepfake signatures + frequency analysis
    fake_sigs = {}
    freq_results = {}

    for fake_dir in args.fake_dirs:
        fake_dir = Path(fake_dir)
        method = fake_dir.name
        fake_paths = sorted(
            p for p in fake_dir.rglob("*") if p.suffix.lower() in exts)

        if not fake_paths:
            print(f"  {method}: no images found")
            continue

        print(f"\n  Computing signature for {method} ({len(fake_paths)} images)...")
        fake_sigs[method] = compute_patch_signatures(
            encoder, real_paths, fake_paths, num_images=args.num_images)

        # Frequency analysis
        print(f"  Computing frequency spectrum for {method}...")
        freq_diff = compute_frequency_signature(
            real_paths, fake_paths, num_images=args.num_images)
        if freq_diff is not None:
            freq_results[method] = freq_diff

    # =============================================
    # Correlation analysis
    # =============================================
    print(f"\n{'='*80}")
    print("CORRUPTION-TO-DEEPFAKE ALIGNMENT")
    print(f"{'='*80}")
    print(f"\nPearson correlation between per-patch embedding signatures:")
    print(f"(Higher = corruption targets same spatial regions as the deepfake)")
    print()

    # Header
    methods = list(fake_sigs.keys())
    header = f"{'Corruption':<25s}" + "".join(f"{m:>15s}" for m in methods)
    print(header)
    print("-" * len(header))

    alignment_scores = defaultdict(dict)

    for ctype in corruption_types:
        row = f"{ctype:<25s}"
        for method in methods:
            # Pearson correlation between corruption signature and deepfake signature
            c_sig = corr_sigs[ctype]
            f_sig = fake_sigs[method]

            # Normalize
            c_norm = (c_sig - c_sig.mean()) / (c_sig.std() + 1e-8)
            f_norm = (f_sig - f_sig.mean()) / (f_sig.std() + 1e-8)

            corr = np.mean(c_norm * f_norm)
            alignment_scores[method][ctype] = corr
            row += f"{corr:>15.3f}"

        print(row)

    # Best corruption for each method
    print(f"\n{'='*80}")
    print("BEST CORRUPTION MATCH PER DEEPFAKE METHOD")
    print(f"{'='*80}")
    for method in methods:
        scores = alignment_scores[method]
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        print(f"  {method:<20s} → {best_type:<25s} (r={best_score:.3f})")

    # Frequency analysis results
    if freq_results:
        print(f"\n{'='*80}")
        print("FREQUENCY SPECTRUM ANALYSIS")
        print(f"{'='*80}")
        print("(Where does the power spectrum of fakes differ from real?)")
        print("  Positive = MORE power than real, Negative = LESS power")
        print()

        for method, diff in freq_results.items():
            max_r = len(diff)
            low = diff[:max_r//4].mean()
            mid = diff[max_r//4:max_r//2].mean()
            high = diff[max_r//2:3*max_r//4].mean()
            vhigh = diff[3*max_r//4:].mean()

            print(f"  {method}:")
            print(f"    Low freq:       {low:+.6f}")
            print(f"    Mid freq:       {mid:+.6f}")
            print(f"    High freq:      {high:+.6f}")
            print(f"    Very high freq: {vhigh:+.6f}")

            if abs(high) > abs(low) * 2 or abs(vhigh) > abs(low) * 2:
                print(f"    → High-frequency anomaly detected.")
                print(f"      frequency_injection corruption should target this.")
            elif abs(low) > abs(high) * 2:
                print(f"    → Low-frequency anomaly (colour/lighting shift).")
                print(f"      colour_temperature or lighting_shift should target this.")
            else:
                print(f"    → No dominant frequency band. Artefacts may be subtle.")

    # Overall recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    for method in methods:
        scores = alignment_scores[method]
        good_matches = [ct for ct, s in scores.items() if s > 0.3]
        weak_matches = [ct for ct, s in scores.items() if 0.1 < s <= 0.3]
        no_matches = [ct for ct, s in scores.items() if s <= 0.1]

        print(f"\n  {method}:")
        if good_matches:
            print(f"    Strong alignment: {', '.join(good_matches)}")
        if weak_matches:
            print(f"    Weak alignment:   {', '.join(weak_matches)}")
        if no_matches and not good_matches:
            print(f"    No alignment with current corruptions.")
            print(f"    May need method-specific corruption design.")


if __name__ == "__main__":
    main()
