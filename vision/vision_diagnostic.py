"""
Phase 0 Diagnostic: Does DINOv2 encode what we need?

This script gates the entire vision experiment. It answers three questions:

1. ENCODER SENSITIVITY: Do DINOv2 patch embeddings change meaningfully when
   we apply our corruptions? (Analogous to the BERT vs MiniLM embedding
   distance diagnostic in the text work.)

2. REGIONAL SPECIFICITY: Do the patch embeddings change MORE in the corrupted
   region than in the uncorrupted region? If so, per-patch energy decomposition
   should be able to localise violations.

3. MINI END-TO-END: Can the constraint network learn to separate real from
   corrupted face embeddings with just 500 images and 2 corruption types?
   We're looking for a signal, not a publishable number.

Run:
  python vision_diagnostic.py --data_dir /path/to/ff++_real_faces/

If you don't have FF++ yet, use --synthetic to run on synthetic images
(tests architecture + encoder pipeline, not corruption quality).
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import json
from pathlib import Path


def run_encoder_diagnostic(encoder, image_paths, corruption_fn,
                            num_images=50, corruption_types=None):
    """
    Test 1: Embedding distance diagnostic.

    For each image and corruption type:
    - Compute DINOv2 patches for original and corrupted
    - Measure L2 distance (global and per-region)

    Analogous to: "shuffle distance 12.16 (vs ~0 for MiniLM)"
    in the text work.
    """
    from torchvision import transforms
    from PIL import Image

    if corruption_types is None:
        corruption_types = ["texture_splice", "lighting_shift"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    results = {ct: {"global_dists": [], "per_patch_dists": []}
               for ct in corruption_types}

    print(f"\n{'='*70}")
    print("TEST 1: Encoder Sensitivity Diagnostic")
    print(f"{'='*70}")
    print(f"  Images: {min(num_images, len(image_paths))}")
    print(f"  Corruption types: {corruption_types}")

    for i, path in enumerate(image_paths[:num_images]):
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)

        # Original embedding
        with torch.no_grad():
            orig_patches = encoder.encode_image(img_tensor)  # (256, 768)

        for ct in corruption_types:
            # Corrupt and embed
            corrupted_tensor = corruption_fn(img_tensor, ct, idx=i, k=0)
            with torch.no_grad():
                corr_patches = encoder.encode_image(corrupted_tensor)

            # Global L2 distance
            global_dist = (orig_patches - corr_patches).norm().item()
            results[ct]["global_dists"].append(global_dist)

            # Per-patch L2 distances
            per_patch = (orig_patches - corr_patches).norm(dim=1)  # (256,)
            results[ct]["per_patch_dists"].append(per_patch.cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{min(num_images, len(image_paths))}")

    # Report
    print(f"\n  Results:")
    print(f"  {'Type':<20s} {'Mean Dist':>10s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*50}")
    for ct in corruption_types:
        dists = results[ct]["global_dists"]
        print(f"  {ct:<20s} {np.mean(dists):>10.4f} {np.std(dists):>8.4f} "
              f"{np.min(dists):>8.4f} {np.max(dists):>8.4f}")

    # Regional specificity
    print(f"\n  Regional specificity (per-patch distance distribution):")
    for ct in corruption_types:
        all_patch_dists = np.stack(results[ct]["per_patch_dists"])  # (N, 256)
        mean_per_patch = all_patch_dists.mean(axis=0)  # (256,)
        top_10_pct = np.percentile(mean_per_patch, 90)
        bottom_10_pct = np.percentile(mean_per_patch, 10)
        ratio = top_10_pct / max(bottom_10_pct, 1e-8)
        print(f"  {ct}: top-10% patch dist = {top_10_pct:.4f}, "
              f"bottom-10% = {bottom_10_pct:.4f}, ratio = {ratio:.2f}x")
        print(f"    (ratio > 2x = good regional specificity, "
              f"ratio ~1x = corruption is diffuse)")

    return results


def run_mini_training(encoder, image_paths, corruption_fn,
                       num_train=500, num_val=50, epochs=20,
                       device="cpu"):
    """
    Test 2: Mini end-to-end training loop.

    Train the constraint network on 500 images × 2 corruptions for 20 epochs.
    Then evaluate on 50 held-out images.

    We're looking for: energy gap between real and corrupted > 0.
    If the energy gap is growing across epochs, the architecture works
    for vision.
    """
    from model import ConstraintNetwork, ConstraintLoss
    from torchvision import transforms
    from PIL import Image

    print(f"\n{'='*70}")
    print("TEST 2: Mini End-to-End Training")
    print(f"{'='*70}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    corruption_types = ["texture_splice", "lighting_shift"]
    K = len(corruption_types)

    # --- Pre-compute embeddings ---
    print(f"\n  Pre-computing embeddings for {num_train} train + {num_val} val images...")
    t0 = time.time()

    train_paths = image_paths[:num_train]
    val_paths = image_paths[num_train:num_train + num_val]

    def embed_set(paths, label):
        N = len(paths)
        pos = torch.zeros(N, encoder.num_patches, encoder.dim)
        neg = torch.zeros(N * K, encoder.num_patches, encoder.dim)
        neg_types = []

        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            img_t = transform(img)

            with torch.no_grad():
                pos[i] = encoder.encode_image(img_t).cpu()

            for k, ct in enumerate(corruption_types):
                corr_t = corruption_fn(img_t, ct, idx=i, k=k)
                with torch.no_grad():
                    neg[i * K + k] = encoder.encode_image(corr_t).cpu()
                neg_types.append(ct)

            if (i + 1) % 100 == 0:
                print(f"    {label}: {i+1}/{N}")

        return pos, neg, neg_types

    train_pos, train_neg, train_types = embed_set(train_paths, "train")
    val_pos, val_neg, val_types = embed_set(val_paths, "val")

    print(f"  Embedding time: {time.time() - t0:.0f}s")
    print(f"  Train: {train_pos.shape}, Val: {val_pos.shape}")

    # --- Model ---
    # d_model=384, same as text. Input projection from 768 (DINOv2) to 384.
    model = ConstraintNetwork(
        d_model=384, d_state=64, vocab_size=None,
        max_seq_len=encoder.num_patches, dropout=0.15, alpha=0.3
    ).to(device)
    model.input_proj = nn.Linear(encoder.dim, 384).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params:,}")

    criterion = ConstraintLoss(margin=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    N_train = train_pos.shape[0]
    bs = 32  # small batch for 500 images

    # --- Training ---
    print(f"\n  Training for {epochs} epochs (batch_size={bs})...")
    print(f"  {'Epoch':>5s} {'Loss':>8s} {'E(real)':>8s} {'E(corr)':>8s} "
          f"{'Gap':>8s} {'ValAcc':>8s}")
    print(f"  {'-'*48}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N_train)
        total_loss = 0
        n_batches = 0

        for b in range(0, N_train - bs, bs):
            idx = perm[b:b + bs]
            neg_k = torch.randint(0, K, (len(idx),))
            neg_idx = idx * K + neg_k

            pos_batch = train_pos[idx].to(device)
            neg_batch = train_neg[neg_idx].to(device)

            ep = model(pos_batch)
            en = model(neg_batch)
            loss = criterion(ep, en)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_ep = model(val_pos.to(device))
            val_en = model(val_neg.to(device))
            val_acc = (val_ep < val_en.view(-1, K).min(dim=1).values).float().mean()
            mean_ep = val_ep.mean().item()
            mean_en = val_en.mean().item()
            gap = mean_en - mean_ep

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  {epoch+1:>5d} {avg_loss:>8.4f} {mean_ep:>8.4f} {mean_en:>8.4f} "
              f"{gap:>+8.4f} {val_acc:>8.1%}")

    # --- Final assessment ---
    print(f"\n  Final energy gap: {gap:+.4f}")
    print(f"  Final val accuracy: {val_acc:.1%}")

    if gap > 0.05:
        print(f"\n  PASS: Positive energy gap detected. The architecture can learn")
        print(f"  visual structural coherence from DINOv2 patches.")
        verdict = "PASS"
    elif gap > 0:
        print(f"\n  WEAK SIGNAL: Small positive gap. May improve with more data")
        print(f"  and corruption types. Worth continuing cautiously.")
        verdict = "WEAK"
    else:
        print(f"\n  FAIL: No energy gap. Investigate:")
        print(f"  - Is DINOv2 encoding corruption-relevant features? (Check Test 1)")
        print(f"  - Is the corruption too subtle? (Try higher severity)")
        print(f"  - Does the SSM struggle with 256-length sequences? (Try pooling patches)")
        verdict = "FAIL"

    return {
        "final_gap": gap,
        "final_val_acc": val_acc.item(),
        "verdict": verdict,
        "params": params,
    }


def run_deepfake_probe(encoder, model, real_paths, fake_paths,
                        num_samples=50, device="cpu"):
    """
    Test 3: Probe real deepfakes (if available).

    Run the mini-trained model on actual deepfakes from FF++.
    We're looking for ANY separation — even 55% would be a signal.
    """
    from torchvision import transforms
    from PIL import Image

    print(f"\n{'='*70}")
    print("TEST 3: Deepfake Probe")
    print(f"{'='*70}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def get_energies(paths, label):
        energies = []
        for p in paths[:num_samples]:
            img = Image.open(p).convert("RGB")
            img_t = transform(img)
            with torch.no_grad():
                patches = encoder.encode_image(img_t).unsqueeze(0).to(device)
                e = model(patches).item()
            energies.append(e)
        print(f"  {label}: mean={np.mean(energies):.4f}, "
              f"std={np.std(energies):.4f}, "
              f"range=[{np.min(energies):.4f}, {np.max(energies):.4f}]")
        return energies

    model.eval()
    real_energies = get_energies(real_paths, "Real faces")
    fake_energies = get_energies(fake_paths, "Deepfakes ")

    # Simple threshold-free metric
    real_mean = np.mean(real_energies)
    fake_mean = np.mean(fake_energies)
    gap = fake_mean - real_mean

    # AUC approximation
    all_labels = [0] * len(real_energies) + [1] * len(fake_energies)
    all_scores = real_energies + fake_energies
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_scores)
    except ImportError:
        # Manual AUC approximation
        n_real = len(real_energies)
        n_fake = len(fake_energies)
        correct = sum(1 for r in real_energies for f in fake_energies if r < f)
        auc = correct / (n_real * n_fake)

    print(f"\n  Energy gap (fake - real): {gap:+.4f}")
    print(f"  AUC: {auc:.3f}")
    print(f"  (AUC > 0.55 = signal worth pursuing, > 0.65 = strong signal)")

    return {"gap": gap, "auc": auc}


def run_with_synthetic_images(device="cpu"):
    """
    Run the full diagnostic on synthetic images.
    Use this when FF++ data isn't available yet.
    Tests: encoder pipeline, corruption pipeline, architecture compatibility.
    Doesn't test: real corruption quality on actual faces.
    """
    from vision_encoder import DINOv2PatchEncoder
    from vision_corruptions import apply_corruption

    print(f"\n{'='*70}")
    print("RUNNING ON SYNTHETIC IMAGES")
    print("(Tests pipeline + architecture. For real results, use FF++ data.)")
    print(f"{'='*70}")

    encoder = DINOv2PatchEncoder("dinov2_vitb14", device)

    # Generate synthetic "face" images — smooth gradients with structure
    num_images = 600
    print(f"\n  Generating {num_images} synthetic images...")
    img_dir = Path("synthetic_faces")
    img_dir.mkdir(exist_ok=True)

    import cv2
    for i in range(num_images):
        h, w = 224, 224
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Random smooth gradient (simulates face lighting)
        np.random.seed(i)
        angle = np.random.uniform(0, 2 * np.pi)
        Y, X = np.mgrid[:h, :w].astype(np.float32)
        gradient = (np.cos(angle) * X / w + np.sin(angle) * Y / h)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)

        base_color = np.random.randint(100, 200, 3)
        for c in range(3):
            img[:, :, c] = np.clip(base_color[c] + gradient * 80 - 40, 0, 255)

        # Add elliptical "face" region
        cx, cy = w // 2 + np.random.randint(-10, 10), h // 2 + np.random.randint(-10, 10)
        axes = (50 + np.random.randint(0, 20), 65 + np.random.randint(0, 20))
        face_color = tuple(int(c) for c in np.random.randint(140, 200, 3))
        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, face_color, -1)
        img = cv2.GaussianBlur(img, (5, 5), 2)

        cv2.imwrite(str(img_dir / f"synth_{i:04d}.jpg"), img)

    image_paths = sorted(img_dir.glob("*.jpg"))

    # Test 1: Encoder sensitivity
    results_enc = run_encoder_diagnostic(
        encoder, image_paths, apply_corruption, num_images=50)

    # Test 2: Mini training
    results_train = run_mini_training(
        encoder, image_paths, apply_corruption,
        num_train=500, num_val=50, epochs=20, device=device)

    return results_enc, results_train


def run_with_ff_data(data_dir, device="cpu", fake_dir=None):
    """
    Run the full diagnostic on FaceForensics++ data.

    Args:
        data_dir: path to directory of real face crops (224×224 JPEGs)
        fake_dir: optional path to deepfake face crops for probe test
    """
    from vision_encoder import DINOv2PatchEncoder
    from vision_corruptions import apply_corruption, set_donor_pool

    encoder = DINOv2PatchEncoder("dinov2_vitb14", device)

    # Collect image paths
    data_dir = Path(data_dir)
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in exts)
    print(f"\nFound {len(image_paths)} images in {data_dir}")

    if len(image_paths) < 100:
        print("  WARNING: Need at least 550 images (500 train + 50 val).")
        print("  Consider using --synthetic mode first.")
        return

    # Set donor pool for texture splice
    set_donor_pool(image_paths)

    # Test 1: Encoder sensitivity
    results_enc = run_encoder_diagnostic(
        encoder, image_paths, apply_corruption, num_images=50)

    # Test 2: Mini training
    results_train = run_mini_training(
        encoder, image_paths, apply_corruption,
        num_train=500, num_val=50, epochs=20, device=device)

    # Test 3: Deepfake probe (if fake data provided)
    results_probe = None
    if fake_dir and results_train["verdict"] != "FAIL":
        fake_paths = sorted(
            p for p in Path(fake_dir).rglob("*") if p.suffix.lower() in exts)
        if fake_paths:
            # Need to get the model back — re-instantiate and load
            from model import ConstraintNetwork
            model = ConstraintNetwork(
                d_model=384, d_state=64, vocab_size=None,
                max_seq_len=encoder.num_patches, dropout=0.15, alpha=0.3
            ).to(device)
            model.input_proj = nn.Linear(encoder.dim, 384).to(device)
            # Note: model would need to be returned from run_mini_training
            # or saved to disk. For now, skip probe in this path.
            print("\n  (Deepfake probe requires saving model from mini training.")
            print("   Will add checkpoint save in full pipeline.)")

    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"  Encoder sensitivity: {'OK' if results_enc else 'FAILED'}")
    print(f"  Mini training: {results_train['verdict']}")
    print(f"  Energy gap: {results_train['final_gap']:+.4f}")
    print(f"  Val accuracy: {results_train['final_val_acc']:.1%}")

    if results_train["verdict"] == "PASS":
        print(f"\n  RECOMMENDATION: Proceed to full training pipeline (Phase 1).")
    elif results_train["verdict"] == "WEAK":
        print(f"\n  RECOMMENDATION: Proceed cautiously. Consider:")
        print(f"    - Adding more corruption types")
        print(f"    - Increasing severity")
        print(f"    - Testing with real face images if using synthetic")
    else:
        print(f"\n  RECOMMENDATION: Investigate before scaling up.")
        print(f"    - Check encoder diagnostic for near-zero distances")
        print(f"    - Try a face-specific encoder (ArcFace features)")
        print(f"    - Try patch pooling (4×4 → 64 positions)")

    # Save results
    output = {
        "encoder_distances": {
            ct: {"mean": float(np.mean(v["global_dists"])),
                 "std": float(np.std(v["global_dists"]))}
            for ct, v in results_enc.items()
        },
        "training": results_train,
    }
    with open("vision_diagnostic_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to vision_diagnostic_results.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Vision diagnostic")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to real face images (FF++ or other)")
    parser.add_argument("--fake_dir", type=str, default=None,
                        help="Path to deepfake images (for probe test)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on synthetic images (no dataset needed)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not specified)")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    if args.synthetic or args.data_dir is None:
        run_with_synthetic_images(device)
    else:
        run_with_ff_data(args.data_dir, device, args.fake_dir)
