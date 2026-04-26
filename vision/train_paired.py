"""
Train constraint network on paired real/fake data from FaceForensics++.

Instead of designing synthetic corruptions and hoping they align with
deepfake artefacts, this script uses actual deepfakes as negatives:
  - Positive: real face from FF++ (should get low energy)
  - Negative: manipulated version of the same face (should get high energy)

The constraint network learns what structural violations deepfakes
introduce, directly from the data.

Key design choices:
  - Matched pairs where possible (same video, same frame) — focuses
    learning on "what changed" rather than "which person"
  - Hold out deepfake methods for generalisation testing — train on
    {Deepfakes, Face2Face, FaceSwap}, test on {NeuralTextures, FaceShifter}
  - Per-patch energy decomposition still gives violation localisation

Usage:
  # Train on 3 methods, hold out 2 for generalisation:
  python train_paired.py \
    --real_dir ff_c23_faces/original \
    --fake_dirs ff_c23_faces/Deepfakes ff_c23_faces/Face2Face ff_c23_faces/FaceSwap \
    --holdout_dirs ff_c23_faces/NeuralTextures ff_c23_faces/FaceShifter

  # Train on all methods (no holdout):
  python train_paired.py \
    --real_dir ff_c23_faces/original \
    --fake_dirs ff_c23_faces/Deepfakes ff_c23_faces/Face2Face \
                ff_c23_faces/FaceSwap ff_c23_faces/NeuralTextures \
                ff_c23_faces/FaceShifter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp
import time
import json
import random
import math
import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from model import ConstraintNetwork, ConstraintLoss


CONFIG = dict(
    # Data
    max_pairs=20000,           # max real/fake pairs to use
    val_fraction=0.1,

    # Encoder
    encoder_model="dinov2_vitb14",
    encoder_dim=768,
    num_patches=256,

    # Model
    d_model=384,
    d_state=64,
    dropout=0.15,
    alpha=0.3,

    # Training
    batch_size=128,
    lr=1e-4,
    weight_decay=0.01,
    epochs=40,
    margin=5.0,
    grad_clip=1.0,
    use_fp16=True,
    patience=10,

    # Hard negatives
    hard_neg_start_epoch=10,
    hard_neg_fraction=0.2,

    # LR schedule
    warmup_epochs=2,
)


# ============================================================
# Pairing logic
# ============================================================

def parse_frame_info(filename):
    """
    Parse video ID and frame number from filename.

    Handles FF++ naming conventions:
      '003_f0100.jpg'       → video_id='003', frame=100
      '003_178_f0100.jpg'   → video_ids=['003','178'], frame=100
      '003_f0100_face.jpg'  → video_id='003', frame=100
    """
    stem = Path(filename).stem

    # Find frame number: _f followed by digits
    frame_match = re.search(r'_f(\d+)', stem)
    frame_num = int(frame_match.group(1)) if frame_match else -1

    # Everything before _f is the video identifier
    if frame_match:
        video_part = stem[:frame_match.start()]
    else:
        video_part = stem

    # Split on underscore to get potential video IDs
    # e.g., "003_178" → ["003", "178"], "003" → ["003"]
    video_ids = [v for v in video_part.split('_') if v.isdigit()]

    return video_ids, frame_num


def build_matched_pairs(real_paths, fake_paths, max_pairs=None):
    """
    Build matched pairs of real and fake images.

    Matching strategy (in order of preference):
    1. Exact match: same video ID + same frame number
    2. Video match: same video ID, closest frame number
    3. Unpaired: random real face paired with remaining fakes

    Returns: list of (real_path, fake_path, match_type) tuples
    """
    # Index real images by video_id and frame
    real_index = defaultdict(dict)  # video_id → {frame → path}
    for p in real_paths:
        vids, frame = parse_frame_info(p.name)
        for vid in vids:
            real_index[vid][frame] = p

    pairs = []
    used_real = set()
    used_fake = set()

    # Pass 1: exact matches (same video, same frame)
    for fp in fake_paths:
        fake_vids, fake_frame = parse_frame_info(fp.name)
        for vid in fake_vids:
            if vid in real_index and fake_frame in real_index[vid]:
                rp = real_index[vid][fake_frame]
                pairs.append((rp, fp, "exact"))
                used_real.add(rp)
                used_fake.add(fp)
                break

    # Pass 2: video matches (same video, different frame)
    for fp in fake_paths:
        if fp in used_fake:
            continue
        fake_vids, fake_frame = parse_frame_info(fp.name)
        for vid in fake_vids:
            if vid in real_index and real_index[vid]:
                # Pick closest frame
                frames = sorted(real_index[vid].keys())
                closest = min(frames, key=lambda f: abs(f - fake_frame))
                rp = real_index[vid][closest]
                pairs.append((rp, fp, "video"))
                used_real.add(rp)
                used_fake.add(fp)
                break

    # Pass 3: unpaired — remaining fakes get random real images
    remaining_fakes = [fp for fp in fake_paths if fp not in used_fake]
    remaining_reals = [rp for rp in real_paths if rp not in used_real]
    if not remaining_reals:
        remaining_reals = list(real_paths)  # reuse if needed

    random.seed(42)
    for fp in remaining_fakes:
        rp = random.choice(remaining_reals)
        pairs.append((rp, fp, "random"))

    # Shuffle and limit
    random.seed(42)
    random.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]

    # Report
    match_counts = defaultdict(int)
    for _, _, mt in pairs:
        match_counts[mt] += 1
    print(f"  Pairs: {len(pairs)} total — "
          f"exact: {match_counts['exact']}, "
          f"video: {match_counts['video']}, "
          f"random: {match_counts['random']}")

    return pairs


# ============================================================
# Embedding pre-computation
# ============================================================

def precompute_paired_embeddings(pairs, encoder, cache_path=None):
    """
    Pre-compute DINOv2 embeddings for all real/fake pairs.
    Returns: real_embs (N, 256, 768), fake_embs (N, 256, 768), method_labels
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        data = torch.load(cache_path, weights_only=True, mmap=True)
        return data["real"], data["fake"], data["methods"]

    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    N = len(pairs)
    num_patches = encoder.num_patches
    dim = encoder.dim

    real_embs = torch.zeros(N, num_patches, dim, dtype=torch.float16)
    fake_embs = torch.zeros(N, num_patches, dim, dtype=torch.float16)
    method_labels = []

    print(f"  Encoding {N} pairs (DINOv2)...")
    t0 = time.time()

    for idx, (real_path, fake_path, match_type) in enumerate(pairs):
        try:
            real_img = transform(Image.open(real_path).convert("RGB"))
            fake_img = transform(Image.open(fake_path).convert("RGB"))

            with torch.no_grad():
                real_embs[idx] = encoder.encode_image(real_img).cpu().half()
                fake_embs[idx] = encoder.encode_image(fake_img).cpu().half()

            # Extract method name from path
            method = fake_path.parent.name
            method_labels.append(method)

        except Exception as e:
            method_labels.append("error")

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (N - idx - 1) / rate
            print(f"    {idx+1}/{N} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({N / elapsed:.1f} pairs/s)")

    if cache_path:
        cache_gb = (real_embs.nelement() + fake_embs.nelement()) * 2 / 1e9
        print(f"  Caching to {cache_path} ({cache_gb:.1f} GB)")
        torch.save({"real": real_embs, "fake": fake_embs,
                     "methods": method_labels}, cache_path)

    return real_embs, fake_embs, method_labels


# ============================================================
# Training
# ============================================================

def train(args):
    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Encoder ---
    if args.dualpass:
        from dual_pass_encoder import DualPassEncoder
        encoder = DualPassEncoder(config["encoder_model"], str(device))
        config["num_patches"] = 512  # 256 structural + 256 frequency
    elif args.augmented:
        from freq_augmented_encoder import FreqAugmentedEncoder
        encoder = FreqAugmentedEncoder(config["encoder_model"], str(device))
    elif args.combined:
        from frequency_encoder import CombinedEncoder
        encoder = CombinedEncoder(config["encoder_model"], str(device))
    else:
        from vision_encoder import DINOv2PatchEncoder
        encoder = DINOv2PatchEncoder(config["encoder_model"], str(device))

    # Use actual encoder dim (768 for all except combined=808)
    config["encoder_dim"] = encoder.dim

    # --- Collect image paths ---
    exts = {".jpg", ".jpeg", ".png"}

    real_paths = sorted(
        p for p in Path(args.real_dir).rglob("*") if p.suffix.lower() in exts)
    print(f"\nReal images: {len(real_paths)} (from {args.real_dir})")

    # Build balanced pairs per method
    train_methods = []
    all_pairs = []

    for fd in args.fake_dirs:
        fd = Path(fd)
        method = fd.name
        fake_paths_m = sorted(
            p for p in fd.rglob("*") if p.suffix.lower() in exts)
        print(f"  {method}: {len(fake_paths_m)} fake images")
        train_methods.append(method)

        pairs = build_matched_pairs(real_paths, fake_paths_m,
                                     max_pairs=config["max_pairs"] // max(len(args.fake_dirs), 1))
        all_pairs.extend(pairs)

    # Holdout methods (for generalisation testing after training)
    holdout_fake_paths = {}
    if args.holdout_dirs:
        for fd in args.holdout_dirs:
            fd = Path(fd)
            method = fd.name
            paths = sorted(p for p in fd.rglob("*") if p.suffix.lower() in exts)
            print(f"  {method} (HOLDOUT): {len(paths)} fake images")
            holdout_fake_paths[method] = paths

    # Train/val split
    random.seed(42)
    random.shuffle(all_pairs)
    n_val = max(int(len(all_pairs) * config["val_fraction"]), 50)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    print(f"\n  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)}")

    # --- Pre-compute embeddings ---
    method_str = "_".join(sorted(train_methods))
    cache_name = f"paired_cache_{len(train_pairs)}_{method_str}.pt"

    print(f"\nPre-computing training embeddings...")
    train_real, train_fake, train_labels = precompute_paired_embeddings(
        train_pairs, encoder, cache_path=cache_name)

    print(f"\nPre-computing validation embeddings...")
    val_real, val_fake, val_labels = precompute_paired_embeddings(
        val_pairs, encoder, cache_path=None)

    N = train_real.shape[0]

    # --- Model ---
    model = ConstraintNetwork(
        d_model=config["d_model"],
        d_state=config["d_state"],
        vocab_size=None,
        max_seq_len=config["num_patches"],
        dropout=config["dropout"],
        alpha=config["alpha"],
    ).to(device)
    model.input_proj = nn.Linear(config["encoder_dim"], config["d_model"]).to(device)

    # Load pre-trained checkpoint if specified
    if args.pretrained:
        print(f"  Loading pre-trained weights from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        # Lower LR for fine-tuning
        config["lr"] = config["lr"] * 0.3
        print(f"  Fine-tuning LR: {config['lr']:.1e}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {params:,}")

    # --- Optimizer + scheduler ---
    bs = config["batch_size"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_steps = config["epochs"] * (N // bs)
    warmup_steps = config["warmup_epochs"] * (N // bs)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = ConstraintLoss(config["margin"])
    scaler = torch.cuda.amp.GradScaler() if config["use_fp16"] else None

    # --- Training loop ---
    history = []
    best_acc = 0
    no_improve = 0
    step = 0
    if hasattr(args, 'checkpoint') and args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = "pretrained_paired_best.pt" if args.pretrained else \
                         ("dualpass_paired_best.pt" if args.dualpass else \
                         ("augmented_paired_best.pt" if args.augmented else \
                         ("combined_paired_best.pt" if args.combined else "paired_constraint_best.pt")))

    print(f"\nTraining: {N} real/fake pairs")
    print(f"  Methods: {', '.join(train_methods)}")
    if holdout_fake_paths:
        print(f"  Holdout: {', '.join(holdout_fake_paths.keys())}")
    print(f"  {N // bs} batches/epoch")
    print("=" * 80)

    for epoch in range(config["epochs"]):
        t0 = time.time()
        model.train()

        perm = torch.randperm(N)
        total_loss = 0
        n_batches = 0

        for b in range(0, N - bs, bs):
            idx = perm[b:b + bs]
            pos_batch = train_real[idx].to(device).float()
            neg_batch = train_fake[idx].to(device).float()

            if config["use_fp16"]:
                with torch.cuda.amp.autocast():
                    ep = model(pos_batch)
                    en = model(neg_batch)
                    loss = criterion(ep, en)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                ep = model(pos_batch)
                en = model(neg_batch)
                loss = criterion(ep, en)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            step += 1

        # --- Validate ---
        model.eval()
        val_energies_pos = []
        val_energies_neg = []
        method_correct = defaultdict(int)
        method_total = defaultdict(int)

        with torch.no_grad():
            N_val = val_real.shape[0]
            for b in range(0, N_val - bs, bs):
                vp = val_real[b:b + bs].to(device).float()
                vn = val_fake[b:b + bs].to(device).float()

                if config["use_fp16"]:
                    with torch.cuda.amp.autocast():
                        ep_v = model(vp)
                        en_v = model(vn)
                else:
                    ep_v = model(vp)
                    en_v = model(vn)

                val_energies_pos.extend(ep_v.cpu().tolist())
                val_energies_neg.extend(en_v.cpu().tolist())

                preds = (ep_v < en_v)
                for j in range(preds.shape[0]):
                    if b + j < len(val_labels):
                        m = val_labels[b + j]
                    else:
                        m = "unknown"
                    method_correct[m] += int(preds[j])
                    method_total[m] += 1

        total_correct = sum(method_correct.values())
        total_total = sum(method_total.values())
        acc = total_correct / max(total_total, 1)

        mean_e_pos = np.mean(val_energies_pos) if val_energies_pos else 0
        mean_e_neg = np.mean(val_energies_neg) if val_energies_neg else 0
        energy_gap = mean_e_neg - mean_e_pos

        dt = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        tag = ""
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            tag = " *"
        else:
            no_improve += 1

        rec = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(n_batches, 1),
            "val_acc": acc,
            "energy_gap": energy_gap,
            "mean_e_pos": mean_e_pos,
            "mean_e_neg": mean_e_neg,
            "lr": lr,
            "time": dt,
            "per_method": {m: method_correct[m] / max(method_total[m], 1)
                           for m in method_total}
        }
        history.append(rec)

        method_str_short = " | ".join(
            f"{m[:6]}={method_correct[m] / max(method_total[m], 1):.0%}"
            for m in sorted(method_total.keys()))

        print(f"E{epoch + 1:3d} | loss={total_loss / max(n_batches, 1):.4f} "
              f"acc={acc:.3f} gap={energy_gap:+.3f} lr={lr:.1e} "
              f"| {method_str_short} | {dt:.0f}s{tag}")

        if no_improve >= config["patience"]:
            print(f"  Early stopping (patience={config['patience']})")
            break

    # --- Holdout evaluation ---
    all_test_results = {}
    if holdout_fake_paths:
        print(f"\n{'='*80}")
        print("HOLDOUT GENERALISATION TEST")
        print(f"{'='*80}")
        print("(These methods were NEVER seen during training)\n")

        model.eval()

        from torchvision import transforms
        from PIL import Image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        def compute_energies_batch(image_paths):
            energies = []
            for i in range(0, len(image_paths), bs):
                batch_paths = image_paths[i:i + bs]
                imgs = []
                for p in batch_paths:
                    try:
                        imgs.append(transform(Image.open(p).convert("RGB")))
                    except Exception:
                        continue
                if not imgs:
                    continue
                batch_t = torch.stack(imgs)
                with torch.no_grad():
                    patches = encoder.encode_batch(batch_t).to(device)
                    if config["use_fp16"]:
                        with torch.cuda.amp.autocast():
                            e = model(patches)
                    else:
                        e = model(patches)
                    energies.extend(e.cpu().tolist())
            return np.array(energies)

        # Real baseline from validation positives
        real_energies = np.array(val_energies_pos)

        print(f"  {'Method':<20s} {'N':>6s} {'Acc':>8s} {'AUC':>8s} {'Gap':>8s}")
        print(f"  {'-'*54}")

        for method, paths in holdout_fake_paths.items():
            test_paths = paths[:min(2000, len(paths))]
            if not test_paths:
                continue

            fake_energies = compute_energies_batch(test_paths)

            n_test = min(len(real_energies), len(fake_energies))
            re = real_energies[:n_test]
            fe = fake_energies[:n_test]

            all_e = np.concatenate([re, fe])
            all_labels = np.concatenate([np.zeros(n_test), np.ones(n_test)])

            thresholds = np.percentile(all_e, np.arange(2, 99, 2))
            best_acc_h = max((all_e > t == all_labels).mean() for t in thresholds)

            n_correct = sum(1 for r in re for f in fe if r < f)
            auc = n_correct / max(n_test * n_test, 1)
            gap = fe.mean() - re.mean()

            all_test_results[method] = {
                "accuracy": float(best_acc_h),
                "auc": float(auc),
                "energy_gap": float(gap),
                "n_samples": int(n_test),
            }

            verdict = "GENERALISES" if auc > 0.7 else ("weak signal" if auc > 0.55 else "not detected")
            print(f"  {method:<20s} {n_test:>6d} "
                  f"{best_acc_h:>8.1%} {auc:>8.3f} {gap:>+8.3f}  → {verdict}")

    # --- Per-patch analysis ---
    print(f"\n{'='*80}")
    print("PER-PATCH ENERGY ANALYSIS")
    print(f"{'='*80}")
    print("(Which patches carry the most signal for each method?)\n")

    model.eval()
    N_val = val_real.shape[0]
    with torch.no_grad():
        method_patch_energies = defaultdict(list)

        for b in range(0, min(N_val, 500) - bs, bs):
            vp = val_real[b:b + bs].to(device).float()
            vn = val_fake[b:b + bs].to(device).float()

            _, per_pos_real = model(vp, return_per_position=True)
            _, per_pos_fake = model(vn, return_per_position=True)

            diff = (per_pos_fake - per_pos_real).cpu().numpy()

            for j in range(diff.shape[0]):
                if b + j < len(val_labels):
                    m = val_labels[b + j]
                    method_patch_energies[m].append(diff[j])

    for method in sorted(method_patch_energies.keys()):
        patches = np.stack(method_patch_energies[method])
        mean_patch = patches.mean(axis=0)

        top_patches = np.argsort(mean_patch)[-10:]
        top_rows = top_patches // 16
        top_cols = top_patches % 16

        center_row = top_rows.mean()
        center_col = top_cols.mean()

        region = ""
        if center_row < 6:
            region = "forehead/upper"
        elif center_row < 10:
            region = "eyes/nose"
        else:
            region = "mouth/jaw"
        if center_col < 6:
            region += "-left"
        elif center_col > 10:
            region += "-right"
        else:
            region += "-center"

        spread = np.std(top_rows) + np.std(top_cols)
        locality = "localised" if spread < 4 else "diffuse"

        print(f"  {method}:")
        print(f"    Hotspot: {region} ({locality})")
        print(f"    Peak patch energy diff: {mean_patch.max():.4f}")
        print(f"    Mean patch energy diff: {mean_patch.mean():.4f}")
        print(f"    Top-10% / bottom-10%: "
              f"{np.percentile(mean_patch, 90):.4f} / "
              f"{np.percentile(mean_patch, 10):.4f}")

    # --- Save results ---
    results = {
        "config": {k: str(v) if not isinstance(v, (int, float, bool)) else v
                   for k, v in config.items()},
        "train_methods": train_methods,
        "holdout_methods": list(holdout_fake_paths.keys()),
        "history": history,
        "best_acc": best_acc,
        "per_method_val": {m: method_correct[m] / max(method_total[m], 1)
                           for m in method_total},
        "holdout_results": all_test_results,
        "params": params,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
    }
    with open("paired_constraint_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest validation accuracy: {best_acc:.3f}")
    print(f"Results saved to paired_constraint_results.json")
    print(f"Model saved to {checkpoint_path}")

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if all_test_results:
        for method, res in all_test_results.items():
            if res["auc"] > 0.7:
                verdict = "strong detection"
            elif res["auc"] > 0.55:
                verdict = "weak signal"
            else:
                verdict = "not detected"
            print(f"  [HOLDOUT] {method:<20s}: "
                  f"AUC={res['auc']:.3f}  acc={res['accuracy']:.1%}  "
                  f"gap={res['energy_gap']:+.3f}  → {verdict}")

    print(f"\nTo evaluate on test sets:")
    print(f"  python eval_vision.py \\")
    print(f"    --checkpoint {checkpoint_path} \\")
    print(f"    --real_dir <test_real_dir> \\")
    print(f"    --fake_dirs <test_fake_dirs...>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train constraint network on paired real/fake data")
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory of real face crops")
    parser.add_argument("--fake_dirs", type=str, nargs="+", required=True,
                        help="Directories of deepfake face crops (training methods)")
    parser.add_argument("--holdout_dirs", type=str, nargs="*", default=None,
                        help="Directories of deepfake methods held out for "
                             "generalisation testing (never trained on)")
    parser.add_argument("--pairs_per_method", type=int, default=None,
                        help="Max pairs per method (default: max_pairs / num_methods)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pre-trained checkpoint for fine-tuning")
    parser.add_argument("--combined", action="store_true",
                        help="Use combined encoder (DINOv2 + frequency branch)")
    parser.add_argument("--augmented", action="store_true",
                        help="Use frequency-augmented encoder (overlay before DINOv2)")
    parser.add_argument("--dualpass", action="store_true",
                        help="Use dual-pass encoder (DINOv2 on image + DINOv2 on freq heatmap)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Energy aggregation alpha (default: use config)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Output checkpoint path (default: auto-generated)")
    args = parser.parse_args()

    if args.pairs_per_method:
        CONFIG["max_pairs"] = args.pairs_per_method * len(args.fake_dirs)
    if args.alpha is not None:
        CONFIG["alpha"] = args.alpha

    # Set global seeds
    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    random.seed(args.seed)

    train(args)
