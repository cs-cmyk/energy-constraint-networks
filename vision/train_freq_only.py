"""
Train frequency constraint network INDEPENDENTLY.

Processes DINOv2(frequency_heatmap) features with gate-weighted loss.
No structural branch — train in isolation, combine at eval time.

The heuristic gate weights the loss per sample:
  - High gate → smoothed fake → train hard (clear frequency signal)
  - Low gate → reenactment fake → train softly (no frequency signal)
  - Real images → always train (keep E_real low)

Usage:
  python train_freq_only.py \
    --real_dir ff_c23_faces/train_original \
    --fake_dirs ff_c23_faces/train_Deepfakes ff_c23_faces/train_Face2Face \
                ff_c23_faces/train_FaceSwap ff_c23_faces/train_NeuralTextures \
                ff_c23_faces/train_FaceShifter \
                ff_c23_faces/train_smooth_corrupted \
                ff_c23_faces/train_bilateral_corrupted \
                ff_c23_faces/train_localgan_corrupted
"""

import torch
import torch.nn as nn
import torch.cuda.amp
import time
import json
import random
import re
import math
import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from model import ConstraintNetwork, ConstraintLoss


CONFIG = dict(
    max_pairs=60000,
    val_fraction=0.1,

    encoder_model="dinov2_vitb14",

    d_model=384,
    d_state=64,
    dropout=0.15,
    alpha=0.3,

    batch_size=128,
    lr=1e-4,
    weight_decay=0.01,
    epochs=60,
    margin=5.0,
    grad_clip=1.0,
    use_fp16=True,
    patience=15,
    warmup_epochs=2,
)


# ============================================================
# Pairing logic
# ============================================================

def parse_frame_info(filename):
    stem = Path(filename).stem
    frame_match = re.search(r'_f(\d+)', stem)
    frame_num = int(frame_match.group(1)) if frame_match else -1
    if frame_match:
        video_part = stem[:frame_match.start()]
    else:
        video_part = stem
    video_ids = [v for v in video_part.split('_') if v.isdigit()]
    return video_ids, frame_num


def build_matched_pairs(real_paths, fake_paths, max_pairs=None):
    real_index = defaultdict(dict)
    for p in real_paths:
        vids, frame = parse_frame_info(p.name)
        for vid in vids:
            real_index[vid][frame] = p

    pairs = []
    used_fake = set()

    for fp in fake_paths:
        fake_vids, fake_frame = parse_frame_info(fp.name)
        for vid in fake_vids:
            if vid in real_index and fake_frame in real_index[vid]:
                pairs.append((real_index[vid][fake_frame], fp, "exact"))
                used_fake.add(fp)
                break

    for fp in fake_paths:
        if fp in used_fake:
            continue
        fake_vids, fake_frame = parse_frame_info(fp.name)
        for vid in fake_vids:
            if vid in real_index and real_index[vid]:
                frames = sorted(real_index[vid].keys())
                closest = min(frames, key=lambda f: abs(f - fake_frame))
                pairs.append((real_index[vid][closest], fp, "video"))
                used_fake.add(fp)
                break

    remaining = [fp for fp in fake_paths if fp not in used_fake]
    real_list = list(real_paths)
    random.seed(42)
    for fp in remaining:
        pairs.append((random.choice(real_list), fp, "random"))

    random.seed(42)
    random.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]

    match_counts = defaultdict(int)
    for _, _, mt in pairs:
        match_counts[mt] += 1
    print(f"  Pairs: {len(pairs)} total — "
          f"exact: {match_counts['exact']}, "
          f"video: {match_counts['video']}, "
          f"random: {match_counts['random']}")
    return pairs


# ============================================================
# Render frequency heatmap as RGB
# ============================================================

def render_freq_image(img_tensor, freq_encoder):
    """Compute frequency heatmap and render as 3-channel RGB image."""
    import torch.nn.functional as F

    heatmap = freq_encoder.compute_heatmap(img_tensor)
    num_bands = freq_encoder.num_bands
    stats_per = freq_encoder.stats_per_band

    low_power = heatmap[0]
    high_power = heatmap[(num_bands - 1) * stats_per]
    ratio = heatmap[-2]

    def norm(x):
        mn, mx = x.min(), x.max()
        if mx > mn:
            return (x - mn) / (mx - mn)
        return torch.ones_like(x) * 0.5

    freq_img = torch.stack([norm(high_power), norm(low_power), norm(ratio)])
    freq_img = F.interpolate(
        freq_img.unsqueeze(0),
        size=(img_tensor.shape[-2], img_tensor.shape[-1]),
        mode="bilinear", align_corners=False
    ).squeeze(0)

    return freq_img.clamp(0, 1)


# ============================================================
# Pre-compute DINOv2(heatmap) features + heuristic gates
# ============================================================

def precompute_freq_embeddings(pairs, dinov2_encoder, freq_heatmap_encoder,
                                cache_path=None):
    """
    Pre-compute DINOv2 features of frequency heatmaps + heuristic gates.

    Returns:
      real_freq:   (N, 256, 768)  — DINOv2(freq_heatmap(real image))
      fake_freq:   (N, 256, 768)  — DINOv2(freq_heatmap(fake image))
      fake_gates:  (N,)           — heuristic confidence per fake sample
      methods: list of method labels
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        data = torch.load(cache_path, weights_only=True, mmap=True)
        return (data["real_freq"], data["fake_freq"],
                data["fake_gates"], data["methods"])

    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    N = len(pairs)
    np_ = dinov2_encoder.num_patches
    dim = dinov2_encoder.dim

    real_freq = torch.zeros(N, np_, dim, dtype=torch.float16)
    fake_freq = torch.zeros(N, np_, dim, dtype=torch.float16)
    methods = []

    # Baseline from real images (median ratio)
    print(f"  Computing baseline frequency stats from first 200 real images...")
    real_medians = []
    for idx in range(min(200, N)):
        try:
            rp = pairs[idx][0]
            real_img = transform(Image.open(rp).convert("RGB"))
            heatmap = freq_heatmap_encoder.compute_heatmap(real_img)
            ratio = heatmap[-2]
            real_medians.append(ratio.median().item())
        except Exception:
            continue

    baseline_median = np.median(real_medians) if real_medians else 0.0003
    baseline_std = np.std(real_medians) if real_medians else 0.0001
    print(f"    Baseline: median_ratio={baseline_median:.6f}, std={baseline_std:.6f}")

    # Encode all pairs
    fake_gates = torch.zeros(N, dtype=torch.float32)

    print(f"  Encoding {N} pairs (DINOv2 on freq heatmap + gates)...")
    t0 = time.time()

    for idx, (rp, fp, _) in enumerate(pairs):
        try:
            real_img = transform(Image.open(rp).convert("RGB"))
            fake_img = transform(Image.open(fp).convert("RGB"))

            with torch.no_grad():
                real_hmap = render_freq_image(real_img, freq_heatmap_encoder)
                fake_hmap = render_freq_image(fake_img, freq_heatmap_encoder)

                real_freq[idx] = dinov2_encoder.encode_image(real_hmap).cpu().half()
                fake_freq[idx] = dinov2_encoder.encode_image(fake_hmap).cpu().half()

            # Heuristic gate for fake image
            fake_heatmap = freq_heatmap_encoder.compute_heatmap(fake_img)
            fake_ratio = fake_heatmap[-2]
            median_ratio = fake_ratio.median().item()

            z_score = (baseline_median - median_ratio) / max(baseline_std, 1e-8)
            gate = max(0.0, min(1.0, z_score / 2.0))
            fake_gates[idx] = gate

            methods.append(fp.parent.name)
        except Exception:
            methods.append("error")

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (N - idx - 1) / rate
            print(f"    {idx+1}/{N} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({N / elapsed:.1f} pairs/s)")

    # Report gate distribution per method
    method_gates = {}
    for i, m in enumerate(methods):
        if m not in method_gates:
            method_gates[m] = []
        method_gates[m].append(fake_gates[i].item())
    print(f"\n  Gate distribution per method:")
    for m in sorted(method_gates.keys()):
        vals = method_gates[m]
        print(f"    {m}: mean={np.mean(vals):.3f}, "
              f"std={np.std(vals):.3f}, "
              f"min={np.min(vals):.3f}, max={np.max(vals):.3f}")

    if cache_path:
        cache_gb = (real_freq.nelement() + fake_freq.nelement()) * 2 / 1e9
        print(f"  Caching to {cache_path} ({cache_gb:.1f} GB)")
        torch.save({
            "real_freq": real_freq, "fake_freq": fake_freq,
            "fake_gates": fake_gates, "methods": methods
        }, cache_path)

    return real_freq, fake_freq, fake_gates, methods


# ============================================================
# Training
# ============================================================

def train(args):
    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Encoders ---
    from vision_encoder import DINOv2PatchEncoder
    from frequency_heatmap import FrequencyHeatmapEncoder

    dinov2_encoder = DINOv2PatchEncoder(config["encoder_model"], str(device))
    freq_heatmap_encoder = FrequencyHeatmapEncoder(224, device=str(device))

    # --- Collect and pair ---
    exts = {".jpg", ".jpeg", ".png"}
    real_paths = sorted(
        p for p in Path(args.real_dir).rglob("*") if p.suffix.lower() in exts)
    print(f"\nReal images: {len(real_paths)}")

    train_methods = []
    all_pairs = []
    for fd in args.fake_dirs:
        fd = Path(fd)
        method = fd.name
        fake_paths_m = sorted(
            p for p in fd.rglob("*") if p.suffix.lower() in exts)
        print(f"  {method}: {len(fake_paths_m)} fake images")
        train_methods.append(method)
        pairs = build_matched_pairs(
            real_paths, fake_paths_m,
            max_pairs=config["max_pairs"] // max(len(args.fake_dirs), 1))
        all_pairs.extend(pairs)

    random.seed(42)
    random.shuffle(all_pairs)
    n_val = max(int(len(all_pairs) * config["val_fraction"]), 50)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    print(f"\n  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)}")

    # --- Pre-compute ---
    method_str = "_".join(sorted(train_methods))[:80]
    cache_train = f"freq_only_cache_train_{len(train_pairs)}_{method_str}.pt"
    cache_val = f"freq_only_cache_val_{len(val_pairs)}_{method_str}.pt"

    print(f"\nPre-computing training embeddings...")
    (train_rf, train_ff,
     train_gates, train_labels) = precompute_freq_embeddings(
        train_pairs, dinov2_encoder, freq_heatmap_encoder,
        cache_path=cache_train)

    print(f"\nPre-computing validation embeddings...")
    (val_rf, val_ff,
     val_gates, val_labels) = precompute_freq_embeddings(
        val_pairs, dinov2_encoder, freq_heatmap_encoder,
        cache_path=cache_val)

    N = train_rf.shape[0]

    # --- Model ---
    model = ConstraintNetwork(
        d_model=config["d_model"], d_state=config["d_state"],
        vocab_size=None, max_seq_len=256,
        dropout=config["dropout"], alpha=config["alpha"]
    ).to(device)
    model.input_proj = nn.Linear(
        dinov2_encoder.dim, config["d_model"]).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Frequency network: {params:,} params")

    # --- Optimizer ---
    bs = config["batch_size"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"])

    total_steps = config["epochs"] * (N // bs)
    warmup_steps = config["warmup_epochs"] * (N // bs)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if config["use_fp16"] else None
    margin = config["margin"]

    # --- Training loop ---
    history = []
    best_acc = 0
    no_improve = 0
    checkpoint_path = getattr(args, 'checkpoint', None) or "freq_only_best.pt"

    print(f"\nTraining: {N} pairs, {N // bs} batches/epoch")
    print(f"  Methods: {', '.join(train_methods)}")
    print(f"  Gate-weighted frequency loss")
    print("=" * 80)

    for epoch in range(config["epochs"]):
        t0 = time.time()
        model.train()

        perm = torch.randperm(N)
        total_loss = 0
        total_gate = 0
        n_batches = 0

        for b in range(0, N - bs, bs):
            idx = perm[b:b + bs]

            rf = train_rf[idx].to(device).float()
            ff = train_ff[idx].to(device).float()
            gates = train_gates[idx].to(device).float()

            if config["use_fp16"]:
                with torch.cuda.amp.autocast():
                    ef_pos = model(rf)
                    ef_neg = model(ff)

                    # Gate-weighted loss:
                    # Real energy always penalized (keep E_real low)
                    # Fake energy weighted by gate (focus on smoothed fakes)
                    loss_pos = ef_pos.pow(2).mean()
                    loss_neg = (gates * torch.relu(margin - ef_neg).pow(2)).mean()
                    loss = loss_pos + loss_neg

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                ef_pos = model(rf)
                ef_neg = model(ff)

                loss_pos = ef_pos.pow(2).mean()
                loss_neg = (gates * torch.relu(margin - ef_neg).pow(2)).mean()
                loss = loss_pos + loss_neg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()
            total_gate += gates.mean().item()
            n_batches += 1

        # --- Validate ---
        model.eval()
        val_e_pos = []
        val_e_neg = []

        N_val = val_rf.shape[0]
        with torch.no_grad():
            for b in range(0, N_val - bs, bs):
                rf = val_rf[b:b + bs].to(device).float()
                ff = val_ff[b:b + bs].to(device).float()

                if config["use_fp16"]:
                    with torch.cuda.amp.autocast():
                        ep = model(rf)
                        en = model(ff)
                else:
                    ep = model(rf)
                    en = model(ff)

                val_e_pos.extend(ep.cpu().tolist())
                val_e_neg.extend(en.cpu().tolist())

        mean_pos = np.mean(val_e_pos) if val_e_pos else 0
        mean_neg = np.mean(val_e_neg) if val_e_neg else 0
        gap = mean_neg - mean_pos
        acc = np.mean([1 if p < n else 0
                       for p, n in zip(val_e_pos, val_e_neg)])

        dt = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        avg_gate = total_gate / max(n_batches, 1)

        tag = ""
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            tag = " *"
        else:
            no_improve += 1

        print(f"E{epoch + 1:3d} | loss={total_loss / max(n_batches, 1):.4f} "
              f"acc={acc:.3f} gap={gap:+.2f} "
              f"E(real)={mean_pos:.2f} E(fake)={mean_neg:.2f} "
              f"gate={avg_gate:.3f} lr={lr:.1e} | {dt:.0f}s{tag}")

        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / max(n_batches, 1),
            "val_acc": acc, "gap": gap,
            "e_real": mean_pos, "e_fake": mean_neg,
            "avg_gate": avg_gate,
            "lr": lr, "time": dt,
        })

        if no_improve >= config["patience"]:
            print(f"  Early stopping (patience={config['patience']})")
            break

    # --- Report ---
    print(f"\n{'='*80}")
    print(f"  Best accuracy: {best_acc:.3f}")
    print(f"  Final E(real): {mean_pos:.3f}")
    print(f"  Final E(fake): {mean_neg:.3f}")
    print(f"  Final gap:     {gap:+.3f}")
    print(f"  Model saved to {checkpoint_path}")

    results = {
        "config": {k: str(v) if not isinstance(v, (int, float, bool)) else v
                   for k, v in config.items()},
        "train_methods": train_methods,
        "history": history,
        "best_acc": best_acc,
    }
    with open("freq_only_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCombine with structural branch at eval time:")
    print(f"  python eval_combined.py \\")
    print(f"    --struct_checkpoint pretrained_paired_best.pt \\")
    print(f"    --freq_checkpoint {checkpoint_path} \\")
    print(f"    --real_dir <test_real> --fake_dirs <test_fakes>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Energy aggregation alpha (default: use config)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Output checkpoint path")
    args = parser.parse_args()

    if args.alpha is not None:
        CONFIG["alpha"] = args.alpha

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train(args)
