"""
Train local texture constraint network INDEPENDENTLY.

Processes DINOv2(original image) features — same encoder as structural,
but trained ONLY on localized texture corruptions (localgan).

This branch detects: "one region of this face has different texture
quality than its neighbors" — exactly what NeuralTextures does.

Trained in isolation. Combined at eval time with structural + frequency.

Usage:
  python train_local_only.py \
    --real_dir ff_c23_faces/train_original \
    --fake_dirs ff_c23_faces/train_localgan_corrupted
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
# Pre-compute DINOv2(original) features
# ============================================================

def precompute_embeddings(pairs, encoder, cache_path=None):
    """Pre-compute DINOv2 embeddings for original images."""
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
    np_ = encoder.num_patches
    dim = encoder.dim

    real_embs = torch.zeros(N, np_, dim, dtype=torch.float16)
    fake_embs = torch.zeros(N, np_, dim, dtype=torch.float16)
    methods = []

    print(f"  Encoding {N} pairs (DINOv2)...")
    t0 = time.time()

    for idx, (rp, fp, _) in enumerate(pairs):
        try:
            real_img = transform(Image.open(rp).convert("RGB"))
            fake_img = transform(Image.open(fp).convert("RGB"))

            with torch.no_grad():
                real_embs[idx] = encoder.encode_image(real_img).cpu().half()
                fake_embs[idx] = encoder.encode_image(fake_img).cpu().half()

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

    if cache_path:
        torch.save({"real": real_embs, "fake": fake_embs, "methods": methods},
                    cache_path)

    return real_embs, fake_embs, methods


# ============================================================
# Training
# ============================================================

def train(args):
    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Encoder ---
    from vision_encoder import DINOv2PatchEncoder
    encoder = DINOv2PatchEncoder(config["encoder_model"], str(device))

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
    cache_train = f"local_cache_train_{len(train_pairs)}_{method_str}.pt"
    cache_val = f"local_cache_val_{len(val_pairs)}_{method_str}.pt"

    print(f"\nPre-computing training embeddings...")
    train_real, train_fake, train_labels = precompute_embeddings(
        train_pairs, encoder, cache_path=cache_train)

    print(f"\nPre-computing validation embeddings...")
    val_real, val_fake, val_labels = precompute_embeddings(
        val_pairs, encoder, cache_path=cache_val)

    N = train_real.shape[0]

    # --- Model ---
    model = ConstraintNetwork(
        d_model=config["d_model"], d_state=config["d_state"],
        vocab_size=None, max_seq_len=256,
        dropout=config["dropout"], alpha=config["alpha"]
    ).to(device)
    model.input_proj = nn.Linear(encoder.dim, config["d_model"]).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Local texture network: {params:,} params")

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
    criterion = ConstraintLoss(config["margin"])
    scaler = torch.cuda.amp.GradScaler() if config["use_fp16"] else None

    # --- Training loop ---
    history = []
    best_acc = 0
    no_improve = 0
    checkpoint_path = getattr(args, 'checkpoint', None) or "local_only_best.pt"

    print(f"\nTraining: {N} pairs, {N // bs} batches/epoch")
    print(f"  Methods: {', '.join(train_methods)}")
    print("=" * 80)

    for epoch in range(config["epochs"]):
        t0 = time.time()
        model.train()

        perm = torch.randperm(N)
        total_loss = 0
        n_batches = 0

        for b in range(0, N - bs, bs):
            idx = perm[b:b + bs]

            pos = train_real[idx].to(device).float()
            neg = train_fake[idx].to(device).float()

            if config["use_fp16"]:
                with torch.cuda.amp.autocast():
                    ep = model(pos)
                    en = model(neg)
                    loss = criterion(ep, en)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                ep = model(pos)
                en = model(neg)
                loss = criterion(ep, en)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        # --- Validate ---
        model.eval()
        val_e_pos = []
        val_e_neg = []

        N_val = val_real.shape[0]
        with torch.no_grad():
            for b in range(0, N_val - bs, bs):
                vp = val_real[b:b + bs].to(device).float()
                vn = val_fake[b:b + bs].to(device).float()

                if config["use_fp16"]:
                    with torch.cuda.amp.autocast():
                        ep = model(vp)
                        en = model(vn)
                else:
                    ep = model(vp)
                    en = model(vn)

                val_e_pos.extend(ep.cpu().tolist())
                val_e_neg.extend(en.cpu().tolist())

        mean_pos = np.mean(val_e_pos) if val_e_pos else 0
        mean_neg = np.mean(val_e_neg) if val_e_neg else 0
        gap = mean_neg - mean_pos
        acc = np.mean([1 if p < n else 0
                       for p, n in zip(val_e_pos, val_e_neg)])

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

        print(f"E{epoch + 1:3d} | loss={total_loss / max(n_batches, 1):.4f} "
              f"acc={acc:.3f} gap={gap:+.2f} "
              f"E(real)={mean_pos:.2f} E(fake)={mean_neg:.2f} "
              f"lr={lr:.1e} | {dt:.0f}s{tag}")

        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / max(n_batches, 1),
            "val_acc": acc, "gap": gap,
            "e_real": mean_pos, "e_fake": mean_neg,
            "lr": lr, "time": dt,
        })

        if no_improve >= config["patience"]:
            print(f"  Early stopping (patience={config['patience']})")
            break

    print(f"\n  Best accuracy: {best_acc:.3f}")
    print(f"  Model saved to {checkpoint_path}")


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

