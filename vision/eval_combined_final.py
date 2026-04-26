"""
Evaluate combined structural + frequency branches.

Loads two independently trained checkpoints:
  - Structural: trained on 5 FF++ methods with corruption pretraining
  - Frequency: trained on DINOv2(heatmap) features with gate-weighted loss

Combines at eval time with heuristic gate:
  E = E_struct + gate * E_freq
  gate = heuristic_confidence(frequency_heatmap)

Usage:
  python eval_combined_final.py \
    --struct_checkpoint pretrained_paired_best.pt \
    --freq_checkpoint freq_only_best.pt \
    --real_dir ff_c23_faces/original \
    --fake_dirs ff_c23_faces/Deepfakes ff_c23_faces/NeuralTextures

  python eval_combined_final.py \
    --struct_checkpoint pretrained_paired_best.pt \
    --freq_checkpoint freq_only_best.pt \
    --real_dir celebdf_faces/real \
    --fake_dirs celebdf_faces/fake
"""

import torch
import torch.nn as nn
import torch.cuda.amp
import numpy as np
import json
import argparse
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from model import ConstraintNetwork


def load_model(checkpoint_path, device, input_dim=768, d_model=384):
    """Load a single constraint network."""
    model = ConstraintNetwork(
        d_model=d_model, d_state=64, vocab_size=None,
        max_seq_len=256, dropout=0.15, alpha=0.3
    ).to(device)
    model.input_proj = nn.Linear(input_dim, d_model).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {params:,} params from {checkpoint_path}")
    return model


def render_freq_image(img_tensor, freq_encoder):
    """Compute frequency heatmap and render as RGB."""
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


def compute_heuristic_gate(img_tensor, freq_encoder, baseline_stats):
    """
    Heuristic confidence gate from median frequency ratio.
    High gate = image looks smoothed. Low gate = normal frequency profile.
    """
    heatmap = freq_encoder.compute_heatmap(img_tensor)
    ratio = heatmap[-2]  # (16, 16)
    median_ratio = ratio.median().item()

    z_score = (baseline_stats["median_ratio"] - median_ratio) / \
              max(baseline_stats["std_ratio"], 1e-8)

    # Activate when ratio is below baseline
    gate = max(0.0, min(1.0, z_score / 2.0))
    return gate


def compute_baseline_stats(image_paths, freq_encoder, num_samples=200):
    """Compute baseline frequency stats from real images."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    medians = []
    for p in image_paths[:num_samples]:
        try:
            img = transform(Image.open(p).convert("RGB"))
            heatmap = freq_encoder.compute_heatmap(img)
            ratio = heatmap[-2]
            medians.append(ratio.median().item())
        except Exception:
            continue

    stats = {
        "median_ratio": np.median(medians),
        "std_ratio": np.std(medians),
    }
    print(f"  Baseline: median_ratio={stats['median_ratio']:.6f}, "
          f"std_ratio={stats['std_ratio']:.6f}")
    return stats


def compute_energies(struct_model, freq_model, dinov2_encoder,
                      freq_heatmap_encoder, baseline_stats,
                      image_paths, device, local_model=None, local_beta=0.3):
    """Compute structural, frequency, local, and combined energies."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    structural = []
    frequency = []
    local = []
    combined = []
    gates = []

    for p in image_paths:
        try:
            img = transform(Image.open(p).convert("RGB"))

            with torch.no_grad():
                # Structural: DINOv2(original)
                s_feat = dinov2_encoder.encode_image(img).unsqueeze(0).to(device)

                # Frequency: DINOv2(heatmap)
                freq_img = render_freq_image(img, freq_heatmap_encoder)
                f_feat = dinov2_encoder.encode_image(freq_img).unsqueeze(0).to(device)

                with torch.cuda.amp.autocast():
                    e_s = struct_model(s_feat).item()
                    e_f = freq_model(f_feat).item()

                    # Local texture: same DINOv2(original) features, different network
                    e_l = 0.0
                    if local_model is not None:
                        e_l = local_model(s_feat).item()

            # Heuristic gate for frequency
            gate = compute_heuristic_gate(img, freq_heatmap_encoder, baseline_stats)

            structural.append(e_s)
            frequency.append(e_f)
            local.append(e_l)
            gates.append(gate)
            combined.append(e_s + gate * e_f + local_beta * e_l)

        except Exception:
            continue

    return (np.array(combined), np.array(structural),
            np.array(frequency), np.array(local), np.array(gates))


def compute_metrics(real_e, fake_e):
    n = min(len(real_e), len(fake_e))
    re, fe = real_e[:n], fake_e[:n]
    all_e = np.concatenate([re, fe])
    all_labels = np.concatenate([np.zeros(n), np.ones(n)])
    thresholds = np.percentile(all_e, np.arange(2, 99, 2))
    best_acc = max(((all_e > t) == all_labels).mean() for t in thresholds)
    n_correct = sum(1 for r in re for f in fe if r < f)
    auc = n_correct / max(n * n, 1)
    gap = fe.mean() - re.mean()
    return {"accuracy": float(best_acc), "auc": float(auc),
            "energy_gap": float(gap), "n": int(n)}


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load encoders ---
    from vision_encoder import DINOv2PatchEncoder
    from frequency_heatmap import FrequencyHeatmapEncoder

    dinov2_encoder = DINOv2PatchEncoder("dinov2_vitb14", str(device))
    freq_heatmap_encoder = FrequencyHeatmapEncoder(224, device=str(device))

    # --- Load both models ---
    print(f"\nStructural branch:")
    struct_model = load_model(args.struct_checkpoint, device,
                               input_dim=dinov2_encoder.dim)
    print(f"Frequency branch:")
    freq_model = load_model(args.freq_checkpoint, device,
                             input_dim=dinov2_encoder.dim)

    # Optional local texture branch
    local_model = None
    local_beta = args.local_beta
    if args.local_checkpoint:
        print(f"Local texture branch:")
        local_model = load_model(args.local_checkpoint, device,
                                  input_dim=dinov2_encoder.dim)
        print(f"  Local beta: {local_beta}")

    # --- Collect paths ---
    exts = {".jpg", ".jpeg", ".png"}
    real_paths = sorted(
        p for p in Path(args.real_dir).rglob("*") if p.suffix.lower() in exts)
    if args.max_images:
        real_paths = real_paths[:args.max_images]
    print(f"\nReal images: {len(real_paths)}")

    # --- Baseline stats ---
    print("Computing baseline frequency statistics...")
    baseline_stats = compute_baseline_stats(
        real_paths, freq_heatmap_encoder, num_samples=200)

    # --- Compute real energies ---
    print("Computing real energies...")
    real_c, real_s, real_f, real_l, real_g = compute_energies(
        struct_model, freq_model, dinov2_encoder,
        freq_heatmap_encoder, baseline_stats, real_paths, device,
        local_model=local_model, local_beta=local_beta)

    print(f"  Real: combined={real_c.mean():.4f}, "
          f"structural={real_s.mean():.4f}, "
          f"frequency={real_f.mean():.4f}, "
          f"{'local=' + f'{real_l.mean():.4f}, ' if local_model else ''}"
          f"gate={real_g.mean():.4f}")

    # --- Dataset-level gating ---
    GATE_THRESHOLD = 0.20
    freq_active = real_g.mean() <= GATE_THRESHOLD
    min_gate = args.min_gate

    if freq_active:
        print(f"  → Frequency branch ACTIVE (real gate {real_g.mean():.4f} ≤ {GATE_THRESHOLD})")
    elif min_gate > 0:
        print(f"  → Frequency branch at FLOOR (real gate {real_g.mean():.4f} > {GATE_THRESHOLD})")
        print(f"    Using min_gate={min_gate} instead of full disable")
        real_c = real_s + min_gate * real_f + local_beta * real_l
    else:
        print(f"  → Frequency branch DISABLED (real gate {real_g.mean():.4f} > {GATE_THRESHOLD})")
        print(f"    Falling back to structural{' + local' if local_model else ''}")
        real_c = real_s + local_beta * real_l if local_model else real_s.copy()

    # --- Evaluate per method ---
    header = f"{'Method':<20s} {'N':>5s} {'Acc':>7s} {'AUC':>7s} " \
             f"{'Gap':>7s} {'S-Gap':>7s} {'F-Gap':>7s} "
    if local_model:
        header += f"{'L-Gap':>7s} "
    header += f"{'S-AUC':>7s} {'F-AUC':>7s} "
    if local_model:
        header += f"{'L-AUC':>7s} "
    header += f"{'R-Gate':>7s} {'F-Gate':>7s}"

    print(f"\n{'='*120}")
    print(header)
    print(f"{'='*120}")

    all_results = {}
    for fake_dir in args.fake_dirs:
        fake_dir = Path(fake_dir)
        method = fake_dir.name
        fake_paths = sorted(
            p for p in fake_dir.rglob("*") if p.suffix.lower() in exts)
        if args.max_images:
            fake_paths = fake_paths[:args.max_images]
        if not fake_paths:
            continue

        print(f"  Computing {method} energies...")
        fake_c, fake_s, fake_f, fake_l, fake_g = compute_energies(
            struct_model, freq_model, dinov2_encoder,
            freq_heatmap_encoder, baseline_stats, fake_paths, device,
            local_model=local_model, local_beta=local_beta)

        # If frequency disabled, use structural + local (if available)
        if not freq_active:
            if min_gate > 0:
                fake_c = fake_s + min_gate * fake_f + local_beta * fake_l
            elif local_model:
                fake_c = fake_s + local_beta * fake_l
            else:
                fake_c = fake_s.copy()

        m = compute_metrics(real_c, fake_c)
        ms = compute_metrics(real_s, fake_s)
        mf = compute_metrics(real_f, fake_f)
        ml = compute_metrics(real_l, fake_l) if local_model else None

        all_results[method] = {
            "combined": m, "structural": ms, "frequency": mf,
            "local": ml,
            "real_gate": float(real_g.mean()),
            "fake_gate": float(fake_g.mean()),
        }

        line = f"  {method:<20s} {m['n']:>5d} " \
               f"{m['accuracy']:>7.1%} {m['auc']:>7.3f} " \
               f"{m['energy_gap']:>+7.3f} " \
               f"{ms['energy_gap']:>+7.3f} {mf['energy_gap']:>+7.3f} "
        if local_model:
            line += f"{ml['energy_gap']:>+7.3f} "
        line += f"{ms['auc']:>7.3f} {mf['auc']:>7.3f} "
        if local_model:
            line += f"{ml['auc']:>7.3f} "
        line += f"{real_g.mean():>7.4f} {fake_g.mean():>7.4f}"
        print(line)

    print(f"\n{'='*120}")
    print("S = structural, F = frequency, L = local texture")
    print("R-Gate = real gate, F-Gate = fake gate (frequency branch)")
    if freq_active:
        print("Frequency branch: ACTIVE (dataset-level gate passed)")
    elif min_gate > 0:
        print(f"Frequency branch: FLOOR at {min_gate} (dataset-level gate failed)")
    else:
        print("Frequency branch: DISABLED (dataset-level gate failed)")
    if local_model:
        print(f"Local texture branch: ALWAYS ON at beta={local_beta}")

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("COMPARISON: Structural only vs Combined")
    print(f"{'='*60}")
    for method, r in all_results.items():
        s_auc = r["structural"]["auc"]
        c_auc = r["combined"]["auc"]
        diff = c_auc - s_auc
        symbol = "+" if diff > 0 else ""
        print(f"  {method:<20s}  S={s_auc:.3f}  Combined={c_auc:.3f}  "
              f"({symbol}{diff:.3f})")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--struct_checkpoint", type=str, required=True,
                        help="Structural branch checkpoint")
    parser.add_argument("--freq_checkpoint", type=str, required=True,
                        help="Frequency branch checkpoint")
    parser.add_argument("--local_checkpoint", type=str, default=None,
                        help="Local texture branch checkpoint (optional)")
    parser.add_argument("--local_beta", type=float, default=0.3,
                        help="Weight for local texture branch contribution")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--min_gate", type=float, default=0.0,
                        help="Minimum gate value (floor). Use 0.3 to always keep "
                             "some frequency contribution even when disabled)")
    parser.add_argument("--output", type=str, default="combined_final_eval.json")
    args = parser.parse_args()
    evaluate(args)
