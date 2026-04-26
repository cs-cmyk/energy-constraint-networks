"""
Variance and ablation experiments for the constraint network paper.

Runs:
  1. Multi-seed training (5 seeds) for variance reporting
  2. Alpha ablation (α = 0.1, 0.2, 0.3, 0.4, 0.5)

Each run trains all 3 branches independently, then evaluates combined.
Pre-computed DINOv2 embeddings are reused across seeds (only model init changes).

Estimated time: ~30 min per branch × 3 branches × 5 seeds = ~7.5 hours
  + eval time (~10 min per eval × 2 datasets × 5 seeds = ~1.5 hours)
  Total: ~9 hours

Usage:
  # Full variance experiment (5 seeds)
  python run_experiments.py --experiment seeds \
    --real_dir ff_c23_faces/train_original \
    --fake_dirs ff_c23_faces/train_Deepfakes ff_c23_faces/train_Face2Face \
                ff_c23_faces/train_FaceSwap ff_c23_faces/train_NeuralTextures \
                ff_c23_faces/train_FaceShifter \
    --corruption_dirs ff_c23_faces/train_smooth_corrupted \
                      ff_c23_faces/train_bilateral_corrupted \
                      ff_c23_faces/train_localgan_corrupted \
    --pretrained vision_constraint_celeb.pt \
    --eval_real ff_c23_faces/original \
    --eval_fake_ff ff_c23_faces/Deepfakes ff_c23_faces/Face2Face \
                   ff_c23_faces/FaceSwap ff_c23_faces/NeuralTextures \
                   ff_c23_faces/FaceShifter \
    --eval_real_celeb celebdf_faces/real \
    --eval_fake_celeb celebdf_faces/fake

  # Alpha ablation (single seed, vary α)
  python run_experiments.py --experiment alpha \
    --real_dir ff_c23_faces/train_original \
    --fake_dirs ff_c23_faces/train_Deepfakes ff_c23_faces/train_Face2Face \
                ff_c23_faces/train_FaceSwap ff_c23_faces/train_NeuralTextures \
                ff_c23_faces/train_FaceShifter \
    --corruption_dirs ff_c23_faces/train_smooth_corrupted \
                      ff_c23_faces/train_bilateral_corrupted \
                      ff_c23_faces/train_localgan_corrupted \
    --pretrained vision_constraint_celeb.pt \
    --eval_real ff_c23_faces/original \
    --eval_fake_ff ff_c23_faces/Deepfakes ff_c23_faces/Face2Face \
                   ff_c23_faces/FaceSwap ff_c23_faces/NeuralTextures \
                   ff_c23_faces/FaceShifter \
    --eval_real_celeb celebdf_faces/real \
    --eval_fake_celeb celebdf_faces/fake
"""

import subprocess
import json
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def train_structural(args, seed, alpha, output_name):
    """Train structural branch with given seed and alpha."""
    cmd = [
        "python", "train_paired.py",
        "--real_dir", args.real_dir,
        "--fake_dirs", *args.fake_dirs,
        "--pretrained", args.pretrained,
        "--seed", str(seed),
        "--alpha", str(alpha),
        "--checkpoint", output_name,
    ]
    return run_cmd(cmd, f"Structural branch (seed={seed}, α={alpha})")


def train_frequency(args, seed, alpha, output_name):
    """Train frequency branch with given seed and alpha."""
    all_dirs = list(args.fake_dirs) + list(args.corruption_dirs)
    cmd = [
        "python", "train_freq_only.py",
        "--real_dir", args.real_dir,
        "--fake_dirs", *all_dirs,
        "--seed", str(seed),
        "--alpha", str(alpha),
        "--checkpoint", output_name,
    ]
    return run_cmd(cmd, f"Frequency branch (seed={seed}, α={alpha})")


def train_local(args, seed, alpha, output_name):
    """Train local texture branch with given seed and alpha."""
    local_dirs = list(args.corruption_dirs)
    # Add NeuralTextures if available
    for fd in args.fake_dirs:
        if "NeuralTextures" in fd:
            local_dirs.append(fd)
    cmd = [
        "python", "train_local_only.py",
        "--real_dir", args.real_dir,
        "--fake_dirs", *local_dirs,
        "--seed", str(seed),
        "--alpha", str(alpha),
        "--checkpoint", output_name,
    ]
    return run_cmd(cmd, f"Local texture branch (seed={seed}, α={alpha})")


def evaluate_combined(struct_ckpt, freq_ckpt, local_ckpt,
                       real_dir, fake_dirs, output_file, max_images=5000):
    """Evaluate three-branch combined system."""
    cmd = [
        "python", "eval_combined_final.py",
        "--struct_checkpoint", struct_ckpt,
        "--freq_checkpoint", freq_ckpt,
        "--local_checkpoint", local_ckpt,
        "--real_dir", real_dir,
        "--fake_dirs", *fake_dirs,
        "--max_images", str(max_images),
        "--output", output_file,
    ]
    return run_cmd(cmd, f"Eval: {real_dir} → {output_file}")


def parse_eval_results(json_path):
    """Parse evaluation results JSON."""
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def run_seed_experiment(args):
    """Run full training + eval across multiple seeds."""
    seeds = [42, 123, 456, 789, 1024]
    alpha = 0.3  # fixed alpha for seed experiment

    results_dir = Path("experiment_seeds")
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"#  SEED {seed}")
        print(f"{'#'*70}")

        tag = f"seed{seed}"

        # Checkpoint names
        struct_ckpt = str(results_dir / f"struct_{tag}.pt")
        freq_ckpt = str(results_dir / f"freq_{tag}.pt")
        local_ckpt = str(results_dir / f"local_{tag}.pt")

        # Train all three branches
        train_structural(args, seed, alpha, struct_ckpt)
        train_frequency(args, seed, alpha, freq_ckpt)
        train_local(args, seed, alpha, local_ckpt)

        # Evaluate on FF++
        ff_output = str(results_dir / f"eval_ff_{tag}.json")
        evaluate_combined(
            struct_ckpt, freq_ckpt, local_ckpt,
            args.eval_real, args.eval_fake_ff,
            ff_output, max_images=5000)

        # Evaluate on Celeb-DF
        celeb_output = str(results_dir / f"eval_celeb_{tag}.json")
        evaluate_combined(
            struct_ckpt, freq_ckpt, local_ckpt,
            args.eval_real_celeb, [args.eval_fake_celeb],
            celeb_output, max_images=5000)

        # Collect results
        ff_results = parse_eval_results(ff_output)
        celeb_results = parse_eval_results(celeb_output)

        all_results[seed] = {
            "ff": ff_results,
            "celeb": celeb_results,
        }

    # ==========================================
    # Summary: mean ± std
    # ==========================================
    print(f"\n\n{'='*70}")
    print("VARIANCE REPORT (5 seeds)")
    print(f"{'='*70}\n")

    # Collect per-method AUCs
    methods_ff = ["Deepfakes", "Face2Face", "FaceSwap",
                   "NeuralTextures", "FaceShifter"]

    method_aucs = {m: [] for m in methods_ff}
    celeb_aucs = []

    for seed, res in all_results.items():
        if res["ff"]:
            for method in methods_ff:
                if method in res["ff"]:
                    method_aucs[method].append(
                        res["ff"][method]["combined"]["auc"])
        if res["celeb"]:
            for key in res["celeb"]:
                celeb_aucs.append(res["celeb"][key]["combined"]["auc"])
                break  # only one method (fake)

    print(f"{'Method':<20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'N':>4s}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    for method in methods_ff:
        aucs = method_aucs[method]
        if aucs:
            print(f"{method:<20s} {np.mean(aucs):>8.3f} {np.std(aucs):>8.3f} "
                  f"{np.min(aucs):>8.3f} {np.max(aucs):>8.3f} {len(aucs):>4d}")

    if celeb_aucs:
        print(f"{'Celeb-DF':<20s} {np.mean(celeb_aucs):>8.3f} {np.std(celeb_aucs):>8.3f} "
              f"{np.min(celeb_aucs):>8.3f} {np.max(celeb_aucs):>8.3f} {len(celeb_aucs):>4d}")

    # Save
    summary = {
        "experiment": "seed_variance",
        "seeds": seeds,
        "alpha": alpha,
        "timestamp": datetime.now().isoformat(),
        "per_seed": {str(s): r for s, r in all_results.items()},
        "summary": {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
            for m, v in method_aucs.items() if v
        },
    }
    if celeb_aucs:
        summary["summary"]["Celeb-DF"] = {
            "mean": float(np.mean(celeb_aucs)),
            "std": float(np.std(celeb_aucs)),
            "min": float(np.min(celeb_aucs)),
            "max": float(np.max(celeb_aucs)),
            "n": len(celeb_aucs),
        }

    with open(results_dir / "seed_variance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {results_dir}/seed_variance_summary.json")


def run_alpha_experiment(args):
    """Run alpha ablation: train with different α values."""
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    seed = 42  # fixed seed for alpha experiment

    results_dir = Path("experiment_alpha")
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    for alpha in alphas:
        print(f"\n{'#'*70}")
        print(f"#  ALPHA = {alpha}")
        print(f"{'#'*70}")

        tag = f"a{int(alpha*100):03d}"

        # Checkpoint names
        struct_ckpt = str(results_dir / f"struct_{tag}.pt")
        freq_ckpt = str(results_dir / f"freq_{tag}.pt")
        local_ckpt = str(results_dir / f"local_{tag}.pt")

        # Train all three branches with this alpha
        train_structural(args, seed, alpha, struct_ckpt)
        train_frequency(args, seed, alpha, freq_ckpt)
        train_local(args, seed, alpha, local_ckpt)

        # Evaluate on FF++
        ff_output = str(results_dir / f"eval_ff_{tag}.json")
        evaluate_combined(
            struct_ckpt, freq_ckpt, local_ckpt,
            args.eval_real, args.eval_fake_ff,
            ff_output, max_images=5000)

        # Evaluate on Celeb-DF
        celeb_output = str(results_dir / f"eval_celeb_{tag}.json")
        evaluate_combined(
            struct_ckpt, freq_ckpt, local_ckpt,
            args.eval_real_celeb, [args.eval_fake_celeb],
            celeb_output, max_images=5000)

        ff_results = parse_eval_results(ff_output)
        celeb_results = parse_eval_results(celeb_output)

        all_results[alpha] = {
            "ff": ff_results,
            "celeb": celeb_results,
        }

    # ==========================================
    # Summary: alpha ablation table
    # ==========================================
    print(f"\n\n{'='*80}")
    print("ALPHA ABLATION: COMBINED AUC")
    print(f"{'='*80}\n")

    methods_ff = ["Deepfakes", "Face2Face", "FaceSwap",
                   "NeuralTextures", "FaceShifter"]

    header = f"{'Alpha':>6s}"
    for m in methods_ff:
        header += f" {m[:8]:>8s}"
    header += f" {'CelebDF':>8s}"
    print(header)
    print("-" * len(header))

    for alpha in alphas:
        res = all_results[alpha]
        line = f"{alpha:>6.1f}"
        for method in methods_ff:
            if res["ff"] and method in res["ff"]:
                auc = res["ff"][method]["combined"]["auc"]
                line += f" {auc:>8.3f}"
            else:
                line += f" {'N/A':>8s}"
        if res["celeb"]:
            for key in res["celeb"]:
                auc = res["celeb"][key]["combined"]["auc"]
                line += f" {auc:>8.3f}"
                break
        else:
            line += f" {'N/A':>8s}"
        print(line)

    # ==========================================
    # Structural-only AUC (isolates α effect)
    # ==========================================
    print(f"\n{'='*80}")
    print("ALPHA ABLATION: STRUCTURAL BRANCH ONLY (isolates α effect)")
    print(f"{'='*80}\n")

    header = f"{'Alpha':>6s}"
    for m in methods_ff:
        header += f" {m[:8]:>8s}"
    header += f" {'CelebDF':>8s}"
    print(header)
    print("-" * len(header))

    for alpha in alphas:
        res = all_results[alpha]
        line = f"{alpha:>6.1f}"
        for method in methods_ff:
            if res["ff"] and method in res["ff"] and "structural" in res["ff"][method]:
                auc = res["ff"][method]["structural"]["auc"]
                line += f" {auc:>8.3f}"
            else:
                line += f" {'N/A':>8s}"
        if res["celeb"]:
            for key in res["celeb"]:
                if "structural" in res["celeb"][key]:
                    auc = res["celeb"][key]["structural"]["auc"]
                    line += f" {auc:>8.3f}"
                else:
                    line += f" {'N/A':>8s}"
                break
        else:
            line += f" {'N/A':>8s}"
        print(line)

    # ==========================================
    # Dilution analysis: sparse vs diffuse
    # ==========================================
    print(f"\n{'='*80}")
    print("DILUTION ANALYSIS: sparse vs diffuse violations")
    print("Prediction: α matters more for sparse (few patches affected)")
    print(f"{'='*80}\n")

    # Categorize methods by violation sparsity
    sparse_methods = ["Deepfakes", "FaceSwap"]  # boundary at ~10/256 patches
    diffuse_methods = ["Face2Face", "FaceShifter"]  # whole-face reenactment
    mixed_methods = ["NeuralTextures"]  # mouth region only

    for category, methods, desc in [
        ("SPARSE", sparse_methods, "Face swap boundaries: ~10/256 patches affected"),
        ("DIFFUSE", diffuse_methods, "Full-face reenactment: most patches affected"),
        ("MIXED", mixed_methods, "Mouth region: ~30/256 patches affected"),
    ]:
        print(f"\n  {category}: {desc}")
        print(f"  {'Alpha':>6s}", end="")
        for m in methods:
            print(f" {m[:10]:>10s}", end="")
        print()

        for alpha in alphas:
            res = all_results[alpha]
            line = f"  {alpha:>6.1f}"
            for method in methods:
                if res["ff"] and method in res["ff"] and "structural" in res["ff"][method]:
                    auc = res["ff"][method]["structural"]["auc"]
                    line += f" {auc:>10.3f}"
                else:
                    line += f" {'N/A':>10s}"
            print(line)

    # Compute α sensitivity per category
    print(f"\n  α SENSITIVITY (max AUC - min AUC across α values):")
    for category, methods in [("Sparse", sparse_methods),
                               ("Diffuse", diffuse_methods),
                               ("Mixed", mixed_methods)]:
        sensitivities = []
        for method in methods:
            aucs = []
            for alpha in alphas:
                res = all_results[alpha]
                if res["ff"] and method in res["ff"] and "structural" in res["ff"][method]:
                    aucs.append(res["ff"][method]["structural"]["auc"])
            if aucs:
                sensitivities.append(max(aucs) - min(aucs))
        if sensitivities:
            avg_sens = np.mean(sensitivities)
            print(f"    {category:<10s}: ±{avg_sens:.3f} AUC")

    print(f"\n  If sparse >> diffuse, the dilution argument is empirically supported.")
    print(f"  If sparse ≈ diffuse, the max term helps uniformly (or not at all).")

    # Save
    summary = {
        "experiment": "alpha_ablation",
        "alphas": alphas,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "per_alpha": {str(a): r for a, r in all_results.items()},
    }
    with open(results_dir / "alpha_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {results_dir}/alpha_ablation_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run variance and ablation experiments")
    parser.add_argument("--experiment", choices=["seeds", "alpha", "both"],
                        required=True)

    # Training data
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--fake_dirs", nargs="+", required=True)
    parser.add_argument("--corruption_dirs", nargs="+", default=[])
    parser.add_argument("--pretrained", required=True)

    # Eval data
    parser.add_argument("--eval_real", required=True,
                        help="FF++ test real images")
    parser.add_argument("--eval_fake_ff", nargs="+", required=True,
                        help="FF++ test fake dirs")
    parser.add_argument("--eval_real_celeb", required=True,
                        help="Celeb-DF real images")
    parser.add_argument("--eval_fake_celeb", required=True,
                        help="Celeb-DF fake images")

    args = parser.parse_args()

    if args.experiment in ["seeds", "both"]:
        run_seed_experiment(args)
    if args.experiment in ["alpha", "both"]:
        run_alpha_experiment(args)
