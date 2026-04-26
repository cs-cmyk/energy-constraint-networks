"""
Create corruption-based training pairs from real face images.

Takes real face images, generates corrupted versions, and saves them
in paired folders ready for train_paired.py.

Usage:
  python create_corruption_pairs.py \
    --real_dir ffpp_quick/faces_224 \
    --output_dir ff_c23_faces \
    --num_images 6000

Creates:
  ff_c23_faces/train_celeb_real/       ← 6000 real celebrity faces
  ff_c23_faces/train_celeb_corrupted/  ← 6000 corrupted versions (matching names)

Then train with all pairs:
  python train_paired.py \
    --real_dir ff_c23_faces/train_original \
    --fake_dirs ff_c23_faces/train_Deepfakes ff_c23_faces/train_Face2Face \
                ff_c23_faces/train_FaceSwap ff_c23_faces/train_NeuralTextures \
                ff_c23_faces/train_FaceShifter ff_c23_faces/train_celeb_corrupted
"""

import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def create_pairs(args):
    exts = {".jpg", ".jpeg", ".png"}
    real_paths = sorted(
        p for p in Path(args.real_dir).rglob("*") if p.suffix.lower() in exts)

    print(f"Source images: {len(real_paths)}")

    # Select subset
    random.seed(42)
    selected = random.sample(real_paths, min(args.num_images, len(real_paths)))
    print(f"Selected: {len(selected)}")

    # Output dirs
    out_real = Path(args.output_dir) / "train_celeb_real"
    out_corrupt = Path(args.output_dir) / "train_celeb_corrupted"
    out_real.mkdir(parents=True, exist_ok=True)
    out_corrupt.mkdir(parents=True, exist_ok=True)

    if len(list(out_real.glob("*"))) > 0:
        print(f"Output dirs already exist. Use --force to overwrite.")
        if not args.force:
            return
        shutil.rmtree(out_real)
        shutil.rmtree(out_corrupt)
        out_real.mkdir(parents=True, exist_ok=True)
        out_corrupt.mkdir(parents=True, exist_ok=True)

    from vision_corruptions import apply_corruption, set_donor_pool
    set_donor_pool([str(p) for p in selected])

    corruption_types = ["texture_splice", "lighting_shift"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    count = 0
    for i, path in enumerate(tqdm(selected, desc="Creating pairs")):
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img)
            img_224 = img.resize((224, 224), Image.LANCZOS)

            for k, ctype in enumerate(corruption_types):
                # Consistent name for pairing
                name = f"celeb_{i:05d}_{ctype}.jpg"

                # Save real (same real image for each corruption)
                img_224.save(str(out_real / name), quality=95)

                # Generate and save corrupted version
                corrupted = apply_corruption(img_tensor, ctype, idx=i, k=k)
                corr_img = (corrupted.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                corr_pil = Image.fromarray(corr_img)
                corr_pil.save(str(out_corrupt / name), quality=95)

                count += 1
        except Exception as e:
            continue

    print(f"\nCreated {count} pairs")
    print(f"  Real:      {out_real} ({len(list(out_real.glob('*')))} images)")
    print(f"  Corrupted: {out_corrupt} ({len(list(out_corrupt.glob('*')))} images)")

    # Print combined training command
    print(f"\nTrain with FF++ pairs + corruption pairs:")
    print(f"  python train_paired.py \\")
    print(f"    --real_dir {Path(args.output_dir) / 'train_original'} \\")
    print(f"    --fake_dirs {Path(args.output_dir) / 'train_Deepfakes'} \\")
    print(f"                {Path(args.output_dir) / 'train_Face2Face'} \\")
    print(f"                {Path(args.output_dir) / 'train_FaceSwap'} \\")
    print(f"                {Path(args.output_dir) / 'train_NeuralTextures'} \\")
    print(f"                {Path(args.output_dir) / 'train_FaceShifter'} \\")
    print(f"                {out_corrupt}")

    print(f"\n  Note: also add {out_real} images to your real training dir:")
    print(f"    cp {out_real}/* {Path(args.output_dir) / 'train_original'}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create corruption-based training pairs from real faces")
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory of real face images")
    parser.add_argument("--output_dir", type=str, default="ff_c23_faces",
                        help="Base output directory")
    parser.add_argument("--num_images", type=int, default=6000,
                        help="Number of pairs to create")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output")
    args = parser.parse_args()
    create_pairs(args)
