"""
Split FF++ face crops into train/test directories.

Logic:
  1. Select N real images from original/
  2. Move them to train_original/
  3. For each fake method, find the matching fakes for those same
     real images and move them to train_<method>/
  4. Whatever remains in the original directories is the test set.

Usage:
  python split_data.py \
    --real_dir ff_c23_faces/original \
    --fake_dirs ff_c23_faces/Deepfakes ff_c23_faces/Face2Face ff_c23_faces/FaceSwap \
    --num_train 6000
"""

import argparse
import random
import re
from pathlib import Path
from collections import defaultdict
import shutil


def parse_frame_info(filename):
    """
    Parse video ID and frame number from filename.
    '003_f0100.jpg'     → video_ids=['003'], frame=100
    '003_178_f0100.jpg' → video_ids=['003','178'], frame=100
    """
    stem = Path(filename).stem
    frame_match = re.search(r'_f(\d+)', stem)
    frame_num = int(frame_match.group(1)) if frame_match else -1

    if frame_match:
        video_part = stem[:frame_match.start()]
    else:
        video_part = stem

    video_ids = [v for v in video_part.split('_') if v.isdigit()]
    return video_ids, frame_num


def find_matching_fakes(real_paths, fake_dir):
    """
    For each real image, find its matching fake in fake_dir.
    Matches by video ID and frame number.

    Returns: list of (real_path, fake_path) pairs
    """
    exts = {".jpg", ".jpeg", ".png"}
    fake_paths = sorted(
        p for p in Path(fake_dir).rglob("*") if p.suffix.lower() in exts)

    # Index fakes by (video_id, frame)
    fake_index = defaultdict(dict)  # video_id → {frame → path}
    for fp in fake_paths:
        vids, frame = parse_frame_info(fp.name)
        for vid in vids:
            fake_index[vid][frame] = fp

    pairs = []
    for rp in real_paths:
        real_vids, real_frame = parse_frame_info(rp.name)

        for vid in real_vids:
            # Exact match: same video, same frame
            if vid in fake_index and real_frame in fake_index[vid]:
                pairs.append((rp, fake_index[vid][real_frame]))
                break

            # Close match: same video, closest frame
            if vid in fake_index and fake_index[vid]:
                frames = sorted(fake_index[vid].keys())
                closest = min(frames, key=lambda f: abs(f - real_frame))
                if abs(closest - real_frame) <= 20:
                    pairs.append((rp, fake_index[vid][closest]))
                    break

    return pairs


def split(args):
    exts = {".jpg", ".jpeg", ".png"}
    base_dir = Path(args.real_dir).parent

    # Check if already split
    train_real_dir = base_dir / "train_original"
    if train_real_dir.exists() and len(list(train_real_dir.glob("*"))) > 0:
        n = len(list(train_real_dir.glob("*")))
        print(f"Train split already exists: {train_real_dir} ({n} images)")
        print(f"Delete train_* directories to redo, or use --force")
        if not args.force:
            return

    # Collect real images
    real_paths = sorted(
        p for p in Path(args.real_dir).rglob("*") if p.suffix.lower() in exts)
    print(f"Real images: {len(real_paths)}")

    # Select training real images
    random.seed(42)
    selected_real = random.sample(real_paths, min(args.num_train, len(real_paths)))
    print(f"Selected {len(selected_real)} real images for training")

    # For each fake method, find matching fakes
    methods = []
    method_matches = {}

    for fd in args.fake_dirs:
        fd = Path(fd)
        method = fd.name
        methods.append(method)

        n_total = len(list(fd.rglob("*.jpg")))
        print(f"\n  {method}: {n_total} total images")

        pairs = find_matching_fakes(selected_real, fd)
        method_matches[method] = pairs
        print(f"    Matched: {len(pairs)} / {len(selected_real)}")

    # Move real images
    train_real_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for rp in selected_real:
        if rp.exists():
            shutil.move(str(rp), str(train_real_dir / rp.name))
            moved += 1
    print(f"\nMoved {moved} real images → {train_real_dir}")

    # Move matched fakes
    for method in methods:
        train_fake_dir = base_dir / f"train_{method}"
        train_fake_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        for _, fp in method_matches[method]:
            if fp.exists():
                shutil.move(str(fp), str(train_fake_dir / fp.name))
                moved += 1
        print(f"Moved {moved} fake images → {train_fake_dir}")

    # Summary
    print(f"\n{'='*60}")
    print("SPLIT COMPLETE")
    print(f"{'='*60}")

    print(f"\nTraining:")
    n = len(list(train_real_dir.glob("*")))
    print(f"  train_original: {n}")
    for method in methods:
        td = base_dir / f"train_{method}"
        n = len(list(td.glob("*")))
        print(f"  train_{method}: {n}")

    print(f"\nTest (remaining):")
    n = len(list(Path(args.real_dir).rglob("*.jpg")))
    print(f"  original: {n}")
    for fd in args.fake_dirs:
        n = len(list(Path(fd).rglob("*.jpg")))
        print(f"  {Path(fd).name}: {n}")

    # Print next commands
    train_dirs = " ".join(str(base_dir / f"train_{m}") for m in methods)
    print(f"\nTrain:")
    print(f"  python train_paired.py \\")
    print(f"    --real_dir {train_real_dir} \\")
    print(f"    --fake_dirs {train_dirs}")

    print(f"\nEvaluate:")
    test_dirs = " ".join(args.fake_dirs)
    print(f"  python eval_vision.py \\")
    print(f"    --checkpoint paired_constraint_best.pt \\")
    print(f"    --real_dir {args.real_dir} \\")
    print(f"    --fake_dirs {test_dirs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split FF++ face crops into train/test directories")
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory of real face crops")
    parser.add_argument("--fake_dirs", type=str, nargs="+", required=True,
                        help="Directories of fake face crops")
    parser.add_argument("--num_train", type=int, default=6000,
                        help="Number of real images to select for training (default: 6000)")
    parser.add_argument("--force", action="store_true",
                        help="Redo split even if train_* dirs exist")
    args = parser.parse_args()
    split(args)
