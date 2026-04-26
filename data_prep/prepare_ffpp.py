"""
Download and prepare FaceForensics++ face crops for the vision constraint network.

FaceForensics++ requires access credentials. To get them:
  1. Go to https://github.com/ondyari/FaceForensics
  2. Fill out the Google Form linked in the README
  3. You'll receive a download script or access credentials via email

This script handles the full pipeline:
  1. Download FF++ videos (original + manipulated)
  2. Extract frames from videos
  3. Detect and crop faces (224×224, aligned)
  4. Organise into directories ready for train_vision.py and eval_vision.py

Usage:
  # Step 1: Download (need credentials from FF++ authors)
  python prepare_ffpp.py download --credentials /path/to/download_script.py

  # Step 2: Extract frames + crop faces (after download completes)
  python prepare_ffpp.py extract --data_dir ./ffpp_data

  # Step 3: Quick alternative — use a small public face dataset instead
  #         (good enough for the diagnostic, not for the full experiment)
  python prepare_ffpp.py quick_start

Install:
  pip install opencv-python pillow requests tqdm facenet-pytorch
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

FFPP_SUBSETS = {
    "real": "original_sequences/youtube",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}

# Compression level: raw, c23 (light compression), c40 (heavy compression)
# c23 is the standard benchmark setting
COMPRESSION = "c23"


# ============================================================
# Frame extraction
# ============================================================

def extract_frames(video_dir, output_dir, every_n=10, max_frames_per_video=50):
    """
    Extract every Nth frame from all videos in a directory.
    """
    import cv2
    from tqdm import tqdm

    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in video_exts)

    print(f"  Found {len(videos)} videos in {video_dir}")
    total_frames = 0

    for video_path in tqdm(videos, desc="  Extracting frames"):
        video_name = video_path.stem
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            continue

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0 and saved < max_frames_per_video:
                out_path = output_dir / f"{video_name}_frame{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
                total_frames += 1

            frame_idx += 1

        cap.release()

    print(f"  Extracted {total_frames} frames total")
    return total_frames


# ============================================================
# Face detection and cropping
# ============================================================

def crop_faces(frame_dir, output_dir, target_size=224, margin=0.3,
               batch_size=32):
    """
    Detect faces in frames and save aligned 224×224 crops.
    Uses MTCNN from facenet-pytorch for reliable detection.
    """
    import cv2
    import torch
    from PIL import Image
    from tqdm import tqdm

    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        print("  Installing facenet-pytorch for face detection...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "facenet-pytorch", "--quiet"])
        from facenet_pytorch import MTCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(
        image_size=target_size,
        margin=int(target_size * margin),
        device=device,
        select_largest=True,
        post_process=False,  # return PIL-friendly uint8
    )

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    frame_paths = sorted(p for p in frame_dir.rglob("*") if p.suffix.lower() in exts)
    print(f"  Found {len(frame_paths)} frames in {frame_dir}")

    detected = 0
    failed = 0

    for path in tqdm(frame_paths, desc="  Cropping faces"):
        try:
            img = Image.open(path).convert("RGB")
            face = mtcnn(img)

            if face is not None:
                # face is a (3, 224, 224) tensor in [0, 255]
                face_np = face.permute(1, 2, 0).numpy().astype("uint8")
                face_pil = Image.fromarray(face_np)
                out_path = output_dir / f"{path.stem}_face.jpg"
                face_pil.save(str(out_path), quality=95)
                detected += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1

    print(f"  Detected: {detected}, Failed: {failed}")
    return detected


# ============================================================
# Quick start: download a small public face dataset
# ============================================================

def quick_start(output_dir="ffpp_quick"):
    """
    Download a small set of face images for running the diagnostic.

    Tries sources in order:
      1. Hugging Face Hub (uses the `datasets` library you already have)
      2. LFW direct download mirrors

    This is NOT a substitute for FF++ for the full experiment, but it's
    enough to run vision_diagnostic.py on real faces and verify the pipeline.
    """
    from PIL import Image
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    faces_dir = output_dir / "faces_224"

    # Skip if already done
    if faces_dir.exists() and len(list(faces_dir.glob("*.jpg"))) > 100:
        n = len(list(faces_dir.glob("*.jpg")))
        print(f"  Face crops already exist: {n} images in {faces_dir}")
        _print_ready(faces_dir)
        return faces_dir

    faces_dir.mkdir(parents=True, exist_ok=True)

    # --- Method 1: Hugging Face Hub (most reliable) ---
    success = False
    try:
        print("Downloading face dataset from Hugging Face Hub...")
        print("  (Using the `datasets` library you already have installed)")
        from datasets import load_dataset

        # LFW on Hugging Face — streams images directly, no archive needed
        ds = load_dataset("tonyassi/celebrity-1000", split="train",
                          trust_remote_code=True)

        print(f"  Loaded {len(ds)} images. Resizing to 224×224...")
        count = 0
        for i, item in enumerate(tqdm(ds, desc="  Processing")):
            try:
                img = item["image"].convert("RGB")
                img = img.resize((224, 224), Image.LANCZOS)
                img.save(str(faces_dir / f"face_{count:05d}.jpg"), quality=95)
                count += 1
            except Exception:
                continue

        if count > 100:
            print(f"  Saved {count} face crops to {faces_dir}")
            success = True
        else:
            print(f"  Only got {count} images, trying next source...")

    except Exception as e:
        print(f"  Hugging Face download failed: {e}")
        print(f"  Trying alternative sources...")

    # --- Method 2: Try another HF dataset ---
    if not success:
        try:
            print("\nTrying alternative: olivierdehaene/face-synthetics subset...")
            from datasets import load_dataset

            ds = load_dataset("nielsr/lfw", split="train",
                              trust_remote_code=True)

            print(f"  Loaded {len(ds)} images. Resizing to 224×224...")
            count = len(list(faces_dir.glob("*.jpg")))
            for i, item in enumerate(tqdm(ds, desc="  Processing")):
                try:
                    img = item["image"].convert("RGB")
                    img = img.resize((224, 224), Image.LANCZOS)
                    img.save(str(faces_dir / f"face_{count:05d}.jpg"), quality=95)
                    count += 1
                except Exception:
                    continue

            if count > 100:
                print(f"  Saved {count} face crops to {faces_dir}")
                success = True

        except Exception as e:
            print(f"  Alternative download also failed: {e}")

    # --- Method 3: LFW direct download with multiple mirrors ---
    if not success:
        import urllib.request
        import tarfile

        mirrors = [
            "https://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
            "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
        ]

        archive_path = output_dir / "lfw-funneled.tgz"
        downloaded = False

        if archive_path.exists():
            downloaded = True
        else:
            for url in mirrors:
                try:
                    print(f"\nTrying direct download: {url}")

                    class DownloadProgress:
                        def __init__(self):
                            self.pbar = None
                        def __call__(self, block_num, block_size, total_size):
                            if self.pbar is None:
                                self.pbar = tqdm(total=total_size, unit='B',
                                                 unit_scale=True, desc="  Downloading")
                            self.pbar.update(block_size)

                    urllib.request.urlretrieve(url, str(archive_path),
                                               DownloadProgress())
                    print()
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  Failed: {e}")

        if downloaded:
            # Extract
            extract_dir = output_dir / "lfw_raw"
            if not extract_dir.exists():
                print("Extracting archive...")
                with tarfile.open(str(archive_path), "r:gz") as tar:
                    tar.extractall(str(extract_dir))

            img_paths = sorted(extract_dir.rglob("*.jpg"))
            print(f"  Found {len(img_paths)} images. Resizing...")

            count = len(list(faces_dir.glob("*.jpg")))
            for p in tqdm(img_paths, desc="  Resizing"):
                try:
                    img = Image.open(p).convert("RGB")
                    img = img.resize((224, 224), Image.LANCZOS)
                    img.save(str(faces_dir / f"face_{count:05d}.jpg"), quality=95)
                    count += 1
                except Exception:
                    continue
            success = count > 100

    # --- Report ---
    n_faces = len(list(faces_dir.glob("*.jpg")))
    if n_faces > 100:
        print(f"\n  Total face crops: {n_faces}")
        _print_ready(faces_dir)
        return faces_dir
    else:
        print(f"\n  ERROR: Could not download enough face images ({n_faces} found).")
        print(f"  Manual options:")
        print(f"    1. pip install datasets && python -c \"")
        print(f"       from datasets import load_dataset;")
        print(f"       ds = load_dataset('tonyassi/celebrity-1000', split='train');")
        print(f"       print(len(ds))\"")
        print(f"    2. Download LFW manually from https://vis-www.cs.umass.edu/lfw/")
        print(f"    3. Use any folder of 600+ face photos (224×224 JPEGs)")
        return None


def _print_ready(faces_dir):
    print(f"\n{'='*60}")
    print(f"Ready! Run the diagnostic with:")
    print(f"  python vision_diagnostic.py --data_dir {faces_dir}")
    print(f"{'='*60}")


# ============================================================
# Full FF++ download
# ============================================================

def download_ffpp(credentials_script, output_dir="ffpp_data",
                   compression="c23", subsets=None):
    """
    Download FaceForensics++ using the official download script.

    The FF++ authors provide a Python download script after you
    request access. This function wraps that script.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subsets is None:
        subsets = list(FFPP_SUBSETS.keys())

    credentials_script = Path(credentials_script)
    if not credentials_script.exists():
        print(f"ERROR: Credentials script not found: {credentials_script}")
        print(f"\nTo get access to FaceForensics++:")
        print(f"  1. Visit https://github.com/ondyari/FaceForensics")
        print(f"  2. Fill out the access request form")
        print(f"  3. You'll receive a download script via email")
        print(f"\nAlternatively, run: python prepare_ffpp.py quick_start")
        print(f"  to use a small public dataset for initial testing.")
        return

    for subset in subsets:
        subset_path = FFPP_SUBSETS.get(subset)
        if not subset_path:
            print(f"  Unknown subset: {subset}")
            continue

        print(f"\nDownloading {subset} ({compression})...")
        subset_dir = output_dir / subset

        cmd = [
            sys.executable, str(credentials_script),
            str(output_dir),
            "-d", subset_path,
            "-c", compression,
            "-t", "videos",
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"  Downloaded to {subset_dir}")
        except subprocess.CalledProcessError as e:
            print(f"  Download failed for {subset}: {e}")
        except FileNotFoundError:
            print(f"  Could not run download script. Check the path.")


def prepare_ffpp(data_dir, output_dir="ffpp_prepared", every_n=10):
    """
    Full pipeline: extract frames from downloaded FF++ videos, crop faces.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    for subset_name, subset_path in FFPP_SUBSETS.items():
        video_dir = data_dir / subset_path / COMPRESSION
        if not video_dir.exists():
            # Try without compression subdirectory
            video_dir = data_dir / subset_path
        if not video_dir.exists():
            # Try flat structure
            video_dir = data_dir / subset_name
        if not video_dir.exists():
            print(f"  Skipping {subset_name}: directory not found")
            print(f"    (looked for {data_dir / subset_path / COMPRESSION})")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {subset_name}")
        print(f"{'='*60}")

        # Extract frames
        frame_dir = output_dir / "frames" / subset_name
        if frame_dir.exists() and len(list(frame_dir.glob("*.jpg"))) > 0:
            n = len(list(frame_dir.glob("*.jpg")))
            print(f"  Frames already extracted: {n}")
        else:
            extract_frames(video_dir, frame_dir, every_n=every_n)

        # Crop faces
        face_dir = output_dir / "faces" / subset_name
        if face_dir.exists() and len(list(face_dir.glob("*.jpg"))) > 0:
            n = len(list(face_dir.glob("*.jpg")))
            print(f"  Face crops already exist: {n}")
        else:
            crop_faces(frame_dir, face_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"PREPARATION COMPLETE")
    print(f"{'='*60}")

    for subset_name in FFPP_SUBSETS:
        face_dir = output_dir / "faces" / subset_name
        if face_dir.exists():
            n = len(list(face_dir.glob("*.jpg")))
            print(f"  {subset_name:<20s}: {n:>6d} face crops")

    real_dir = output_dir / "faces" / "real"
    if real_dir.exists():
        print(f"\nRun the diagnostic:")
        print(f"  python vision_diagnostic.py --data_dir {real_dir}")
        print(f"\nRun full training:")
        print(f"  python train_vision.py --data_dir {real_dir}")
        print(f"\nRun deepfake evaluation:")
        fake_dirs = [str(output_dir / "faces" / s)
                     for s in FFPP_SUBSETS if s != "real"]
        print(f"  python eval_vision.py \\")
        print(f"    --checkpoint vision_constraint_best.pt \\")
        print(f"    --real_dir {real_dir} \\")
        print(f"    --fake_dirs {' '.join(fake_dirs)}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare FaceForensics++ for vision constraint network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with public data (no credentials needed):
  python prepare_ffpp.py quick_start

  # Download FF++ (needs credentials):
  python prepare_ffpp.py download --credentials path/to/download_script.py

  # Prepare downloaded FF++ data:
  python prepare_ffpp.py extract --data_dir ./ffpp_data

  # All-in-one with existing FF++ videos:
  python prepare_ffpp.py extract --data_dir ./ffpp_data --output_dir ./ffpp_prepared
        """)

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Quick start
    p_quick = subparsers.add_parser("quick_start",
        help="Download a small public face dataset for initial testing")
    p_quick.add_argument("--output_dir", type=str, default="ffpp_quick",
                         help="Where to save the data")

    # Download
    p_dl = subparsers.add_parser("download",
        help="Download FaceForensics++ (needs access credentials)")
    p_dl.add_argument("--credentials", type=str, required=True,
                      help="Path to the FF++ download script (received via email)")
    p_dl.add_argument("--output_dir", type=str, default="ffpp_data")
    p_dl.add_argument("--compression", type=str, default="c23",
                      choices=["raw", "c23", "c40"])
    p_dl.add_argument("--subsets", nargs="+", default=None,
                      choices=list(FFPP_SUBSETS.keys()))

    # Extract
    p_ext = subparsers.add_parser("extract",
        help="Extract frames and crop faces from downloaded FF++ videos")
    p_ext.add_argument("--data_dir", type=str, required=True,
                       help="Root directory of downloaded FF++ data")
    p_ext.add_argument("--output_dir", type=str, default="ffpp_prepared")
    p_ext.add_argument("--every_n", type=int, default=10,
                       help="Extract every Nth frame (default: 10)")

    args = parser.parse_args()

    if args.command == "quick_start":
        quick_start(args.output_dir)
    elif args.command == "download":
        download_ffpp(args.credentials, args.output_dir,
                      args.compression, args.subsets)
    elif args.command == "extract":
        prepare_ffpp(args.data_dir, args.output_dir, args.every_n)
    else:
        parser.print_help()
        print("\n  Tip: run 'python prepare_ffpp.py quick_start' to get started fast.")
