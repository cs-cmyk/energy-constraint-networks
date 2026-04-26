"""
Extract frames from FF++ videos and crop faces using OpenCV's DNN face detector.
No extra dependencies — uses only opencv-python which is already installed.

Usage:
  python extract_faces.py --base_dir ff_c23_data/FaceForensics++_C23

  # Process specific subsets only:
  python extract_faces.py --base_dir ff_c23_data/FaceForensics++_C23 \
    --subsets original Deepfakes

  # Quick test with fewer videos:
  python extract_faces.py --base_dir ff_c23_data/FaceForensics++_C23 \
    --subsets original Deepfakes --max_videos 10
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


# OpenCV DNN face detector (ships with opencv-python, no download needed)
def get_face_detector():
    """
    Get OpenCV's DNN face detector using the built-in Haar cascade
    as primary, with DNN as upgrade if the caffemodel is available.
    """
    # Use Haar cascade — always available with OpenCV
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    return detector


def detect_and_crop_face(image, detector, target_size=224, margin=0.3):
    """
    Detect the largest face in an image and return a cropped, resized version.

    Args:
        image: BGR numpy array
        detector: OpenCV CascadeClassifier
        target_size: output size (224×224)
        margin: fractional margin around the detected face

    Returns: (target_size, target_size, 3) BGR numpy array, or None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]

    # Detect faces
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        # Fallback: if no face detected, try with looser params
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

    if len(faces) == 0:
        # Last resort: center crop (FF++ videos are face-centric)
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        crop = image[y1:y1+size, x1:x1+size]
        return cv2.resize(crop, (target_size, target_size))

    # Take the largest face
    areas = [fw * fh for (_, _, fw, fh) in faces]
    x, y, fw, fh = faces[np.argmax(areas)]

    # Add margin
    mx = int(fw * margin)
    my = int(fh * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + fw + mx)
    y2 = min(h, y + fh + my)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (target_size, target_size))


def process_subset(base_dir, subset, output_dir, detector,
                    every_n=10, max_frames_per_video=50,
                    max_videos=None):
    """
    Extract frames from videos and crop faces for one FF++ subset.
    """
    video_dir = Path(base_dir) / subset
    face_dir = Path(output_dir) / subset
    face_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = len(list(face_dir.glob("*.jpg")))
    if existing > 100:
        print(f"  {subset}: {existing} face crops already exist, skipping.")
        return existing

    # Find videos
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in video_exts)

    if max_videos:
        videos = videos[:max_videos]

    if not videos:
        print(f"  {subset}: no videos found in {video_dir}")
        return 0

    print(f"  {subset}: processing {len(videos)} videos...")
    total_faces = 0
    failed_detect = 0

    for video_path in tqdm(videos, desc=f"  {subset}"):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        video_name = video_path.stem
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0 and saved < max_frames_per_video:
                face_crop = detect_and_crop_face(frame, detector)

                if face_crop is not None:
                    out_path = face_dir / f"{video_name}_f{frame_idx:04d}.jpg"
                    cv2.imwrite(str(out_path), face_crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    total_faces += 1
                    saved += 1
                else:
                    failed_detect += 1

            frame_idx += 1

        cap.release()

    print(f"  {subset}: {total_faces} face crops saved, "
          f"{failed_detect} detection failures")
    return total_faces


def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops from FaceForensics++ videos")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to FaceForensics++_C23 directory")
    parser.add_argument("--output_dir", type=str, default="ff_c23_faces",
                        help="Output directory for face crops")
    parser.add_argument("--subsets", nargs="+",
                        default=["original", "Deepfakes", "Face2Face",
                                 "FaceSwap", "NeuralTextures", "FaceShifter"],
                        help="Which subsets to process")
    parser.add_argument("--every_n", type=int, default=10,
                        help="Extract every Nth frame")
    parser.add_argument("--max_frames", type=int, default=50,
                        help="Max frames per video")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Max videos per subset (for quick testing)")
    args = parser.parse_args()

    print(f"Base directory: {args.base_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Subsets: {args.subsets}")
    print(f"Every {args.every_n}th frame, max {args.max_frames} per video")

    detector = get_face_detector()
    print(f"Face detector: OpenCV Haar cascade\n")

    results = {}
    for subset in args.subsets:
        print(f"\n{'='*60}")
        n = process_subset(args.base_dir, subset, args.output_dir, detector,
                           every_n=args.every_n,
                           max_frames_per_video=args.max_frames,
                           max_videos=args.max_videos)
        results[subset] = n

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for subset, count in results.items():
        print(f"  {subset:<20s}: {count:>6d} face crops")

    total = sum(results.values())
    print(f"  {'TOTAL':<20s}: {total:>6d}")

    real_dir = Path(args.output_dir) / "original"
    if real_dir.exists():
        fake_dirs = [str(Path(args.output_dir) / s)
                     for s in args.subsets if s != "original"
                     and (Path(args.output_dir) / s).exists()]
        print(f"\nRun evaluation:")
        print(f"  python eval_vision.py \\")
        print(f"    --checkpoint vision_constraint_best.pt \\")
        print(f"    --real_dir {real_dir} \\")
        print(f"    --fake_dirs {' '.join(fake_dirs)} \\")
        print(f"    --heatmaps --max_images 500")


if __name__ == "__main__":
    main()
