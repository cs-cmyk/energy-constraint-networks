"""
Vision corruption strategies for the constraint network.

Analogous to nl_adaptation.py corruptions, but for face images.
Each corruption breaks a specific structural property of real faces:

Phase 1 (implemented here — for the minimum viable experiment):
  1. Texture boundary splice → regional texture/identity compatibility
  2. Lighting direction shift → global lighting consistency

Phase 2 (to implement after Phase 1 validates):
  3. Colour temperature shift → colour consistency
  4. Frequency injection → frequency spectrum consistency
  5. Bilateral symmetry break → geometric consistency
  6. Skin/material inconsistency → material consistency

Design principles (from the text corruption experience):
  - Each corruption implicitly defines a structural property
  - Corruption severity is controllable (0 = none, 1 = maximum)
  - Blending must be smooth enough that the network needs structural
    understanding, not just edge detection
  - Corruptions target the SAME properties as deepfake artifacts,
    but are applied to REAL images

Install:
  pip install opencv-python mediapipe numpy torch pillow
"""

import torch
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


# ============================================================
# Corruption result container (mirrors nl_adaptation.CorruptionResult)
# ============================================================

@dataclass
class VisionCorruptionResult:
    """Result of a visual corruption."""
    image: np.ndarray  # (H, W, 3) uint8 BGR
    corruption_type: str
    corrupted_region: Optional[np.ndarray] = None  # (H, W) binary mask
    severity: float = 1.0
    metadata: dict = field(default_factory=dict)


# ============================================================
# Face region utilities
# ============================================================

def get_face_landmarks_mediapipe(image_rgb):
    """
    Get face mesh landmarks using MediaPipe.
    Returns 478 (x, y) landmarks normalized to image dimensions,
    or None if no face detected.

    Handles both the legacy API (mediapipe < 0.10.8, mp.solutions.face_mesh)
    and the new task-based API (mediapipe >= 0.10.8).
    Falls back to None if MediaPipe is unavailable or detection fails.
    """
    h, w = image_rgb.shape[:2]

    try:
        import mediapipe as mp

        # --- Try legacy API first (mp.solutions.face_mesh) ---
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                results = face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0]
            points = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark],
                              dtype=np.float32)
            return points

        # --- Try new task-based API (mediapipe >= 0.10.8) ---
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request
            import tempfile
            import os

            # Download the face landmarker model if not cached
            model_path = os.path.join(tempfile.gettempdir(),
                                       "face_landmarker_v2_with_blendshapes.task")
            if not os.path.exists(model_path):
                model_url = ("https://storage.googleapis.com/mediapipe-models/"
                             "face_landmarker/face_landmarker/float16/latest/"
                             "face_landmarker.task")
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                except Exception:
                    # Can't download model — fall back to no landmarks
                    return None

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )

            with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                return None

            face_lms = result.face_landmarks[0]
            points = np.array([(lm.x * w, lm.y * h) for lm in face_lms],
                              dtype=np.float32)
            return points

    except (ImportError, Exception) as e:
        # MediaPipe not installed or something else went wrong.
        # Callers handle None by using fallback circular masks.
        return None


# MediaPipe face mesh region indices (approximate)
# These index groups define facial regions for corruption targeting
FACE_REGIONS = {
    "nose": [1, 2, 3, 4, 5, 6, 19, 94, 141, 168, 188, 196, 197,
             236, 239, 278, 279, 327, 343, 344, 370, 412, 419, 456],
    "left_cheek": [36, 47, 50, 101, 116, 117, 118, 119, 120, 123,
                   132, 135, 136, 137, 138, 139, 140, 147, 177, 187,
                   192, 205, 206, 207, 208, 209, 210],
    "right_cheek": [266, 280, 329, 330, 346, 347, 348, 349, 350,
                    352, 357, 361, 366, 367, 368, 369, 371, 376,
                    411, 416, 425, 426, 427, 428, 429, 430],
    "forehead": [10, 21, 54, 67, 68, 69, 70, 71, 103, 104, 108,
                 109, 151, 162, 193, 245, 251, 284, 297, 298, 299,
                 300, 301, 332, 333, 337, 338, 389, 417],
    "mouth": [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41,
              42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84,
              85, 86, 87, 88, 89, 91, 95, 146, 178, 179, 180, 181,
              183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291,
              292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314,
              315, 316, 317, 318, 319, 321, 324, 375, 402, 403, 404,
              405, 407, 408, 409, 415],
}


def create_region_mask(image_shape, landmarks, region_indices, dilate_px=5):
    """
    Create a binary mask for a facial region from landmark indices.
    Returns: (H, W) float32 mask in [0, 1] with smooth edges.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Get the convex hull of the region landmarks
    region_pts = landmarks[region_indices].astype(np.int32)
    hull = cv2.convexHull(region_pts)
    cv2.fillConvexPoly(mask, hull, 255)

    # Dilate slightly for smoother boundaries
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel)

    # Gaussian blur for smooth edges (prevents trivial edge detection)
    mask = cv2.GaussianBlur(mask, (21, 21), 7)
    return mask.astype(np.float32) / 255.0


def create_region_mask_simple(image_shape, center_y, center_x, radius):
    """
    Fallback: create a circular mask when landmarks aren't available.
    Returns: (H, W) float32 mask in [0, 1].
    """
    h, w = image_shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = np.clip(1.0 - (dist - radius) / (radius * 0.3), 0, 1)
    return mask.astype(np.float32)


# ============================================================
# Corruption 1: Texture Boundary Splice
# ============================================================

def corrupt_texture_splice(image, donor_image, landmarks=None,
                           donor_landmarks=None, region="nose",
                           severity=1.0, blend_mode="poisson"):
    """
    Replace a facial region with the same region from a donor face.
    Uses Poisson blending (or alpha blending fallback) for smooth boundaries.

    This is the closest analogue to face-swap deepfakes — a foreign region
    blended into a face. If the constraint network can't detect this,
    something fundamental isn't working.

    Args:
        image: (H, W, 3) uint8 BGR — the target face
        donor_image: (H, W, 3) uint8 BGR — the donor face
        landmarks: target face landmarks (478 points) or None
        donor_landmarks: donor face landmarks or None
        region: which facial region to splice ("nose", "mouth", "left_cheek", etc.)
        severity: 0-1, controls blend opacity
        blend_mode: "poisson" (seamless clone) or "alpha" (simple blend)

    Returns: VisionCorruptionResult
    """
    h, w = image.shape[:2]
    result = image.copy()
    donor_resized = cv2.resize(donor_image, (w, h))

    # Try to get landmarks for precise region masks
    if landmarks is None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = get_face_landmarks_mediapipe(image_rgb)
    if donor_landmarks is None:
        donor_rgb = cv2.cvtColor(donor_resized, cv2.COLOR_BGR2RGB)
        donor_landmarks = get_face_landmarks_mediapipe(donor_rgb)

    # Create region mask
    if landmarks is not None and region in FACE_REGIONS:
        mask = create_region_mask((h, w), landmarks, FACE_REGIONS[region])
    else:
        # Fallback: circular mask in approximate region location
        regions_approx = {
            "nose": (0.55, 0.50, 0.12),
            "mouth": (0.70, 0.50, 0.15),
            "left_cheek": (0.55, 0.30, 0.12),
            "right_cheek": (0.55, 0.70, 0.12),
            "forehead": (0.25, 0.50, 0.18),
        }
        cy_frac, cx_frac, r_frac = regions_approx.get(region, (0.5, 0.5, 0.15))
        mask = create_region_mask_simple((h, w), int(cy_frac * h),
                                         int(cx_frac * w), int(r_frac * h))

    # Apply severity
    mask = mask * severity

    # Blend
    if blend_mode == "poisson" and mask.max() > 0.1:
        try:
            # Poisson blending needs a binary mask and center point
            binary_mask = (mask > 0.3).astype(np.uint8) * 255
            moments = cv2.moments(binary_mask)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                center = (cx, cy)
                result = cv2.seamlessClone(donor_resized, result,
                                            binary_mask, center,
                                            cv2.NORMAL_CLONE)
            else:
                blend_mode = "alpha"  # fallback
        except cv2.error:
            blend_mode = "alpha"  # fallback

    if blend_mode == "alpha":
        mask_3c = mask[:, :, np.newaxis]
        result = (result * (1 - mask_3c) + donor_resized * mask_3c).astype(np.uint8)

    return VisionCorruptionResult(
        image=result,
        corruption_type="texture_splice",
        corrupted_region=(mask > 0.1).astype(np.uint8),
        severity=severity,
        metadata={"region": region, "blend_mode": blend_mode}
    )


# ============================================================
# Corruption 2: Lighting Direction Shift
# ============================================================

def estimate_lighting_gradient(image_gray):
    """
    Estimate the dominant lighting direction from the intensity gradient.
    Returns: (angle_radians, magnitude) of the dominant gradient direction.
    """
    # Sobel gradients
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)

    # Average gradient direction (weighted by magnitude)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    weights = magnitude / (magnitude.sum() + 1e-8)

    avg_gx = (grad_x * weights).sum()
    avg_gy = (grad_y * weights).sum()

    angle = np.arctan2(avg_gy, avg_gx)
    mag = np.sqrt(avg_gx ** 2 + avg_gy ** 2)
    return angle, mag


def corrupt_lighting_shift(image, landmarks=None, region="left_cheek",
                            angle_offset_deg=90, intensity=0.4,
                            severity=1.0):
    """
    Shift the apparent lighting direction in a facial sub-region.

    Creates a directional lighting gradient inconsistent with the rest
    of the face. This mimics the lighting inconsistencies common in
    face-swap and face-reenactment deepfakes.

    The corruption applies a synthetic directional light (additive gradient)
    to a masked region, making that region appear lit from a different
    direction than the rest of the face.

    Args:
        image: (H, W, 3) uint8 BGR
        landmarks: face landmarks or None
        region: which facial region to re-light
        angle_offset_deg: how many degrees to rotate the lighting direction
        intensity: strength of the synthetic light (0-1)
        severity: overall corruption severity (0-1)

    Returns: VisionCorruptionResult
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)

    # Create region mask
    if landmarks is not None and region in FACE_REGIONS:
        mask = create_region_mask((h, w), landmarks, FACE_REGIONS[region],
                                  dilate_px=8)
    else:
        regions_approx = {
            "left_cheek": (0.55, 0.30, 0.15),
            "right_cheek": (0.55, 0.70, 0.15),
            "forehead": (0.25, 0.50, 0.20),
            "nose": (0.55, 0.50, 0.12),
        }
        cy_frac, cx_frac, r_frac = regions_approx.get(region, (0.5, 0.35, 0.15))
        mask = create_region_mask_simple((h, w), int(cy_frac * h),
                                         int(cx_frac * w), int(r_frac * h))

    # Estimate current lighting direction from the full face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_angle, _ = estimate_lighting_gradient(gray)

    # Create a new lighting gradient at a different angle
    new_angle = current_angle + np.radians(angle_offset_deg)

    # Generate directional gradient
    Y, X = np.mgrid[:h, :w].astype(np.float32)
    X_norm = (X - w / 2) / (w / 2)
    Y_norm = (Y - h / 2) / (h / 2)

    # Directional light intensity
    light_gradient = np.cos(new_angle) * X_norm + np.sin(new_angle) * Y_norm
    light_gradient = light_gradient * intensity * severity * 255.0

    # Apply only within the masked region
    mask_3c = mask[:, :, np.newaxis]
    light_3c = light_gradient[:, :, np.newaxis]

    result = result + mask_3c * light_3c
    result = np.clip(result, 0, 255).astype(np.uint8)

    return VisionCorruptionResult(
        image=result,
        corruption_type="lighting_shift",
        corrupted_region=(mask > 0.1).astype(np.uint8),
        severity=severity,
        metadata={
            "region": region,
            "angle_offset_deg": angle_offset_deg,
            "intensity": intensity,
            "original_angle_deg": np.degrees(current_angle),
        }
    )


# ============================================================
# Unified corruption interface
# ============================================================

# Pool of donor images for texture splice — set by the caller
_donor_pool = []


def set_donor_pool(image_paths):
    """Set the pool of donor images for texture splice corruptions."""
    global _donor_pool
    _donor_pool = list(image_paths)


def apply_corruption(image_tensor, corruption_type, idx=0, k=0,
                      donor_pool=None, severity=None):
    """
    Unified corruption interface for the embedding pre-computation pipeline.

    Args:
        image_tensor: (3, H, W) torch tensor in [0, 1]
        corruption_type: "texture_splice" or "lighting_shift"
        idx: image index (for deterministic donor selection)
        k: corruption variant index
        donor_pool: list of image file paths for texture splice donors
        severity: corruption severity (default: random 0.5-1.0)

    Returns: (3, H, W) torch tensor in [0, 1]
    """
    import random as rand

    # Convert tensor to numpy BGR for OpenCV
    img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if severity is None:
        rand.seed(idx * 1000 + k)
        severity = rand.uniform(0.5, 1.0)

    # Get landmarks (cached per image in production — here computed fresh)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    landmarks = get_face_landmarks_mediapipe(img_rgb)

    regions = ["nose", "mouth", "left_cheek", "right_cheek", "forehead"]
    rand.seed(idx * 1000 + k + 42)
    region = rand.choice(regions)

    if corruption_type == "texture_splice":
        # Select a donor image
        pool = donor_pool or _donor_pool
        if not pool:
            # Fallback: use a colour-shifted version of the same image as donor
            donor_bgr = img_bgr.copy()
            donor_bgr = cv2.cvtColor(donor_bgr, cv2.COLOR_BGR2HSV)
            donor_bgr[:, :, 0] = (donor_bgr[:, :, 0].astype(int) + 30) % 180
            donor_bgr = cv2.cvtColor(donor_bgr, cv2.COLOR_HSV2BGR)
        else:
            donor_idx = (idx + k + 1) % len(pool)
            donor_img = cv2.imread(str(pool[donor_idx]))
            if donor_img is None:
                # Fallback
                donor_bgr = img_bgr.copy()
            else:
                donor_bgr = donor_img

        result = corrupt_texture_splice(
            img_bgr, donor_bgr, landmarks=landmarks,
            region=region, severity=severity
        )

    elif corruption_type == "lighting_shift":
        rand.seed(idx * 1000 + k + 99)
        angle = rand.choice([60, 90, 120, 150])
        intensity = rand.uniform(0.3, 0.6)

        result = corrupt_lighting_shift(
            img_bgr, landmarks=landmarks, region=region,
            angle_offset_deg=angle, intensity=intensity,
            severity=severity
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    # Convert back to tensor
    out_rgb = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
    out_tensor = torch.from_numpy(out_rgb).float().permute(2, 0, 1) / 255.0
    return out_tensor


# ============================================================
# Visualization utility
# ============================================================

def visualize_corruption(original, corrupted_result, save_path=None):
    """
    Side-by-side visualization of original and corrupted image.
    Shows: original | corrupted | corruption mask | difference map
    """
    orig = original
    corr = corrupted_result.image
    h, w = orig.shape[:2]

    # Difference map (amplified)
    diff = cv2.absdiff(orig, corr)
    diff_amplified = np.clip(diff.astype(np.float32) * 5, 0, 255).astype(np.uint8)

    # Mask overlay
    if corrupted_result.corrupted_region is not None:
        mask_vis = cv2.applyColorMap(
            (corrupted_result.corrupted_region * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        mask_overlay = cv2.addWeighted(orig, 0.5, mask_vis, 0.5, 0)
    else:
        mask_overlay = orig.copy()

    # Compose
    row = np.hstack([orig, corr, mask_overlay, diff_amplified])

    if save_path:
        cv2.imwrite(str(save_path), row)
        print(f"  Saved: {save_path}")

    return row


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys

    print("Vision corruption strategies")
    print("=" * 60)

    # Create a synthetic test face (gradient image)
    # In real usage, this would be an actual face image
    h, w = 224, 224
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient background (simulates lighting)
    for y in range(h):
        for x in range(w):
            test_img[y, x] = [
                int(128 + 60 * np.sin(x / w * np.pi)),
                int(128 + 60 * np.cos(y / h * np.pi)),
                int(140 + 40 * np.sin((x + y) / (w + h) * np.pi * 2))
            ]

    # Add a "face-like" ellipse
    cv2.ellipse(test_img, (w // 2, h // 2), (60, 80), 0, 0, 360,
                (180, 160, 150), -1)

    print(f"\nTest image: {test_img.shape}")

    # Create a different "donor" image
    donor_img = np.zeros((h, w, 3), dtype=np.uint8)
    donor_img[:] = [100, 120, 200]  # different skin tone
    cv2.ellipse(donor_img, (w // 2, h // 2), (60, 80), 0, 0, 360,
                (200, 180, 170), -1)

    # Test texture splice (without MediaPipe — will use fallback masks)
    print("\n--- Texture Splice ---")
    result1 = corrupt_texture_splice(test_img, donor_img, region="nose", severity=0.8)
    print(f"  Type: {result1.corruption_type}")
    print(f"  Severity: {result1.severity}")
    print(f"  Metadata: {result1.metadata}")
    diff1 = np.abs(test_img.astype(float) - result1.image.astype(float)).mean()
    print(f"  Mean pixel diff: {diff1:.1f}")

    # Test lighting shift
    print("\n--- Lighting Shift ---")
    result2 = corrupt_lighting_shift(test_img, region="left_cheek",
                                      angle_offset_deg=90, severity=0.8)
    print(f"  Type: {result2.corruption_type}")
    print(f"  Severity: {result2.severity}")
    print(f"  Metadata: {result2.metadata}")
    diff2 = np.abs(test_img.astype(float) - result2.image.astype(float)).mean()
    print(f"  Mean pixel diff: {diff2:.1f}")

    # Test the unified interface
    print("\n--- Unified Interface ---")
    img_tensor = torch.from_numpy(
        cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    ).float().permute(2, 0, 1) / 255.0

    out1 = apply_corruption(img_tensor, "texture_splice", idx=0, k=0)
    out2 = apply_corruption(img_tensor, "lighting_shift", idx=0, k=1)
    print(f"  texture_splice output: {out1.shape}")
    print(f"  lighting_shift output: {out2.shape}")

    # Save visualizations if output dir exists
    out_dir = Path("corruption_samples")
    out_dir.mkdir(exist_ok=True)
    visualize_corruption(test_img, result1, out_dir / "texture_splice_test.jpg")
    visualize_corruption(test_img, result2, out_dir / "lighting_shift_test.jpg")

    print("\nDone. For real usage, provide actual face images.")
