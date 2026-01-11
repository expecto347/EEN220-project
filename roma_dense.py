import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import cv2


def _imread_color_no_exif_rotation(path: str):
    flags = cv2.IMREAD_COLOR
    if hasattr(cv2, "IMREAD_IGNORE_ORIENTATION"):
        flags |= cv2.IMREAD_IGNORE_ORIENTATION
    return cv2.imread(path, flags)


@dataclass(frozen=True)
class DenseReconstructionResult:
    points_3d: np.ndarray  # 3xN float
    colors_rgb: Optional[np.ndarray]  # Nx3 uint8 (or None)
    pairs_processed: int
    pairs_skipped: int


def _round_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return int(((int(x) + m - 1) // m) * m)


def _resize_hw_to_max(h: int, w: int, max_size: int) -> Tuple[int, int]:
    if max_size <= 0:
        return int(h), int(w)
    max_dim = max(h, w)
    if max_dim <= max_size:
        return int(h), int(w)
    s = float(max_size) / float(max_dim)
    h2 = max(1, int(round(h * s)))
    w2 = max(1, int(round(w * s)))
    return h2, w2


def _get_torch_device():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep MPS branch for completeness (macOS), but this project runs on Linux too.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    return torch, device


def _build_roma_model(img_path: str, *, downsample_max_size: int):
    """Build a RoMa model following the demo defaults.

    We set `upsample_res` based on the dataset image size (optionally downsampled)
    to keep memory bounded.
    """
    torch, device = _get_torch_device()

    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise ImportError("PIL (Pillow) is required for RoMa dense.") from e

    try:
        from romatch import roma_outdoor
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "romatch is required for RoMa dense. Install it (and torch) first."
        ) from e

    w, h = Image.open(img_path).size
    h2, w2 = _resize_hw_to_max(h, w, int(downsample_max_size))
    # RoMa models typically prefer resolutions divisible by 8/16.
    h2 = _round_to_multiple(h2, 8)
    w2 = _round_to_multiple(w2, 8)

    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(h2, w2))
    return roma_model, torch, device


def _safe_match(roma_model, im1_path: str, im2_path: str, *, device):
    """Call roma_model.match with best-effort signature compatibility."""
    try:
        return roma_model.match(im1_path, im2_path, device=device)
    except TypeError:
        return roma_model.match(im1_path, im2_path)


def run_dense_reconstruction_roma(
    dataset_num: int,
    cameras_pose: dict,
    K: np.ndarray,
    img_names: Sequence[str],
    *,
    confidence_thresh: float = 0.7,
    downsample_max_size: int = 1024,
    max_points_per_pair: int = 25000,
) -> Optional[DenseReconstructionResult]:
    """Create a denser point cloud using RoMa matches + known camera poses.

    Strategy:
    - Use RoMa to get many 2D-2D correspondences for image pairs.
    - For pairs where BOTH camera poses are known, triangulate with P = K [R|t].
    - Keep points that satisfy a cheirality (positive depth) check.

    Returns None if RoMa dependencies are missing or no dense points were produced.
    """
    _ = dataset_num  # kept for future logging/conditioning

    if K is None or len(img_names) == 0 or cameras_pose is None:
        return None

    # Pick pairs among localized images: (i_k, i_{k+1}) in index order.
    localized = sorted(int(i) for i in cameras_pose.keys())
    if len(localized) < 2:
        return None

    # Build RoMa model once (use first available image).
    try:
        roma_model, torch, device = _build_roma_model(
            img_names[localized[0]],
            downsample_max_size=downsample_max_size,
        )
    except Exception as e:
        print(
            "[RoMa] Dense reconstruction unavailable. "
            "Install dependencies (torch, Pillow, romatch) and try again. "
            f"Details: {e}"
        )
        return None

    try:
        from PIL import Image
    except Exception as e:
        print(f"[RoMa] Pillow missing: {e}")
        return None

    points_all = []
    colors_all = []

    pairs_processed = 0
    pairs_skipped = 0

    K = np.asarray(K, dtype=np.float64)

    for a, b in zip(localized[:-1], localized[1:]):
        if a < 0 or b < 0 or a >= len(img_names) or b >= len(img_names):
            pairs_skipped += 1
            continue

        imA_path = img_names[a]
        imB_path = img_names[b]

        # Load original sizes (needed for to_pixel_coordinates).
        try:
            wA, hA = Image.open(imA_path).size
            wB, hB = Image.open(imB_path).size
        except Exception:
            pairs_skipped += 1
            continue

        try:
            warp, certainty = _safe_match(roma_model, imA_path, imB_path, device=device)
            matches, cert_s = roma_model.sample(warp, certainty)
            kptsA, kptsB = roma_model.to_pixel_coordinates(matches, hA, wA, hB, wB)
        except Exception:
            pairs_skipped += 1
            continue

        kptsA = kptsA.detach().cpu().numpy().astype(np.float64, copy=False)
        kptsB = kptsB.detach().cpu().numpy().astype(np.float64, copy=False)
        cert_s = cert_s.detach().cpu().numpy().reshape(-1)

        if kptsA.ndim != 2 or kptsA.shape[1] != 2 or kptsA.shape[0] < 8:
            pairs_skipped += 1
            continue

        # Filter by confidence + finiteness + bounds.
        mask = np.isfinite(cert_s) & (cert_s >= float(confidence_thresh))
        mask &= np.all(np.isfinite(kptsA), axis=1) & np.all(np.isfinite(kptsB), axis=1)
        mask &= (kptsA[:, 0] >= 0) & (kptsA[:, 0] < wA) & (kptsA[:, 1] >= 0) & (kptsA[:, 1] < hA)
        mask &= (kptsB[:, 0] >= 0) & (kptsB[:, 0] < wB) & (kptsB[:, 1] >= 0) & (kptsB[:, 1] < hB)

        idx = np.where(mask)[0]
        if idx.size < 16:
            pairs_skipped += 1
            continue

        if max_points_per_pair > 0 and idx.size > max_points_per_pair:
            idx = np.random.choice(idx, int(max_points_per_pair), replace=False)

        ptsA = kptsA[idx, :]
        ptsB = kptsB[idx, :]

        # Projection matrices in pixel coordinates.
        RA, tA = cameras_pose[a]
        RB, tB = cameras_pose[b]
        RA = np.asarray(RA, dtype=np.float64)
        RB = np.asarray(RB, dtype=np.float64)
        tA = np.asarray(tA, dtype=np.float64).reshape(3, 1)
        tB = np.asarray(tB, dtype=np.float64).reshape(3, 1)

        PA = K @ np.hstack([RA, tA])
        PB = K @ np.hstack([RB, tB])

        # Triangulate (OpenCV expects 2xN).
        X_h = cv2.triangulatePoints(PA, PB, ptsA.T, ptsB.T)
        with np.errstate(all="ignore"):
            X = (X_h[:3, :] / X_h[3:4, :]).astype(np.float64, copy=False)

        finite = np.all(np.isfinite(X), axis=0)
        if not np.any(finite):
            pairs_skipped += 1
            continue

        X = X[:, finite]
        ptsA_f = ptsA[finite, :]

        # Cheirality + sanity bounds.
        X_camA = (RA @ X) + tA
        X_camB = (RB @ X) + tB
        cheir = (X_camA[2, :] > 0) & (X_camB[2, :] > 0)
        cheir &= np.all(np.abs(X) < 1e6, axis=0)

        if not np.any(cheir):
            pairs_skipped += 1
            continue

        X = X[:, cheir]
        ptsA_f = ptsA_f[cheir, :]

        # Colors from image A (nearest-neighbor).
        imgA = _imread_color_no_exif_rotation(imA_path)
        if imgA is None:
            # Still keep geometry.
            colors = None
        else:
            uu = np.clip(np.round(ptsA_f[:, 0]).astype(np.int32), 0, wA - 1)
            vv = np.clip(np.round(ptsA_f[:, 1]).astype(np.int32), 0, hA - 1)
            bgr = imgA[vv, uu, :]
            colors = bgr[:, ::-1].copy()  # RGB

        points_all.append(X)
        if colors is not None:
            colors_all.append(colors)

        pairs_processed += 1

    if pairs_processed == 0 or len(points_all) == 0:
        return None

    points_3d = np.hstack(points_all) if len(points_all) > 1 else points_all[0]

    colors_rgb: Optional[np.ndarray]
    if len(colors_all) == len(points_all) and len(colors_all) > 0:
        colors_rgb = np.vstack(colors_all).astype(np.uint8, copy=False)
    else:
        colors_rgb = None

    return DenseReconstructionResult(
        points_3d=points_3d,
        colors_rgb=colors_rgb,
        pairs_processed=int(pairs_processed),
        pairs_skipped=int(pairs_skipped),
    )


def save_ply(path: str, points_3d: np.ndarray, colors_rgb: Optional[np.ndarray] = None) -> None:
    """Save a point cloud to an ASCII PLY file."""
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] != 3:
        raise ValueError("points_3d must be 3xN")

    n = int(pts.shape[1])
    if n == 0:
        raise ValueError("No points to save")

    cols = None
    if colors_rgb is not None:
        cols = np.asarray(colors_rgb)
        if cols.shape[0] != n or cols.shape[1] != 3:
            cols = None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if cols is not None:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header += ["end_header"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        if cols is None:
            for i in range(n):
                x, y, z = pts[0, i], pts[1, i], pts[2, i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            cols = cols.astype(np.uint8, copy=False)
            for i in range(n):
                x, y, z = pts[0, i], pts[1, i], pts[2, i]
                r, g, b = int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2])
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
