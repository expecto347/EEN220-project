import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from project_helpers import get_dataset_info
from project_helpers import homography_to_RT
from project_helpers import correct_H_sign

# ==========================================
# Global Utilities
# ==========================================

def set_global_seed(seed: int, *, deterministic_cv2: bool = False) -> None:
    """Best-effort global seeding for reproducibility."""
    seed_i = int(seed)
    np.random.seed(seed_i)
    random.seed(seed_i)
    if hasattr(cv2, "setRNGSeed"):
        try:
            cv2.setRNGSeed(seed_i)
        except Exception:
            pass
    if deterministic_cv2 and hasattr(cv2, "setNumThreads"):
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass


def _make_rng(seed: int | None = None, rng: np.random.Generator | None = None) -> np.random.Generator:
    if rng is not None:
        return rng
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _imread_color_no_exif_rotation(path: str):
    """Reads an image ensuring no auto-rotation from EXIF data."""
    flags = cv2.IMREAD_COLOR
    if hasattr(cv2, "IMREAD_IGNORE_ORIENTATION"):
        flags |= cv2.IMREAD_IGNORE_ORIENTATION
    return cv2.imread(path, flags)


def _camera_center_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera center in world coordinates for x_cam = R X_world + t."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return (-R.T @ t).reshape(3)


def _min_parallax_mask_from_centers(
    X_world: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    *,
    min_angle_deg: float = 1.5,
) -> np.ndarray:
    """Return mask of points with parallax angle >= min_angle_deg.

    Uses viewing rays (X - C1) and (X - C2) in world frame.
    """
    X = np.asarray(X_world, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != 3:
        return np.zeros((0,), dtype=bool)
    n = int(X.shape[1])
    if n == 0:
        return np.zeros((0,), dtype=bool)

    C1 = np.asarray(C1, dtype=np.float64).reshape(3, 1)
    C2 = np.asarray(C2, dtype=np.float64).reshape(3, 1)
    v1 = X - C1
    v2 = X - C2
    n1 = np.linalg.norm(v1, axis=0)
    n2 = np.linalg.norm(v2, axis=0)

    # Angle >= min_angle  <=>  cos(angle) <= cos(min_angle)
    min_angle_rad = np.deg2rad(float(min_angle_deg))
    cos_thresh = float(np.cos(min_angle_rad))

    with np.errstate(all="ignore"):
        dot = np.sum(v1 * v2, axis=0)
        denom = (n1 * n2) + 1e-12
        cosang = dot / denom

    finite = np.isfinite(cosang) & np.isfinite(n1) & np.isfinite(n2)
    finite &= (n1 > 1e-12) & (n2 > 1e-12)
    # Clamp only for numeric stability; thresholding uses the unclamped value.
    cosang = np.clip(cosang, -1.0, 1.0)
    return finite & (cosang <= cos_thresh)


# ==========================================
# Dense Reconstruction Module (RoMa)
# ==========================================

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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    return torch, device


def _build_roma_model(img_path: str, *, downsample_max_size: int):
    """Build a RoMa model following the demo defaults."""
    import torch
    from PIL import Image
    from romatch import roma_outdoor

    _, device = _get_torch_device()

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
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[DenseReconstructionResult]:
    """Create a denser point cloud using RoMa matches + known camera poses."""
    
    # Assume imports work as environment is pre-configured
    from PIL import Image
    
    if K is None or len(img_names) == 0 or cameras_pose is None:
        return None

    if rng is None:
        rng = np.random.default_rng(None if seed is None else int(seed))

    # Pick pairs among localized images: (i_k, i_{k+1}) in index order.
    localized = sorted(int(i) for i in cameras_pose.keys())
    if len(localized) < 2:
        return None

    # Build RoMa model once
    roma_model, torch, device = _build_roma_model(
        img_names[localized[0]],
        downsample_max_size=downsample_max_size,
    )

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

        try:
            wA, hA = Image.open(imA_path).size
            wB, hB = Image.open(imB_path).size
            
            warp, certainty = _safe_match(roma_model, imA_path, imB_path, device=device)
            matches, cert_s = roma_model.sample(warp, certainty)
            kptsA, kptsB = roma_model.to_pixel_coordinates(matches, hA, wA, hB, wB)
        except Exception as e:
            print(f"Skipping pair {a}-{b} due to error: {e}")
            pairs_skipped += 1
            continue

        kptsA = kptsA.detach().cpu().numpy().astype(np.float64, copy=False)
        kptsB = kptsB.detach().cpu().numpy().astype(np.float64, copy=False)
        cert_s = cert_s.detach().cpu().numpy().reshape(-1)

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
            idx = rng.choice(idx, int(max_points_per_pair), replace=False)

        ptsA = kptsA[idx, :]
        ptsB = kptsB[idx, :]

        # Projection matrices in pixel coordinates.
        RA, tA = cameras_pose[a]
        RB, tB = cameras_pose[b]
        PA = K @ np.hstack([np.asarray(RA), np.asarray(tA).reshape(3,1)])
        PB = K @ np.hstack([np.asarray(RB), np.asarray(tB).reshape(3,1)])

        # Triangulate
        X_h = cv2.triangulatePoints(PA, PB, ptsA.T, ptsB.T)
        with np.errstate(all="ignore"):
            X = (X_h[:3, :] / X_h[3:4, :]).astype(np.float64, copy=False)

        # Filter points behind cameras or with small parallax angles
        finite = np.all(np.isfinite(X), axis=0)
        X = X[:, finite]
        ptsA_f = ptsA[finite, :]
        
        # Simple Cheirality check
        X_camA = (RA @ X) + tA.reshape(3,1)
        X_camB = (RB @ X) + tB.reshape(3,1)
        cheir = (X_camA[2, :] > 0) & (X_camB[2, :] > 0)
        cheir &= np.all(np.abs(X) < 1e6, axis=0) # Remove points at infinity

        if not np.any(cheir):
            pairs_skipped += 1
            continue

        X = X[:, cheir]
        ptsA_f = ptsA_f[cheir, :]

        # Minimum triangulation parallax angle filter (world frame)
        C_A = _camera_center_world(RA, tA)
        C_B = _camera_center_world(RB, tB)
        par_mask = _min_parallax_mask_from_centers(X, C_A, C_B, min_angle_deg=1.5)
        if par_mask.size != X.shape[1] or (not np.any(par_mask)):
            pairs_skipped += 1
            continue
        X = X[:, par_mask]
        ptsA_f = ptsA_f[par_mask, :]

        # Colors from image A
        imgA = _imread_color_no_exif_rotation(imA_path)
        if imgA is not None:
            uu = np.clip(np.round(ptsA_f[:, 0]).astype(np.int32), 0, wA - 1)
            vv = np.clip(np.round(ptsA_f[:, 1]).astype(np.int32), 0, hA - 1)
            bgr = imgA[vv, uu, :]
            colors = bgr[:, ::-1].copy()  # RGB
            colors_all.append(colors)

        points_all.append(X)
        pairs_processed += 1

    if pairs_processed == 0 or len(points_all) == 0:
        return None

    points_3d = np.hstack(points_all) if len(points_all) > 1 else points_all[0]
    colors_rgb = np.vstack(colors_all).astype(np.uint8, copy=False) if colors_all else None

    return DenseReconstructionResult(
        points_3d=points_3d,
        colors_rgb=colors_rgb,
        pairs_processed=int(pairs_processed),
        pairs_skipped=int(pairs_skipped),
    )


def save_ply(path: str, points_3d: np.ndarray, colors_rgb: Optional[np.ndarray] = None) -> None:
    """Save a point cloud to an ASCII PLY file."""
    pts = np.asarray(points_3d, dtype=np.float64)
    n = int(pts.shape[1])
    if n == 0:
        return

    cols = None
    if colors_rgb is not None:
        cols = np.asarray(colors_rgb)

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
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        if cols is None:
            for i in range(n):
                f.write(f"{pts[0, i]:.6f} {pts[1, i]:.6f} {pts[2, i]:.6f}\n")
        else:
            cols = cols.astype(np.uint8, copy=False)
            for i in range(n):
                f.write(f"{pts[0, i]:.6f} {pts[1, i]:.6f} {pts[2, i]:.6f} {cols[i, 0]} {cols[i, 1]} {cols[i, 2]}\n")


# ==========================================
# Core Algorithm Implementation (SfM)
# ==========================================

def estimate_H_DLT(x1, x2):
    """
    Estimate Homography H using DLT algorithm from at least 4 point pairs.
    x1, x2: 3xN normalized coordinates.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    if x1.ndim != 2 or x2.ndim != 2 or x1.shape[0] != 3 or x2.shape[0] != 3:
        return None

    valid = np.all(np.isfinite(x1), axis=0) & np.all(np.isfinite(x2), axis=0)
    x1 = x1[:, valid]
    x2 = x2[:, valid]
    n = x1.shape[1]
    if n < 4:
        return None

    # Hartley normalization improves numerical stability
    T1, x1n = _hartley_normalize_2d(x1)
    T2, x2n = _hartley_normalize_2d(x2)

    x1i = _to_inhomogeneous_2d(x1n)
    x2i = _to_inhomogeneous_2d(x2n)
    u1, v1 = x1i[0, :], x1i[1, :]
    u2, v2 = x2i[0, :], x2i[1, :]

    A = np.zeros((2 * n, 9), dtype=np.float64)
    # DLT constraints for x' ~ H x.
    A[0::2, 0] = -u1
    A[0::2, 1] = -v1
    A[0::2, 2] = -1.0
    A[0::2, 6] = u1 * u2
    A[0::2, 7] = v1 * u2
    A[0::2, 8] = u2

    A[1::2, 3] = -u1
    A[1::2, 4] = -v1
    A[1::2, 5] = -1.0
    A[1::2, 6] = u1 * v2
    A[1::2, 7] = v1 * v2
    A[1::2, 8] = v2

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    Hn = Vt[-1, :].reshape(3, 3)
    try:
        T2inv = np.linalg.inv(T2)
    except np.linalg.LinAlgError:
        return None

    H = T2inv @ Hn @ T1
    if not np.all(np.isfinite(H)):
        return None

    # Normalize scale for consistency.
    if np.isfinite(H[2, 2]) and abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def _symmetric_transfer_errors_sq(H, x1s, x2s):
    """Symmetric transfer error for homography (squared), in normalized coordinates."""
    H = np.asarray(H, dtype=np.float64)
    x1s = np.asarray(x1s, dtype=np.float64)
    x2s = np.asarray(x2s, dtype=np.float64)
    
    with np.errstate(all='ignore'):
        x2p = H @ x1s
    x2p_i = _to_inhomogeneous_2d(x2p)
    x2_i = _to_inhomogeneous_2d(x2s)
    with np.errstate(all='ignore'):
        d_fwd = np.sum((x2_i - x2p_i) ** 2, axis=0)

    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    with np.errstate(all='ignore'):
        x1p = Hinv @ x2s
    x1p_i = _to_inhomogeneous_2d(x1p)
    x1_i = _to_inhomogeneous_2d(x1s)
    with np.errstate(all='ignore'):
        d_bwd = np.sum((x1_i - x1p_i) ** 2, axis=0)

    d2 = d_fwd + d_bwd
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    return d2


def _cheirality_positive_count(R, t, x1s, x2s, inliers, *, max_points=50, rng: np.random.Generator | None = None):
    """Count points with positive depth in both views for pose (R,t)."""
    if inliers is None:
        return 0, 0
    idx = np.where(np.asarray(inliers, dtype=bool))[0]
    if idx.size == 0:
        return 0, 0
    if max_points > 0 and idx.size > max_points:
        rng = _make_rng(rng=rng)
        idx = rng.choice(idx, int(max_points), replace=False)

    P1 = np.eye(4)[:3]
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    P2 = np.hstack((np.asarray(R, dtype=np.float64), t))

    pos = 0
    checked = 0
    for j in idx:
        X = triangulate_point_DLT(P1, P2, x1s[:, j], x2s[:, j])
        if not np.all(np.isfinite(X)):
            continue
        X_cam2 = (P2 @ X)
        checked += 1
        if X[2] > 0 and X_cam2[2] > 0:
            pos += 1
    return int(pos), int(checked)


def check_pose_inliers(R, t, x1, x2, threshold_sq):
    """
    Helper function: Check inlier count and geometric validity for a given (R, t).
    Used to evaluate solutions from E or H.
    """
    t = t.reshape(3, 1)
    # Construct Essential Matrix E = [t]x R
    Tx = np.array([[0, -t[2,0], t[1,0]],
                   [t[2,0], 0, -t[0,0]],
                   [-t[1,0], t[0,0], 0]])
    E_hyp = Tx @ R
    
    # 1. Compute Sampson error to determine inliers
    d2 = compute_sampson_errors(E_hyp, x1, x2)
    inliers = d2 < threshold_sq
    count = np.sum(inliers)
    
    # If too few inliers, skip expensive triangulation check
    if count < 8:
        return 0, inliers, False

    # 2. Cheirality check: Use multiple inliers for robustness (prevents failure on planar scenes)
    pos, checked = _cheirality_positive_count(R, t, x1, x2, inliers, max_points=25)
    if checked == 0:
        return int(count), inliers, False

    frac = float(pos) / float(checked)
    is_valid = (pos >= 5) and (frac >= 0.6)
    return int(count), inliers, bool(is_valid)


def ransac_estimate_parallel(x1s, x2s, threshold=0.001, num_iterations=2000, *, rng: np.random.Generator | None = None):
    """
    Parallel RANSAC: Simultaneously search for E (8-point) and H (4-point).
    Returns the best (R, t) and the inlier mask directly.
    """
    threshold_sq = threshold ** 2
    n_points = x1s.shape[1]
    
    best_score = -1
    best_Rt = (None, None)
    best_inliers = None
    
    if n_points < 8:
        return None, None, None

    # Pre-build Identity matrix for select_correct_pose
    P1_identity = np.eye(4)[:3]

    rng = _make_rng(rng=rng)
    for _ in range(num_iterations):
        # 1. Sample 8 points (E needs 8, H needs 4 of these)
        sample_idx = rng.choice(n_points, 8, replace=False)
        x1_sample, x2_sample = x1s[:, sample_idx], x2s[:, sample_idx]
        
        # ==========================================
        # Branch A: Estimate Essential Matrix (8-point)
        # ==========================================
        E_approx = estimate_F_DLT(x1_sample, x2_sample)
        E_cand = enforce_essential(E_approx)
        
        if E_cand is not None:
            # Decompose E into 4 possible solutions
            candidates = extract_P_from_E(E_cand)
            
            # Use Cheirality to pick the unique solution (R, t) for the sample
            res = select_correct_pose(candidates, P1_identity, x1_sample, x2_sample)
            
            if res is not None:
                R_E, t_E = res
                # Score against all data points
                count, inliers, valid = check_pose_inliers(R_E, t_E, x1s, x2s, threshold_sq)
                
                if valid and count > best_score:
                    best_score = count
                    best_Rt = (R_E, t_E)
                    best_inliers = inliers

        # ==========================================
        # Branch B: Estimate Homography (4-point)
        # ==========================================
        # Use first 4 points of the sample
        H_cand = estimate_H_DLT(x1_sample[:, :4], x2_sample[:, :4])
        
        if H_cand is not None:
            # Stabilize sign for decomposition
            try:
                H_cand = correct_H_sign(H_cand, x1_sample[:, :4], x2_sample[:, :4])
            except Exception:
                pass

            # Decompose H into possible (R, t) list
            try:
                RTs = homography_to_RT(H_cand)
                RTs = np.asarray(RTs)
                
                # If decomposition succeeds
                if RTs.ndim == 3 and RTs.shape[1:] == (3, 4):
                    # Score H by symmetric transfer error
                    d2_h = _symmetric_transfer_errors_sq(H_cand, x1s, x2s)
                    if d2_h is None:
                        continue
                    inliers_h = d2_h < (2.0 * threshold_sq)
                    count_h = int(np.sum(inliers_h))

                    # Check each hypothesis from H decomposition
                    for k in range(RTs.shape[0]):
                        R_H = RTs[k, :, :3]
                        t_H = RTs[k, :, 3]
                        
                        # Cheirality validation using H-inliers
                        pos, checked = _cheirality_positive_count(
                            R_H, t_H, x1s, x2s, inliers_h, max_points=50, rng=rng
                        )
                        valid = (checked > 0) and (pos >= 8) and ((pos / checked) >= 0.6)
                        
                        if valid and count_h > best_score:
                            best_score = count_h
                            best_Rt = (R_H, t_H)
                            best_inliers = inliers_h
            except Exception:
                pass # Ignore H decomposition failures

    return best_Rt[0], best_Rt[1], best_inliers


def _reprojection_errors_px(object_points, image_points, R, t, K):
    """Compute per-point reprojection errors in pixels."""
    if object_points is None or image_points is None:
        return np.array([], dtype=np.float64)
    X = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    x = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if X.shape[0] == 0 or x.shape[0] == 0 or X.shape[0] != x.shape[0]:
        return np.array([], dtype=np.float64)

    finite = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(x), axis=1)
    if not np.any(finite):
        return np.array([], dtype=np.float64)
    X = X[finite, :]
    x = x[finite, :]

    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64)

    with np.errstate(all='ignore'):
        X_cam = (R @ X.T) + t
        z = X_cam[2, :]
        valid = np.isfinite(z) & (np.abs(z) > 1e-12)
    if not np.any(valid):
        return np.array([], dtype=np.float64)

    with np.errstate(all='ignore'):
        u = X_cam[0, valid] / z[valid]
        v = X_cam[1, valid] / z[valid]
        proj = (K @ np.vstack([u, v, np.ones_like(u)]))[:2, :].T
        err = np.linalg.norm(proj - x[valid, :], axis=1)
    err = err[np.isfinite(err)]
    return err


def normalize_points(pts, K):
    """Normalize image points: x_n = K^(-1) * x."""
    pts64 = np.asarray(pts, dtype=np.float64)
    if pts64.shape[0] == 2: # Make homogeneous if 2xN
        pts64 = np.vstack((pts64, np.ones((1, pts64.shape[1]))))
    
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    with np.errstate(all='ignore'):
        normalized = Kinv @ pts64
    return normalized[:3, :] # Return 3xN


def _to_inhomogeneous_2d(x_h):
    """Convert 3xN homogeneous to 2xN inhomogeneous (safe)."""
    x_h = np.asarray(x_h, dtype=np.float64)
    if x_h.shape[0] == 2:
        return x_h
    w = x_h[2, :]
    with np.errstate(all='ignore'):
        u = x_h[0, :] / w
        v = x_h[1, :] / w
    return np.vstack([u, v])


def _hartley_normalize_2d(x_h):
    """Hartley isotropic normalization for 2D homogeneous points."""
    x_h = np.asarray(x_h, dtype=np.float64)
    if x_h.shape[0] == 2:
        x_h = np.vstack([x_h, np.ones((1, x_h.shape[1]), dtype=np.float64)])

    x = _to_inhomogeneous_2d(x_h)
    finite = np.all(np.isfinite(x), axis=0)
    if not np.any(finite):
        return np.eye(3), x_h

    xf = x[:, finite]
    centroid = np.mean(xf, axis=1)
    dx = xf - centroid.reshape(2, 1)
    d = np.sqrt(np.sum(dx * dx, axis=0))
    mean_d = float(np.mean(d)) if d.size else 0.0
    s = 1.0 if (not np.isfinite(mean_d) or mean_d < 1e-12) else (np.sqrt(2.0) / mean_d)

    T = np.array([
        [s, 0.0, -s * centroid[0]],
        [0.0, s, -s * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    x_norm = T @ x_h
    return T, x_norm


def estimate_F_DLT(x1s, x2s):
    """Estimate Fundamental/Essential matrix using normalized 8-point algorithm."""
    valid_idx = np.all(np.isfinite(x1s), axis=0) & np.all(np.isfinite(x2s), axis=0)
    x1 = np.asarray(x1s[:, valid_idx], dtype=np.float64)
    x2 = np.asarray(x2s[:, valid_idx], dtype=np.float64)

    n_points = x1.shape[1]
    if n_points < 8:
        return None

    T1, x1n = _hartley_normalize_2d(x1)
    T2, x2n = _hartley_normalize_2d(x2)

    x1i = _to_inhomogeneous_2d(x1n)
    x2i = _to_inhomogeneous_2d(x2n)
    u1, v1 = x1i[0, :], x1i[1, :]
    u2, v2 = x2i[0, :], x2i[1, :]

    M = np.zeros((n_points, 9), dtype=np.float64)
    M[:, 0] = u2 * u1
    M[:, 1] = u2 * v1
    M[:, 2] = u2
    M[:, 3] = v2 * u1
    M[:, 4] = v2 * v1
    M[:, 5] = v2
    M[:, 6] = u1
    M[:, 7] = v1
    M[:, 8] = 1.0

    _, _, Vt = np.linalg.svd(M)
    F = Vt[-1, :].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    F_rank2 = U @ np.diag(S) @ Vt

    # Denormalize
    F_denorm = T2.T @ F_rank2 @ T1
    return F_denorm


def enforce_essential(E_approx):
    """Enforce singular values (1,1,0) for Essential Matrix."""
    if E_approx is None: return None
    U, S, Vt = np.linalg.svd(E_approx)
    if np.linalg.det(U @ Vt) < 0: Vt = -Vt 
    S_new = np.diag([1, 1, 0])
    E_constrained = U @ S_new @ Vt
    return E_constrained


def compute_sampson_errors(E, x1s, x2s):
    """Compute Sampson approximation of geometric reprojection error (squared)."""
    E = np.asarray(E, dtype=np.float64)
    x1s = np.asarray(x1s, dtype=np.float64)
    x2s = np.asarray(x2s, dtype=np.float64)

    with np.errstate(all='ignore'):
        Ex1 = E @ x1s
        Etx2 = E.T @ x2s
        x2tEx1 = np.sum(x2s * Ex1, axis=0)
        denom = Ex1[0, :]**2 + Ex1[1, :]**2 + Etx2[0, :]**2 + Etx2[1, :]**2
        denom = denom + 1e-12
        d2 = (x2tEx1**2) / denom
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    return d2


def extract_P_from_E(E):
     """Extract 4 possible camera matrices P2 from E."""
     U, S, Vt = np.linalg.svd(E)
     if np.linalg.det(U @ Vt) < 0: Vt = -Vt
     W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
     
     R1 = U @ W @ Vt
     R2 = U @ W.T @ Vt
     u3 = U[:, 2] # Translation up to scale
     
     # 4 configurations: (R1, t), (R1, -t), (R2, t), (R2, -t)
     candidates = []
     for R in [R1, R2]:
         for t in [u3, -u3]:
             candidates.append((R, t))
     return candidates


def triangulate_point_DLT(P1, P2, point1, point2):
    """Triangulate a single point using DLT."""
    A = np.zeros((4, 4))
    A[0, :] = point1[0] * P1[2, :] - P1[0, :]
    A[1, :] = point1[1] * P1[2, :] - P1[1, :]
    A[2, :] = point2[0] * P2[2, :] - P2[0, :]
    A[3, :] = point2[1] * P2[2, :] - P2[1, :]
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X / X[3] # Normalize homogeneous coordinate


def select_correct_pose(candidates, P1, points1, points2):
    """Select the correct (R, t) configuration using Cheirality check."""
    best_Rt = None
    max_positive_depths = -1
    
    # Check subset of points for speed
    check_indices = range(min(50, points1.shape[1]))
    
    for R, t in candidates:
        P2_cand = np.hstack((R, t.reshape(3,1)))
        pos_depth_count = 0
        for j in check_indices:
            X = triangulate_point_DLT(P1, P2_cand, points1[:,j], points2[:,j])
            X_cam2 = P2_cand @ X
            # Check Z > 0 in both cameras (Cheirality constraint)
            if X[2] > 0 and X_cam2[2] > 0:
                pos_depth_count += 1
                
        if pos_depth_count > max_positive_depths:
            max_positive_depths = pos_depth_count
            best_Rt = (R, t)
            
    return best_Rt


def estimate_T_linear(points2D, points3D, R):
    """
    Linear estimation of T given R and 2D-3D matches.
    Equation: lambda * x = R X + T
    """
    points2D = np.asarray(points2D, dtype=np.float64)
    points3D = np.asarray(points3D, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    # Filter invalid/extreme points to keep least-squares stable.
    finite = np.all(np.isfinite(points2D), axis=0) & np.all(np.isfinite(points3D), axis=0)
    if np.any(finite):
        p3 = points3D[:, finite]
        bounded = np.all(np.abs(p3) < 1e6, axis=0)
        keep = np.zeros(points2D.shape[1], dtype=bool)
        keep[np.where(finite)[0][bounded]] = True
        points2D = points2D[:, keep]
        points3D = points3D[:, keep]

    N = points2D.shape[1]
    if N < 2:
        return None
    A = np.zeros((2 * N, 3))
    b = np.zeros(2 * N)
    
    with np.errstate(all='ignore'):
        X_rotated = R @ points3D 
    
    for i in range(N):
        u, v = points2D[0, i], points2D[1, i]
        Xr, Yr, Zr = X_rotated[:, i]
        
        # Eliminating lambda (depth): u = (Xr + Tx)/(Zr + Tz)
        A[2*i, 0] = 1        # Tx
        A[2*i, 1] = 0        # Ty
        A[2*i, 2] = -u       # Tz
        b[2*i]    = u*Zr - Xr
        
        A[2*i+1, 0] = 0      # Tx
        A[2*i+1, 1] = 1      # Ty
        A[2*i+1, 2] = -v     # Tz
        b[2*i+1]  = v*Zr - Yr

    T, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(T, dtype=np.float64)


def estimate_T_robust(points2D, points3D, R, threshold=0.005, num_iterations=1000, *, rng: np.random.Generator | None = None):
    """
    Robustly estimate T using RANSAC (Reduced camera resectioning).
    points2D should be NORMALIZED coordinates.
    """
    points2D = np.asarray(points2D, dtype=np.float64)
    points3D = np.asarray(points3D, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    finite = np.all(np.isfinite(points2D), axis=0) & np.all(np.isfinite(points3D), axis=0)
    if np.any(finite):
        p3 = points3D[:, finite]
        bounded = np.all(np.abs(p3) < 1e6, axis=0)
        keep = np.zeros(points2D.shape[1], dtype=bool)
        keep[np.where(finite)[0][bounded]] = True
        points2D = points2D[:, keep]
        points3D = points3D[:, keep]

    best_T = None
    max_inliers = 0
    best_inliers_mask = None
    N = points2D.shape[1]
    
    if N < 2: return None, None 
    
    rng = _make_rng(rng=rng)
    for i in range(num_iterations):
        # 1. Minimal sample (2 points)
        idx = rng.choice(N, 2, replace=False)
        sample_2d = points2D[:, idx]
        sample_3d = points3D[:, idx]
        
        T_curr = estimate_T_linear(sample_2d, sample_3d, R)
        if T_curr is None or not np.all(np.isfinite(T_curr)):
            continue
        
        # 2. Count inliers
        with np.errstate(all='ignore'):
            X_cam = R @ points3D + T_curr.reshape(3, 1)
        # Avoid zero depth
        X_cam[2, np.abs(X_cam[2,:]) < 1e-7] = 1e-6
        
        u_proj = X_cam[0, :] / X_cam[2, :]
        v_proj = X_cam[1, :] / X_cam[2, :]
        
        dist_sq = (u_proj - points2D[0, :])**2 + (v_proj - points2D[1, :])**2
        inliers = dist_sq < threshold**2
        num_inliers = np.sum(inliers)
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_T = T_curr
            best_inliers_mask = inliers
            
    # Optional: Refine using all inliers
    if max_inliers > 2:
        best_T = estimate_T_linear(points2D[:, best_inliers_mask], 
                                   points3D[:, best_inliers_mask], R)
                                   
    return best_T, best_inliers_mask


# ==========================================
# Matching and Auxiliary Logic
# ==========================================

def _collect_2d3d_correspondences(cloud_des, cloud_points, cloud_ref_px, des_i, kp_i, ratio=0.75):
    if cloud_des is None or des_i is None:
        return []
    if len(cloud_des) == 0 or len(des_i) == 0:
        return []
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(cloud_des, des_i, k=2)
    correspondences = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            pt_img = kp_i[m.trainIdx].pt
            pt_3d = cloud_points[:, m.queryIdx]
            if np.all(np.isfinite(pt_3d)):
                pt_ref = cloud_ref_px[m.queryIdx]
                correspondences.append((m.distance, m.queryIdx, m.trainIdx, pt_img, pt_3d, pt_ref))
    return correspondences


def _append_to_cloud(cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2,
                     X_world_list, des_ref_list, des_i_list, px_ref_list, px_i_list):
    """Append new 3D points + paired descriptors/pixels to the running cloud."""
    if len(X_world_list) == 0:
        return cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2

    X_new = np.asarray(X_world_list, dtype=np.float64).T  # 3xM
    d1_new = np.asarray(des_ref_list)
    d2_new = np.asarray(des_i_list)
    px1_new = np.asarray(px_ref_list, dtype=np.float64)
    px2_new = np.asarray(px_i_list, dtype=np.float64)

    if cloud_points is None or cloud_points.size == 0:
        cloud_points = X_new
        cloud_des_1 = d1_new
        cloud_des_2 = d2_new
        cloud_px_1 = px1_new
        cloud_px_2 = px2_new
        return cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2

    cloud_points = np.hstack([cloud_points, X_new])
    cloud_des_1 = np.vstack([cloud_des_1, d1_new])
    cloud_des_2 = np.vstack([cloud_des_2, d2_new])
    cloud_px_1 = np.vstack([cloud_px_1, px1_new])
    cloud_px_2 = np.vstack([cloud_px_2, px2_new])
    return cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2


def filter_points_distance_quantile(
    cloud_points,
    cloud_des_1,
    cloud_des_2,
    cloud_px_1,
    cloud_px_2,
    factor=5.0,
    quantile=90,
):
    """Filters 3D points based on distance from the center of gravity."""
    if cloud_points is None:
        return cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2

    cloud_points = np.asarray(cloud_points, dtype=np.float64)
    if cloud_points.ndim != 2 or cloud_points.shape[0] != 3 or cloud_points.shape[1] == 0:
        return cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2

    # 1. Calculate Center of Gravity (Centroid)
    center = np.mean(cloud_points, axis=1, keepdims=True)

    # 2. Calculate Euclidean distances from center
    diff = cloud_points - center
    distances = np.linalg.norm(diff, axis=0)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2

    # 3. Calculate the quantile of distances
    q_val = np.percentile(distances, float(quantile))

    # 4. Create Mask
    limit = float(factor) * float(q_val)
    mask = np.linalg.norm(diff, axis=0) <= limit

    n_removed = int(cloud_points.shape[1] - np.sum(mask))
    if n_removed > 0:
        print(f"  [Filter] Removed {n_removed} points > {limit:.2f} units from center.")

    # 5. Apply Mask
    new_points = cloud_points[:, mask]
    new_des_1 = cloud_des_1[mask] if cloud_des_1 is not None else None
    new_des_2 = cloud_des_2[mask] if cloud_des_2 is not None else None
    new_px_1 = cloud_px_1[mask] if cloud_px_1 is not None else None
    new_px_2 = cloud_px_2[mask] if cloud_px_2 is not None else None

    return new_points, new_des_1, new_des_2, new_px_1, new_px_2


def _merge_unique_correspondences(corr_lists):
    all_corr = []
    for corr in corr_lists:
        all_corr.extend(corr)
    all_corr.sort(key=lambda x: x[0]) 

    used_cloud = set()
    used_kp = set()
    valid_2d = []
    valid_3d = []
    for dist, cloud_idx, kp_idx, pt_img, pt_3d, _pt_ref in all_corr:
        if cloud_idx in used_cloud or kp_idx in used_kp:
            continue
        used_cloud.add(cloud_idx)
        used_kp.add(kp_idx)
        valid_2d.append(pt_img)
        valid_3d.append(pt_3d)
    return valid_2d, valid_3d


def plot_3d_reconstruction_maybe(points_3d, cameras, *, show=True, save_path=None):
    if (not show) and (save_path is None):
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    mean = np.mean(points_3d, axis=1)
    std = np.std(points_3d, axis=1)
    mask = (np.abs(points_3d[0] - mean[0]) < 3 * std[0]) & \
           (np.abs(points_3d[1] - mean[1]) < 3 * std[1]) & \
           (np.abs(points_3d[2] - mean[2]) < 3 * std[2])
    points_filtered = points_3d[:, mask]

    ax.scatter(points_filtered[0], points_filtered[1], points_filtered[2],
               s=2, c=points_filtered[2], cmap='viridis', alpha=0.6)
    for i, C in enumerate(cameras):
        ax.scatter(C[0], C[1], C[2], s=50, c='red', marker='^', label=f'{i}')
    
    # Make axes limits equal
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ==========================================
# MAIN SFM LOOP
# ==========================================

def run_sfm(
    dataset_num,
    *,
    visualize=True,
    save_plot_dir=None,
    return_metrics=True,
    use_roma_dense=False,
    roma_confidence_thresh=0.7,
    roma_downsample_max_size=1024,
    filter_final_cloud: bool = True,
    seed: int | None = None,
):
    print(f"--- Running SfM on Dataset {dataset_num} ---")
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset_num)
    if K is None: return

    if seed is not None:
        set_global_seed(int(seed), deterministic_cv2=True)
    rng = _make_rng(seed=seed)

    metrics = {
        "dataset": int(dataset_num),
        "summary": {},
        "init": {},
        "pnp": {}
    }

    # 1. Load Images
    images = []
    for name in img_names:
        img = _imread_color_no_exif_rotation(name)
        if img is None: print(f"Error reading {name}"); return
        images.append(img)
    print(f"Loaded {len(images)} images.")
    
    # Calculate normalized threshold (Project suggestion: threshold / f)
    focal_length = (K[0,0] + K[1,1]) / 2.0
    norm_threshold = float(pixel_threshold) / focal_length

    idx1, idx2 = init_pair
    print(f"Step 2: Initial reconstruction using pair {idx1}-{idx2}...")
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(images[idx1], None)
    kp2, des2 = sift.detectAndCompute(images[idx2], None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    pts1_good = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).T
    pts2_good = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).T
    
    # 2.1 Normalized Coordinates
    pts1_good_n = normalize_points(pts1_good, K)
    pts2_good_n = normalize_points(pts2_good, K)
    
    # 2.2 Parallel RANSAC (Search for E and H simultaneously)
    print(f"Running Parallel RANSAC on {pts1_good_n.shape[1]} matches...")
    R_rel, t_rel, inlier_mask_init = ransac_estimate_parallel(
        pts1_good_n, pts2_good_n, threshold=norm_threshold, rng=rng
    )
    
    if R_rel is None:
        print("Init pose estimation failed (Parallel RANSAC).")
        metrics["summary"]["status"] = "failed_init_pose"
        return metrics

    # Parallel RANSAC returns best pose directly
    P1 = np.eye(4)[:3]
    P2 = np.hstack((R_rel, t_rel.reshape(3, 1)))
    
    init_inliers = int(np.sum(inlier_mask_init))
    metrics["init"]["E_inliers"] = init_inliers
    print(f"Parallel RANSAC found {init_inliers} inliers.")

    # 2.4 Triangulate inliers
    cloud_points = []
    cloud_des_1 = []
    cloud_des_2 = []
    cloud_px_1 = []
    cloud_px_2 = []
    
    p1_in = pts1_good_n[:, inlier_mask_init]
    p2_in = pts2_good_n[:, inlier_mask_init]
    # Keep pixel coords for future PnP/checking
    p1_px_in = pts1_good[:, inlier_mask_init]
    p2_px_in = pts2_good[:, inlier_mask_init]
    
    match_indices = np.array(good_matches)[inlier_mask_init]

    # Minimum triangulation parallax angle filter (in degrees)
    min_parallax_deg = 1.5
    C1_init = np.zeros(3, dtype=np.float64)
    C2_init = _camera_center_world(R_rel, t_rel)
    cos_thresh_init = float(np.cos(np.deg2rad(min_parallax_deg)))

    for i in range(p1_in.shape[1]):
        X_h = triangulate_point_DLT(P1, P2, p1_in[:, i], p2_in[:, i])
        X = X_h[:3]
        
        # Cheirality Check (Depth > 0)
        X_cam2 = R_rel @ X + t_rel
        if X[2] > 0 and X_cam2[2] > 0:
            # Parallax angle filter (avoid near-zero baseline triangulation)
            v1 = X - C1_init
            v2 = X - C2_init
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if (not np.isfinite(n1)) or (not np.isfinite(n2)) or (n1 < 1e-12) or (n2 < 1e-12):
                continue
            with np.errstate(all="ignore"):
                cosang = float(np.dot(v1, v2) / (n1 * n2 + 1e-12))
            if (not np.isfinite(cosang)):
                continue
            cosang = float(np.clip(cosang, -1.0, 1.0))
            if cosang > cos_thresh_init:
                continue
            cloud_points.append(X)
            # Save descriptors for future matching
            cloud_des_1.append(des1[match_indices[i].queryIdx])
            cloud_des_2.append(des2[match_indices[i].trainIdx])
            cloud_px_1.append(kp1[match_indices[i].queryIdx].pt)
            cloud_px_2.append(kp2[match_indices[i].trainIdx].pt)
            
    cloud_points = np.array(cloud_points).T
    cloud_des_1 = np.array(cloud_des_1)
    cloud_des_2 = np.array(cloud_des_2)
    cloud_px_1 = np.array(cloud_px_1, dtype=np.float64)
    cloud_px_2 = np.array(cloud_px_2, dtype=np.float64)

    # Filter out far-away outliers
    cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2 = filter_points_distance_quantile(
        cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2
    )

    print(f"Initialized cloud with {cloud_points.shape[1]} points.")
    metrics["init"]["triangulated_kept"] = int(cloud_points.shape[1])

    # ========================================================
    # STEP 3: Resectioning loop (Sequential)
    # ========================================================
    cameras_pose = {}
    cameras_pose[idx1] = (np.eye(3), np.zeros(3))
    cameras_pose[idx2] = (R_rel, t_rel)
    
    # Processing Order: Prioritize images between init_pair
    processing_queue = []
    step = 1 if idx2 > idx1 else -1
    processing_queue.extend(range(idx1 + step, idx2, step))
    if step == 1:
        processing_queue.extend(range(idx2 + 1, len(images)))
        processing_queue.extend(range(idx1 - 1, -1, -1))
    else:
        processing_queue.extend(range(idx2 - 1, -1, -1))
        processing_queue.extend(range(idx1 + 1, len(images)))
    
    final_points = cloud_points
    
    for i in processing_queue:
        # Determine reference frame (neighbor that is already reconstructed)
        if (i - 1) in cameras_pose:
            ref_idx = i - 1
        elif (i + 1) in cameras_pose:
            ref_idx = i + 1
        else:
            print(f"Skipping {i}, no neighbor reconstructed.")
            continue
            
        print(f"Registering Image {i} against {ref_idx}...")
        
        kp_i, des_i = sift.detectAndCompute(images[i], None)
        kp_ref, des_ref = sift.detectAndCompute(images[ref_idx], None)
        
        # 3.1 Calculate Relative Rotation using Parallel RANSAC (Essential Matrix)
        # Needed for upgrading to absolute rotations before calculating T
        bf_seq = cv2.BFMatcher()
        matches_seq = bf_seq.knnMatch(des_ref, des_i, k=2)
        good_seq = [m for m, n in matches_seq if m.distance < 0.75 * n.distance]
        
        if len(good_seq) < 8:
            print(f"Not enough matches for rotation estimation {i}-{ref_idx}")
            continue
            
        p_ref_seq = np.float32([kp_ref[m.queryIdx].pt for m in good_seq]).T
        p_i_seq = np.float32([kp_i[m.trainIdx].pt for m in good_seq]).T
        
        # Normalize
        p_ref_n = normalize_points(p_ref_seq, K)
        p_i_n = normalize_points(p_i_seq, K)

        # Robust pairwise pose for potentially planar scenes
        R_rel_seq, t_rel_seq, mask_seq = ransac_estimate_parallel(
            p_ref_n,
            p_i_n,
            threshold=norm_threshold,
            num_iterations=2000,
            rng=rng,
        )
        if R_rel_seq is None or mask_seq is None or np.sum(mask_seq) < 8:
            print(f"Pose estimation failed (E/H parallel RANSAC) {ref_idx}-{i}")
            continue
                                                     
        # 3.2 Upgrade to Absolute Rotation
        # R_i = R_rel * R_ref (Note: R_rel here is ref -> i)
        R_ref_abs, _ = cameras_pose[ref_idx]
        R_i_abs = R_rel_seq @ R_ref_abs
        
        # 3.3 Robustly calculate Translation T using 2D-3D matches
        # Collect matches between Image i and existing Cloud
        corr1 = _collect_2d3d_correspondences(cloud_des_1, cloud_points, cloud_px_1, des_i, kp_i)
        corr2 = _collect_2d3d_correspondences(cloud_des_2, cloud_points, cloud_px_2, des_i, kp_i)
        
        valid_2d, valid_3d = _merge_unique_correspondences([corr1, corr2])
        
        if len(valid_2d) < 6:
            print(f"Image {i}: Not enough 2D-3D matches.")
            continue
            
        pts2d_np = np.array(valid_2d, dtype=np.float64).T
        pts3d_np = np.array(valid_3d, dtype=np.float64).T
        
        # Normalize 2D points for T estimation
        pts2d_norm = normalize_points(pts2d_np, K)
        
        # Run custom T estimation
        t_i_abs, t_inliers = estimate_T_robust(pts2d_norm, pts3d_np, R_i_abs, 
                                               threshold=norm_threshold*2.0,
                                               rng=rng)
        
        if t_i_abs is not None:
            num_inliers = np.sum(t_inliers)
            print(f"Image {i} localized. T-RANSAC Inliers: {num_inliers}/{pts2d_norm.shape[1]}")
            cameras_pose[i] = (R_i_abs, t_i_abs)

            # 3.4 Grow the point cloud by triangulating new matches between ref and i
            R_ref_w, t_ref_w = cameras_pose[ref_idx]
            P_ref = np.hstack([R_ref_w, np.asarray(t_ref_w, dtype=np.float64).reshape(3, 1)])
            P_i = np.hstack([R_i_abs, np.asarray(t_i_abs, dtype=np.float64).reshape(3, 1)])

            # For parallax thresholding on newly triangulated points
            min_parallax_deg = 1.5
            cos_thresh = float(np.cos(np.deg2rad(min_parallax_deg)))
            C_ref = _camera_center_world(R_ref_w, t_ref_w)
            C_i = _camera_center_world(R_i_abs, t_i_abs)

            # Use inliers from the pairwise pose estimation for triangulation candidates
            in_mask = np.asarray(mask_seq, dtype=bool)
            if np.sum(in_mask) >= 8:
                # Indices in keypoint arrays
                good_seq_arr = np.array(good_seq)
                good_in = good_seq_arr[in_mask]

                X_new_list = []
                d_ref_list = []
                d_i_list = []
                px_ref_list = []
                px_i_list = []

                p_ref_in = p_ref_n[:, in_mask]
                p_i_in = p_i_n[:, in_mask]

                # Cap additions per image to avoid runaway duplicates
                max_add = 1600
                for j in range(p_ref_in.shape[1]):
                    if len(X_new_list) >= max_add:
                        break
                    X_h = triangulate_point_DLT(P_ref, P_i, p_ref_in[:, j], p_i_in[:, j])
                    X = X_h[:3]

                    # Cheirality check in both cameras
                    X_cam_ref = R_ref_w @ X + np.asarray(t_ref_w, dtype=np.float64).reshape(3)
                    X_cam_i = R_i_abs @ X + np.asarray(t_i_abs, dtype=np.float64).reshape(3)
                    if (X_cam_ref[2] <= 0) or (X_cam_i[2] <= 0):
                        continue
                    if not np.all(np.isfinite(X)):
                        continue

                    # Parallax angle filter
                    v1 = X - C_ref
                    v2 = X - C_i
                    n1 = float(np.linalg.norm(v1))
                    n2 = float(np.linalg.norm(v2))
                    if (not np.isfinite(n1)) or (not np.isfinite(n2)) or (n1 < 1e-12) or (n2 < 1e-12):
                        continue
                    with np.errstate(all="ignore"):
                        cosang = float(np.dot(v1, v2) / (n1 * n2 + 1e-12))
                    if (not np.isfinite(cosang)):
                        continue
                    cosang = float(np.clip(cosang, -1.0, 1.0))
                    if cosang > cos_thresh:
                        continue

                    m = good_in[j]
                    X_new_list.append(X)
                    d_ref_list.append(des_ref[m.queryIdx])
                    d_i_list.append(des_i[m.trainIdx])
                    px_ref_list.append(kp_ref[m.queryIdx].pt)
                    px_i_list.append(kp_i[m.trainIdx].pt)

                cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2 = _append_to_cloud(
                    cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2,
                    X_new_list, d_ref_list, d_i_list, px_ref_list, px_i_list
                )
                final_points = cloud_points
                print(f"Cloud size after adding from {ref_idx}-{i}: {cloud_points.shape[1]}")
            
            # Record metrics
            metrics["pnp"][int(i)] = {
                "inliers": int(num_inliers),
                "matches": int(pts2d_norm.shape[1])
            }
        else:
            print(f"Image {i} failed T estimation.")

    # Optional final cleanup on the accumulated sparse cloud
    if filter_final_cloud:
        final_points, _, _, _, _ = filter_points_distance_quantile(
            cloud_points, cloud_des_1, cloud_des_2, cloud_px_1, cloud_px_2
        )

    # ========================================================
    # Output & Visualization
    # ========================================================
    cam_centers = []
    for i in sorted(cameras_pose.keys()):
        R, t = cameras_pose[i]
        C = -R.T @ t
        cam_centers.append(C)
        
    metrics["summary"]["localized_images"] = len(cameras_pose)
    metrics["summary"]["cloud_points"] = final_points.shape[1]

    # --------------------------------------------------------
    # STEP 4 (Bonus): Dense reconstruction with RoMa (optional)
    # --------------------------------------------------------
    dense_cloud = None
    dense_colors = None
    if use_roma_dense:
        dense_res = run_dense_reconstruction_roma(
            int(dataset_num),
            cameras_pose,
            K,
            img_names,
            confidence_thresh=float(roma_confidence_thresh),
            downsample_max_size=int(roma_downsample_max_size),
            rng=rng,
        )
        if dense_res is not None:
            dense_cloud = dense_res.points_3d
            dense_colors = dense_res.colors_rgb
            metrics["summary"]["dense_points"] = int(dense_cloud.shape[1])
            metrics["summary"]["dense_pairs_processed"] = int(dense_res.pairs_processed)
            metrics["summary"]["dense_pairs_skipped"] = int(dense_res.pairs_skipped)

            if save_plot_dir:
                ply_path = os.path.join(save_plot_dir, f"dense_{dataset_num}.ply")
                save_ply(ply_path, dense_cloud, dense_colors)
                print(f"Saved dense point cloud to {ply_path}")
        else:
            print("RoMa dense ran but returned no points (check dependencies or match confidence).")
    
    points_to_plot = dense_cloud if dense_cloud is not None else final_points

    if visualize or save_plot_dir:
        save_path = os.path.join(save_plot_dir, f"res_{dataset_num}.png") if save_plot_dir else None
        plot_3d_reconstruction_maybe(points_to_plot, cam_centers, show=visualize, save_path=save_path)
        
    print("\n" + "="*50)
    print(f"FINAL RESULTS FOR DATASET {dataset_num}")
    print("="*50)
    
    total_imgs = len(img_names)
    reg_imgs = len(cameras_pose)
    sparse_count = final_points.shape[1]
    
    print(f"Images Registered: {reg_imgs}/{total_imgs}")
    print(f"Sparse Points:     {sparse_count}")
    
    if dense_cloud is not None:
        dense_count = dense_cloud.shape[1]
        print(f"Dense Points:      {dense_count}")
    else:
        print("Dense Points:      N/A (Not used or failed)")
        
    print("="*50 + "\n")
        
    return metrics

def run_all_datasets(start=1, end=9, *, visualize=False, use_roma_dense=True, save_plot_dir=None, seed: int | None = None):
    for d in range(start, end + 1):
        try:
            run_sfm(
                d,
                visualize=visualize,
                use_roma_dense=use_roma_dense,
                save_plot_dir=save_plot_dir,
                seed=seed,
            )
        except Exception as e:
            print(f"Dataset {d} error: {e}")

if __name__ == "__main__":
    run_all_datasets(1, 8, visualize=False, use_roma_dense=True, save_plot_dir="./final_output")
    # run_sfm(7, visualize=False, use_roma_dense=True, save_plot_dir="./output")