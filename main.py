import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from project_helpers import get_dataset_info
from project_helpers import homography_to_RT


def _imread_color_no_exif_rotation(path: str):
    flags = cv2.IMREAD_COLOR
    # OpenCV may auto-rotate based on EXIF orientation; disable for consistency.
    if hasattr(cv2, "IMREAD_IGNORE_ORIENTATION"):
        flags |= cv2.IMREAD_IGNORE_ORIENTATION
    return cv2.imread(path, flags)

# Optional dense reconstruction (RoMa)
try:
    from roma_dense import run_dense_reconstruction_roma, save_ply
    ROMA_DENSE_AVAILABLE = True
except Exception:
    run_dense_reconstruction_roma = None
    save_ply = None
    ROMA_DENSE_AVAILABLE = False

# ==========================================
# 辅助函数 / 核心算法实现
# ==========================================

def estimate_H_DLT(x1, x2):
    """
    使用 DLT 算法从至少 4 对点估计单应性矩阵 H [cite: 144]。
    x1, x2: 3xN 归一化坐标
    """
    n = x1.shape[1]
    if n < 4: return None
    
    # 转换为非齐次坐标 (u, v) 以构建矩阵 A
    # 注意：输入通常已经是归一化平面的点 (z=1)，但为了安全重新除以 z
    with np.errstate(all='ignore'):
        u1, v1 = x1[0]/x1[2], x1[1]/x1[2]
        u2, v2 = x2[0]/x2[2], x2[1]/x2[2]
    
    A = []
    # 使用前 4 个点构建矩阵 (通常 RANSAC 中只传 4 个点)
    for i in range(n):
        x, y = u1[i], v1[i]
        xp, yp = u2[i], v2[i]
        # DLT 行公式: x' = Hx -> x' x Hx = 0
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
        
    A = np.asarray(A)
    # SVD 解 Ax = 0
    try:
        U, S, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
        
    H = Vt[-1].reshape(3, 3)
    return H

def check_pose_inliers(R, t, x1, x2, threshold_sq):
    """
    辅助函数：检查给定 (R, t) 的内点数和几何有效性。
    用于评估来自 E 或 H 的解 [cite: 147, 161]。
    """
    t = t.reshape(3, 1)
    # 构建本质矩阵 E = [t]x R
    Tx = np.array([[0, -t[2,0], t[1,0]],
                   [t[2,0], 0, -t[0,0]],
                   [-t[1,0], t[0,0], 0]])
    E_hyp = Tx @ R
    
    # 1. 计算 Sampson 误差确定内点 [cite: 147]
    d2 = compute_sampson_errors(E_hyp, x1, x2)
    inliers = d2 < threshold_sq
    count = np.sum(inliers)
    
    # 如果内点太少，跳过昂贵的三角化检查
    if count < 5: return 0, inliers, False

    # 2. Cheirality 检查 (点必须在两个相机前方) [cite: 128]
    # 仅三角化第一个内点进行快速检查
    idx = np.where(inliers)[0][0]
    P1 = np.eye(4)[:3]
    P2 = np.hstack((R, t))
    
    # 使用现有的单点三角化函数
    X = triangulate_point_DLT(P1, P2, x1[:, idx], x2[:, idx])
    
    # 检查两个相机坐标系下的 Z > 0
    X_cam2 = R @ X[:3] + t.flatten()
    is_valid = (X[2] > 0) and (X_cam2[2] > 0)
    
    return count, inliers, is_valid

def ransac_estimate_parallel(x1s, x2s, threshold=0.001, num_iterations=2000):
    """
    并行 RANSAC：同时搜索 E (8点法) 和 H (4点法) [cite: 138, 152]。
    直接返回最佳的 (R, t) 以及内点掩码。
    """
    threshold_sq = threshold ** 2
    n_points = x1s.shape[1]
    
    best_score = -1
    best_Rt = (None, None)
    best_inliers = None
    
    if n_points < 8:
        return None, None, None

    # 预先构建 Identity 矩阵用于 select_correct_pose
    P1_identity = np.eye(4)[:3]

    for _ in range(num_iterations):
        # 1. 采样 8 个点 (E 需要 8 个，H 只需要其中 4 个)
        sample_idx = np.random.choice(n_points, 8, replace=False)
        x1_sample, x2_sample = x1s[:, sample_idx], x2s[:, sample_idx]
        
        # ==========================================
        # 分支 A: 估计 Essential Matrix (8-point) [cite: 142]
        # ==========================================
        E_approx = estimate_F_DLT(x1_sample, x2_sample)
        E_cand = enforce_essential(E_approx)
        
        if E_cand is not None:
            # 分解 E 得到 4 个解
            candidates = extract_P_from_E(E_cand)
            
            # 使用 Cheirality 选出唯一解 (R, t) (针对样本集) [cite: 147]
            # 这里复用现有的 select_correct_pose 快速筛选
            res = select_correct_pose(candidates, P1_identity, x1_sample, x2_sample)
            
            if res is not None:
                R_E, t_E = res
                # 在所有数据点上评分
                count, inliers, valid = check_pose_inliers(R_E, t_E, x1s, x2s, threshold_sq)
                
                if valid and count > best_score:
                    best_score = count
                    best_Rt = (R_E, t_E)
                    best_inliers = inliers

        # ==========================================
        # 分支 B: 估计 Homography (4-point) [cite: 144]
        # ==========================================
        # 使用样本的前4个点
        H_cand = estimate_H_DLT(x1_sample[:, :4], x2_sample[:, :4])
        
        if H_cand is not None:
            # 从 H 分解出可能的 (R, t) 列表 [cite: 145]
            # homography_to_RT 返回 [(R1, t1), (R2, t2), ...]
            try:
                RTs = homography_to_RT(H_cand)
                RTs = np.asarray(RTs)
                # project_helpers.homography_to_RT returns shape (2, 3, 4): [R|t] for 2 hypotheses.
                if RTs.ndim == 3 and RTs.shape[1:] == (3, 4):
                    for k in range(RTs.shape[0]):
                        R_H = RTs[k, :, :3]
                        t_H = RTs[k, :, 3]
                        # 检查 H 分解出的每个解
                        count, inliers, valid = check_pose_inliers(R_H, t_H, x1s, x2s, threshold_sq)
                        if valid and count > best_score:
                            best_score = count
                            best_Rt = (R_H, t_H)
                            best_inliers = inliers
            except Exception:
                pass # 忽略分解失败的情况，继续循环

    return best_Rt[0], best_Rt[1], best_inliers

def _error_stats(errors_px):
    """Return robust summary stats for a 1D array of pixel errors."""
    if errors_px is None:
        return {"n": 0}
    arr = np.asarray(errors_px, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _format_stats(stats):
    if not stats or stats.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={stats['n']} median={stats['median']:.2f}px "
        f"p95={stats['p95']:.2f}px max={stats['max']:.2f}px"
    )


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
    """Hartley isotropic normalization for 2D homogeneous points.

    Returns (T, x_norm) where x_norm = T @ x_h.
    """
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
    if not np.isfinite(mean_d) or mean_d < 1e-12:
        s = 1.0
    else:
        s = np.sqrt(2.0) / mean_d

    T = np.array([
        [s, 0.0, -s * centroid[0]],
        [0.0, s, -s * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    x_norm = T @ x_h
    return T, x_norm


def estimate_F_DLT(x1s, x2s):
    """Estimate Fundamental/Essential matrix using normalized 8-point algorithm.

    NOTE: When x1s/x2s are already in camera-normalized coordinates, this
    effectively estimates the Essential matrix up to scale.
    """
    valid_idx = np.all(np.isfinite(x1s), axis=0) & np.all(np.isfinite(x2s), axis=0)
    x1 = np.asarray(x1s[:, valid_idx], dtype=np.float64)
    x2 = np.asarray(x2s[:, valid_idx], dtype=np.float64)

    n_points = x1.shape[1]
    if n_points < 8:
        return None

    # Hartley normalization improves numerical stability.
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

    # Enforce rank-2 constraint before denormalization.
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
    if np.linalg.det(U @ Vt) < 0: Vt = -Vt # Ensure proper rotation component
    S_new = np.diag([1, 1, 0])
    E_constrained = U @ S_new @ Vt
    return E_constrained


def compute_sampson_errors(E, x1s, x2s):
    """Compute Sampson approximation of geometric reprojection error.

    Returns squared Sampson errors (in normalized image coordinates).
    """
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


def ransac_estimate_E(x1s, x2s, threshold=0.001, num_iterations=2000):
    """
    RANSAC loop for Essential Matrix estimation[cite: 117, 139].
    x1s, x2s: Normalized coordinates (3xN)
    """
    best_E, best_inliers = None, None
    max_inliers_count = 0
    n_points = x1s.shape[1]
    
    # 简单的随机采样
    for _ in range(num_iterations):
        if n_points < 8: break
        sample_idx = np.random.choice(n_points, 8, replace=False)
        x1_sample, x2_sample = x1s[:, sample_idx], x2s[:, sample_idx]
        
        E_approx = estimate_F_DLT(x1_sample, x2_sample)
        E_cand = enforce_essential(E_approx)
        if E_cand is None: continue
        
        # Use Sampson distance (squared) for more stable inlier selection.
        d2 = compute_sampson_errors(E_cand, x1s, x2s)
        inliers = d2 < (threshold ** 2)
        inliers_count = np.sum(inliers)
        
        if inliers_count > max_inliers_count:
            max_inliers_count = inliers_count
            best_E = E_cand
            best_inliers = inliers
            
    return best_E, best_inliers


def extract_P_from_E(E):
     """Extract 4 possible camera matrices P2 from E[cite: 143, 123]."""
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
    """
    Select the correct (R, t) configuration using Cheirality check[cite: 128, 147].
    Points must be in front of both cameras.
    """
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
    Linear estimation of T given R and 2D-3D matches[cite: 41, 106].
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
        
        # Eliminating lambda (depth):
        # u = (Xr + Tx)/(Zr + Tz) -> Tx - u*Tz = u*Zr - Xr
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


def estimate_T_robust(points2D, points3D, R, threshold=0.005, num_iterations=1000):
    """
    Robustly estimate T using RANSAC[cite: 106, 129].
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
    
    for i in range(num_iterations):
        # 1. Minimal sample (2 points) [cite: 106]
        idx = np.random.choice(N, 2, replace=False)
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
# 匹配和辅助逻辑
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
):
    print(f"--- Running SfM on Dataset {dataset_num} ---")
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset_num)
    if K is None: return

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
    
    # 计算归一化阈值 (Project suggestion: threshold / f) [cite: 212]
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
    
    # 2.1 归一化坐标 
    pts1_good_n = normalize_points(pts1_good, K)
    pts2_good_n = normalize_points(pts2_good, K)
    
    # 2.2 使用并行 RANSAC (同时搜索 E 和 H) [cite: 138]
    print(f"Running Parallel RANSAC on {pts1_good_n.shape[1]} matches...")
    R_rel, t_rel, inlier_mask_init = ransac_estimate_parallel(
        pts1_good_n, pts2_good_n, threshold=norm_threshold
    )
    
    if R_rel is None:
        print("Init pose estimation failed (Parallel RANSAC).")
        metrics["summary"]["status"] = "failed_init_pose"
        return metrics

    # 并行 RANSAC 直接返回了最佳位姿，不需要再做分解和选择
    P1 = np.eye(4)[:3]
    P2 = np.hstack((R_rel, t_rel.reshape(3, 1)))
    
    init_inliers = int(np.sum(inlier_mask_init))
    metrics["init"]["E_inliers"] = init_inliers
    print(f"Parallel RANSAC found {init_inliers} inliers.")

    # 2.4 Triangulate inliers (后续代码保持不变)
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

    for i in range(p1_in.shape[1]):
        X_h = triangulate_point_DLT(P1, P2, p1_in[:, i], p2_in[:, i])
        X = X_h[:3]
        
        # Cheirality Check (Depth > 0)
        X_cam2 = R_rel @ X + t_rel
        if X[2] > 0 and X_cam2[2] > 0:
            cloud_points.append(X)
            # 保存 descriptors 用于后续匹配
            cloud_des_1.append(des1[match_indices[i].queryIdx])
            cloud_des_2.append(des2[match_indices[i].trainIdx])
            cloud_px_1.append(kp1[match_indices[i].queryIdx].pt)
            cloud_px_2.append(kp2[match_indices[i].trainIdx].pt)
            
    cloud_points = np.array(cloud_points).T
    cloud_des_1 = np.array(cloud_des_1)
    cloud_des_2 = np.array(cloud_des_2)
    cloud_px_1 = np.array(cloud_px_1, dtype=np.float64)
    cloud_px_2 = np.array(cloud_px_2, dtype=np.float64)

    print(f"Initialized cloud with {cloud_points.shape[1]} points.")
    metrics["init"]["triangulated_kept"] = int(cloud_points.shape[1])

    # ========================================================
    # STEP 3: Resectioning loop (Sequential) [cite: 37, 38, 41]
    # ========================================================
    cameras_pose = {}
    cameras_pose[idx1] = (np.eye(3), np.zeros(3))
    cameras_pose[idx2] = (R_rel, t_rel)
    
    # 定义处理顺序：优先处理 init_pair 之间的图像（这对 dataset 2 很关键，init=(0,8) 否则队列为空）
    # 然后再处理 init_pair 之外的两侧。
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
        # We need this to calculate Relative Rotation
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
        
        # 3.1 Calculate Relative Rotation using Essential Matrix RANSAC [cite: 37]
        # (Needed because we must 'Upgrade to absolute rotations' before calculating T)
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

        # Robust pairwise pose for potentially planar scenes:
        # Use parallel RANSAC (E vs H) to avoid degeneracy of E-only estimation.
        R_rel_seq, t_rel_seq, mask_seq = ransac_estimate_parallel(
            p_ref_n,
            p_i_n,
            threshold=norm_threshold,
            num_iterations=2000,
        )
        if R_rel_seq is None or mask_seq is None or np.sum(mask_seq) < 8:
            print(f"Pose estimation failed (E/H parallel RANSAC) {ref_idx}-{i}")
            continue
                                                     
        # 3.2 Upgrade to Absolute Rotation [cite: 38, 69]
        # R_i = R_rel * R_ref (注意：这里的 R_rel 是 ref -> i 的旋转)
        # Camera pose definition: X_cam = R * X_world + t
        # X_i = R_rel * X_ref + t_rel
        # X_i = R_rel * (R_ref * X_w + t_ref) + t_rel
        # X_i = (R_rel * R_ref) * X_w + ...
        R_ref_abs, _ = cameras_pose[ref_idx]
        R_i_abs = R_rel_seq @ R_ref_abs
        
        # 3.3 Robustly calculate Translation T using 2D-3D matches [cite: 41, 106]
        # Collect matches between Image i and existing Cloud
        corr1 = _collect_2d3d_correspondences(cloud_des_1, cloud_points, cloud_px_1, des_i, kp_i)
        corr2 = _collect_2d3d_correspondences(cloud_des_2, cloud_points, cloud_px_2, des_i, kp_i)
        
        # Use E-test filter provided in code? Skipped for brevity, rely on T-RANSAC
        valid_2d, valid_3d = _merge_unique_correspondences([corr1, corr2])
        
        if len(valid_2d) < 6:
            print(f"Image {i}: Not enough 2D-3D matches.")
            continue
            
        pts2d_np = np.array(valid_2d, dtype=np.float64).T
        pts3d_np = np.array(valid_3d, dtype=np.float64).T
        
        # Normalize 2D points for T estimation
        pts2d_norm = normalize_points(pts2d_np, K)
        
        # Run custom T estimation [cite: 129]
        t_i_abs, t_inliers = estimate_T_robust(pts2d_norm, pts3d_np, R_i_abs, 
                                               threshold=norm_threshold*2.0) # slightly looser for PnP
        
        if t_i_abs is not None:
            num_inliers = np.sum(t_inliers)
            print(f"Image {i} localized. T-RANSAC Inliers: {num_inliers}/{pts2d_norm.shape[1]}")
            cameras_pose[i] = (R_i_abs, t_i_abs)

            # 3.4 Grow the point cloud by triangulating new matches between ref and i.
            # This is essential for multi-view datasets like #2.
            R_ref_w, t_ref_w = cameras_pose[ref_idx]
            P_ref = np.hstack([R_ref_w, np.asarray(t_ref_w, dtype=np.float64).reshape(3, 1)])
            P_i = np.hstack([R_i_abs, np.asarray(t_i_abs, dtype=np.float64).reshape(3, 1)])

            # Use inliers from the pairwise pose estimation for triangulation candidates.
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
        if not ROMA_DENSE_AVAILABLE:
            print("RoMa dense module not available; skipping. (Install torch + romatch, or check import)")
        else:
            dense_res = run_dense_reconstruction_roma(
                int(dataset_num),
                cameras_pose,
                K,
                img_names,
                confidence_thresh=float(roma_confidence_thresh),
                downsample_max_size=int(roma_downsample_max_size),
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
                print("RoMa dense ran but returned no points (missing deps or too few confident matches).")
    
    points_to_plot = dense_cloud if dense_cloud is not None else final_points

    if visualize or save_plot_dir:
        save_path = os.path.join(save_plot_dir, f"res_{dataset_num}.png") if save_plot_dir else None
        plot_3d_reconstruction_maybe(points_to_plot, cam_centers, show=visualize, save_path=save_path)
        
    return metrics

def run_all_datasets(start=1, end=9, *, visualize=False, use_roma_dense=True, save_plot_dir=None):
    for d in range(start, end + 1):
        try:
            run_sfm(d, visualize=visualize, use_roma_dense=use_roma_dense, save_plot_dir=save_plot_dir)
        except Exception as e:
            print(f"Dataset {d} error: {e}")

if __name__ == "__main__":
    # run_all_datasets(1, 7, visualize=False, use_roma_dense=True, save_plot_dir="./sfm_plots_copy")
    run_sfm(5, visualize=True, use_roma_dense=False, save_plot_dir="./output")