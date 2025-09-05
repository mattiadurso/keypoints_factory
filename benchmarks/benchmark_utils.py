import cv2
import torch
import random
import argparse
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fix_rng(seed=42):
    """  Set seeds for reproducibility
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # optional: turn off TF32 to avoid tiny nondet diffs on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # If using scaled_dot_product_attention, force the math kernel (deterministic)
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash(False)
        sdp_kernel.enable_mem_efficient(False)
        sdp_kernel.enable_math(True)
    except Exception:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
    np.random.seed(seed)
    random.seed(seed)


def parse_pair(pair):
    parts = pair.strip().split()
    img1, img2 = parts[0], parts[1]
    nums = list(map(float, parts[2:]))

    K1 = np.array(nums[0:9]).reshape(3, 3)
    K2 = np.array(nums[9:18]).reshape(3, 3)
    R = np.array(nums[18:27]).reshape(3, 3)
    t = np.array(nums[27:30]).reshape(3, 1)

    # # build E and F
    # tx = np.array([[0, -t[2,0], t[1,0]],
    #                [t[2,0], 0, -t[0,0]],
    #                [-t[1,0], t[0,0], 0]])
    # E = tx @ R
    # F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    # return img1, img2, K1, K2, R, t, E, F

    return img1, img2, K1, K2, R, t


def print_metrics(wrapper, metrics: dict):
    print(f"\nEvaluation results for {wrapper.name}:")
    for k, v in metrics.items():
        v = v if k == 'inlier' else v*100
        print(f"{k:<8}: {v:.1f}")


def get_best_device(verbose=False):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if verbose:
        print(f"Fastest device found is: {device}")
    return device


def recover_pose(E, kpts0, kpts1, K0, K1, mask):
    best_num_inliers = 0
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0_n = (K0inv @ (kpts0-K0[None, :2, 2]).T).T
    kpts1_n = (K1inv @ (kpts1-K1[None, :2, 2]).T).T

    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = \
            cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t, mask.ravel() > 0)
    return ret


def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0-K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1-K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = \
                cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans
