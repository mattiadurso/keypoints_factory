import cv2
import os
import pandas as pd
import json
from pathlib import Path
import random
import torch
import argparse
import numpy as np


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fix_rng(seed=42):
    """Set seeds for reproducibility"""
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
    """Parse a line from a .pairs file"""
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
    """Pretty print metrics from a benchmark wrapper"""
    print(f"\nEvaluation results for {wrapper.name}:")
    for k, v in metrics.items():
        v = v if k == "inlier" else v * 100
        print(f"{k:<8}: {v:.1f}")


def get_best_device(verbose=False):
    """Get the best available device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
        if verbose:
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        if verbose:
            print("Using CPU device")
    return device


def recover_pose(E, kpts0, kpts1, K0, K1, mask):
    """Recover pose from essential matrix"""
    best_num_inliers = 0
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0_n = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1_n = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T

    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t, mask.ravel() > 0)
    return ret


def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    """Estimate pose using essential matrix"""
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret


def angle_error_mat(R1, R2):
    """Compute angle error between two rotation matrices"""
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    """Compute angle error between two vectors"""
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    """Compute pose error given ground truth and estimated pose"""
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    """Compute AUC for pose estimation"""
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


def display_hpatches_results(
    results_file="hpatches/results/results.json",
    partition="overall",
    method=None,
    ths=None,  # Allow None as default
    tostring=True,
):
    """
    Display HPatches benchmark results in a formatted DataFrame.

    Args:
        results_file: Path to the results JSON file
        partition: Which partition to display ('overall', 'i', 'v')
        method: Specific method to display (if None, displays all methods)
        ths: List of thresholds to display (if None, auto-detect from results)
        tostring: If True, print the DataFrame as a string

    Returns:
        pandas.DataFrame: Formatted results table
    """
    # Load results
    if isinstance(results_file, str):
        results_file = Path(results_file)

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    if not data:
        print("No results found in the file.")
        return pd.DataFrame()

    # Filter by specific method if provided
    if method is not None:
        if method in data:
            data = {method: data[method]}
        else:
            print(f"Method '{method}' not found in results.")
            return pd.DataFrame()

    # Auto-detect thresholds if not provided
    if ths is None:
        ths = []
        for method_key, results in data.items():
            if partition in results:
                partition_data = results[partition]
                for key in partition_data.keys():
                    if key.startswith("repeatability_"):
                        thr = key.split("_")[-1]
                        try:
                            thr_val = int(float(thr))
                            if thr_val not in ths:
                                ths.append(thr_val)
                        except ValueError:
                            continue
        ths = sorted(ths) if ths else [1, 2, 3]  # Default fallback

    # Create DataFrame
    rows = []

    for method_key, results in data.items():
        if partition not in results:
            continue

        partition_data = results[partition]

        # Base row info
        row = {
            "Method": method_key.split("_")[0],  # Extract method name
        }

        # Add metrics for all thresholds - GROUPED BY METRIC TYPE
        for thr in ths:
            # Repeatability
            rep_key = f"repeatability_{thr}"
            if rep_key in partition_data:
                rep_data = partition_data[rep_key]
                row[f"Rep@{thr}"] = f"{rep_data.get('mean', 0.0)*100:.1f}"

        for thr in ths:
            # Matching Accuracy
            ma_key = f"matching_accuracy_{thr}"
            if ma_key in partition_data:
                ma_data = partition_data[ma_key]
                row[f"MA@{thr}"] = f"{ma_data.get('mean', 0.0)*100:.1f}"

        for thr in ths:
            # Matching Score
            ms_key = f"matching_score_{thr}"
            if ms_key in partition_data:
                ms_data = partition_data[ms_key]
                row[f"MS@{thr}"] = f"{ms_data.get('mean', 0.0)*100:.1f}"

        for thr in ths:
            # Homography Accuracy
            ha_key = f"homography_accuracy_{thr}"
            if ha_key in partition_data:
                ha_data = partition_data[ha_key]
                row[f"HA@{thr}"] = f"{ha_data.get('mean', 0.0)*100:.1f}"

        rows.append(row)

    if not rows:
        print(f"No results found for partition '{partition}'")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by method name for consistent ordering
    if len(df) > 1:
        df = df.sort_values(["Method"], ascending=[True])

    # Reset index
    df.reset_index(drop=True, inplace=True)

    if tostring:
        print(df.to_string(index=False))
    return df
