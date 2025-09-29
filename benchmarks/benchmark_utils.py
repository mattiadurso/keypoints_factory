import cv2
import h5py
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn.functional as F

from typing import List, Dict, Iterable, Optional, Mapping

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fake_tqdm(x, **kwargs):
    return x


def parse_poses(poses_file, benchmark_name):
    """Parse poses from a given file based on the benchmark format."""
    if benchmark_name in [
        "megadepth1500",
        "graz_high_res",
        "megadepth_view",
        "megadepth_air2ground",
    ]:
        return parse_md1500_poses(poses_file)
    elif benchmark_name in ["scannet1500"]:
        # Implement parse_scannet1500_poses if needed
        raise NotImplementedError("Parsing for scannet1500 not implemented yet.")
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")


def load_depth(depth_path, scale_factor, target):
    """Load depth data from a given file."""
    device, dtype = target.device, target.dtype
    depth = torch.tensor(h5py.File(depth_path, "r")["depth"][()], device=device).float()
    # crop multiple of 16 for compatibility with disk
    H, W = depth.shape
    depth = depth[: (H // 16) * 16, : (W // 16) * 16]  # same as image in img_from_numpy

    if scale_factor != 1.0:  # mostly for GHR, with md is must be 1.0
        depth = F.interpolate(
            depth[None, None],
            scale_factor=scale_factor,
            mode="nearest",
            align_corners=False,
        )[0, 0]

    return depth.to(device=device, dtype=dtype)


def parse_md1500_poses(poses_file):

    with open(poses_file, "r") as f:
        lines = f.readlines()

    view_dict = {}
    for line in lines:
        if line[0] == "#":
            continue
        elems = line.split()
        img_path = elems[0]
        R = np.array(elems[1:10], dtype=float).reshape(3, 3)
        t = np.array(elems[10:13], dtype=float).reshape(3, 1)
        camera_model = elems[13]
        image_size = np.array(elems[14:16], dtype=int)
        K_ = np.array(elems[16:], dtype=float)  # fx, fy, cx, cy
        K = np.eye(3)
        K[0, 0] = K_[0]
        K[1, 1] = K_[1]
        K[0, 2] = K_[2]
        K[1, 2] = K_[3]
        P = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1.0]).reshape(1, 4)))

        view_dict[img_path] = {
            "img_path": img_path,
            "K": torch.from_numpy(K),
            "R": torch.from_numpy(R),
            "t": torch.from_numpy(t),
            "P": torch.from_numpy(P),
            "camera_model": camera_model,
            "image_size": image_size,
        }

    return view_dict


def process_pose_estimation_batch(pair_matches_data, th, worker_seed=None):
    """Process pose estimation for a batch of pairs."""
    fix_rng(42)

    results = []

    for (img1, img2), matches, kpts1, kpts2, K1, K2, R, t in pair_matches_data:
        try:
            if len(matches) < 5:
                results.append((img1, img2, 180, 180, 180, 0))
                continue

            # Get matched keypoints
            matched_kpts1 = kpts1[matches[:, 0]]
            matched_kpts2 = kpts2[matches[:, 1]]

            # Shuffle matches
            shuffling = np.random.permutation(len(matched_kpts1))
            matched_kpts1 = matched_kpts1[shuffling]
            matched_kpts2 = matched_kpts2[shuffling]

            # Pose estimation
            threshold = th
            norm_threshold = threshold / (
                np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2]))
            )

            R_est, t_est, mask = estimate_pose(
                matched_kpts1, matched_kpts2, K1, K2, norm_threshold, conf=0.99999
            )

            T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)
            e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
            e_pose = min(10, max(e_t, e_R))
            num_inliers = np.sum(mask)

            results.append((img1, img2, e_t, e_R, e_pose, num_inliers))
            logger.debug(
                f"Pair ({img1}, {img2}): e_t={e_t:.2f}, e_R={e_R:.2f}, e_pose={e_pose:.2f}, inliers={num_inliers}"
            )

        except Exception as e:
            results.append((img1, img2, 180, 180, 180, 0))

            logging.debug(f"Error processing pair ({img1}, {img2}): {e}")

    return results


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


def parse_pair(pair, benchmark_name):
    """Parse a line from a .pairs file"""
    if benchmark_name in [
        "megadepth1500",
        "graz_high_res",
        "megadepth_view",
        "megadepth_air2ground",
    ]:
        # Pose is stores as "... flat(R) flat(t)""
        parts = pair.strip().split()
        img1, img2 = parts[0], parts[1]
        nums = list(map(float, parts[2:]))

        K1 = np.array(nums[0:9]).reshape(3, 3)
        K2 = np.array(nums[9:18]).reshape(3, 3)
        R = np.array(nums[18:27]).reshape(3, 3)
        t = np.array(nums[27:30]).reshape(3, 1)

    elif benchmark_name in ["scannet1500"]:
        # Pose is stored as a "... flat(P)" with P = [R|t;0 0 0 1]
        parts = pair.strip().split()
        img1, img2 = parts[0], parts[1]
        nums = list(map(float, parts[2:]))
        K1 = np.array(nums[0:9]).reshape(3, 3)
        K2 = np.array(nums[9:18]).reshape(3, 3)
        P = np.array(nums[18:34]).reshape(4, 4)
        R = P[:3, :3]
        t = P[:3, 3:].reshape(3, 1)

    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    return img1, img2, K1, K2, R, t


def print_metrics(wrapper, metrics: dict):
    """Pretty print metrics from a benchmark wrapper"""
    print(f"\nEvaluation results for {wrapper.name}:")
    for k, v in metrics.items():
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


def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.999999, maxIters=10_000):
    """Estimate pose using essential matrix"""
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0,
        kpts1,
        cameraMatrix=np.eye(3),
        threshold=norm_thresh,
        prob=conf,
        maxIters=maxIters,
        method=cv2.RANSAC,
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


def print_table(
    rows: List[Dict[str, object]],
    *,
    columns: Optional[Iterable[str]] = None,
    left_align: Optional[Iterable[str]] = None,
    right_align: Optional[Iterable[str]] = None,
    min_widths: Optional[Mapping[str, int]] = None,
    float_format: str = ".3f",
    fill_missing: str = "",
    clip_widths: Optional[Mapping[str, int]] = None,
    header: bool = True,
    sep: str = " | ",
) -> None:
    """
    Pretty-print a table from a list of dicts (assumes all dicts share the same keys).

    Args:
        rows: List of row dicts.
        columns: Optional explicit column order. Defaults to keys() of the first row.
        left_align: Columns to force left-align (others may be auto-detected).
        right_align: Columns to force right-align.
        min_widths: Dict of minimum widths per column (e.g., {"Method": 18, "Desc": 12}).
        float_format: Format spec for floats (e.g., ".2f" → 2 decimals).
        fill_missing: String to use when a key is missing in a row.
        clip_widths: Dict of maximum widths; values longer than the limit are clipped with “…” .
        header: Whether to print a header row and separator.
        sep: Column separator string.
    """
    if not rows:
        print("(empty)")
        return

    # Column order
    cols = list(columns) if columns is not None else list(rows[0].keys())

    # Alignment sets
    left_align = set(left_align or ())
    right_align = set(right_align or ())

    # Detect numeric columns if not explicitly aligned
    def is_numeric_value(v: object) -> bool:
        if isinstance(v, (int, float)):
            return True
        # Accept numeric-looking strings, ignore empty/missing
        if isinstance(v, str) and v.strip():
            try:
                float(v)
                return True
            except ValueError:
                return False
        return False

    numeric_cols = set()
    for c in cols:
        # If user forced alignment, skip auto-detect for that col
        if c in left_align or c in right_align:
            continue
        # Numeric if every non-empty value parses as number
        is_numeric = True
        for r in rows:
            v = r.get(c, fill_missing)
            if v == fill_missing or v is None or (isinstance(v, str) and v == ""):
                continue
            if not is_numeric_value(v):
                is_numeric = False
                break
        if is_numeric:
            numeric_cols.add(c)

    # Widths
    min_widths = dict(min_widths or {})
    widths = {c: max(len(str(c)), min_widths.get(c, 0)) for c in cols}

    def format_cell(v: object, col: str) -> str:
        # Numeric formatting
        if isinstance(v, float):
            return format(v, float_format)
        # Numeric-looking strings: leave as-is (assumed already formatted)
        return str(v)

    # Compute widths from data
    for r in rows:
        for c in cols:
            v = r.get(c, fill_missing)
            s = format_cell(v, c)
            widths[c] = max(widths[c], len(s))

    # Apply clipping limits
    clip_widths = dict(clip_widths or {})

    def clip(s: str, w: int, maxw: Optional[int]) -> str:
        if maxw is None or maxw <= 0 or len(s) <= maxw:
            return s
        # keep room for ellipsis
        return s[: max(0, maxw - 1)] + "…"

    # Render header
    def align_text(s: str, col: str) -> str:
        # Priority: explicit align → auto numeric → default left
        if col in right_align:
            return s.rjust(widths[col])
        if col in left_align or col not in numeric_cols:
            return s.ljust(widths[col])
        return s.rjust(widths[col])

    if header:
        hdr = sep.join(
            align_text(clip(str(c), widths[c], clip_widths.get(c)), c) for c in cols
        )
        bar = sep.join("-" * widths[c] for c in cols)
        print(hdr)
        print(bar)

    # Render rows
    for r in rows:
        cells = []
        for c in cols:
            raw = r.get(c, fill_missing)
            s = format_cell(raw, c)
            s = clip(s, widths[c], clip_widths.get(c))
            cells.append(align_text(s, c))
        print(sep.join(cells))
