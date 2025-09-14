import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings

warnings.filterwarnings("ignore")

import cv2
import gc
import torch
import numpy as np
import pandas as pd
import time
import json
import imageio.v3 as io
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from datetime import datetime
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial

from benchmarks.benchmark_utils import (
    compute_repeatability,
    fix_rng,
    fix_worker_rng,
)
from matchers.mnn import MNN

logger = logging.getLogger(__name__)


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


def process_metrics_batch(batch_data, thresholds, worker_seed=None):
    """Process metrics computation for a batch of matches."""
    if worker_seed is not None:
        fix_worker_rng(worker_seed)

    results = []

    for (
        (seq_name, img_idx),
        matches,
        kp_ref,
        kp_cur,
        homography,
        img_shape,
    ) in batch_data:
        try:
            # Convert keypoint data back to OpenCV keypoints
            def data_to_cv_keypoints(kp_data):
                if kp_data is None or len(kp_data) == 0:
                    return []
                return [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0) for pt in kp_data]

            kp_ref_cv = data_to_cv_keypoints(kp_ref)
            kp_cur_cv = data_to_cv_keypoints(kp_cur)

            result = {
                "sequence_name": seq_name,
                "img_idx": img_idx,
                "keypoints": len(kp_cur) if kp_cur is not None else 0,
            }

            # Compute metrics for all thresholds
            for threshold in thresholds:
                # Compute repeatability
                repeatability = 0.0
                if kp_ref_cv and kp_cur_cv:
                    try:
                        repeatability = compute_repeatability(
                            kp_ref_cv, kp_cur_cv, homography, threshold=threshold
                        )
                    except Exception as e:
                        repeatability = 0.0

                # Compute MMA and MS using pre-computed matches
                matching_accuracy = 0.0  # MMA - percentage of correct matches
                matching_score = 0.0  # MS - correct matches / avg keypoints in overlap
                correct_matches = 0
                total_matches = len(matches)

                if matches and kp_ref_cv and kp_cur_cv:
                    # Get matched keypoint coordinates
                    pts1 = np.array([kp_ref_cv[m[0]].pt for m in matches])
                    pts2 = np.array([kp_cur_cv[m[1]].pt for m in matches])

                    # STANDARD HPATCHES: Transform pts1 (ref) to current image space
                    pts1_homogeneous = np.column_stack([pts1, np.ones(len(pts1))])
                    try:
                        pts1_transformed = cv2.perspectiveTransform(
                            pts1.reshape(-1, 1, 2).astype(np.float32),
                            homography.astype(np.float32),
                        ).reshape(-1, 2)
                    except cv2.error:
                        # Fallback to manual transformation
                        pts1_transformed = (homography @ pts1_homogeneous.T).T
                        pts1_transformed = (
                            pts1_transformed[:, :2] / pts1_transformed[:, 2:3]
                        )

                    # Count correct matches (within threshold)
                    distances = np.sqrt(np.sum((pts2 - pts1_transformed) ** 2, axis=1))
                    correct_matches = np.sum(distances < threshold)

                    # MMA: Percentage of correct matches within threshold
                    matching_accuracy = (
                        correct_matches / len(matches) if len(matches) > 0 else 0.0
                    )

                    # MS: Correct matches divided by average keypoints in overlapping area
                    # We need to compute keypoints in overlapping area
                    h, w = img_shape

                    # Define image corners and transform to find overlapping area
                    corners_ref = np.array(
                        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                    )
                    corners_cur = np.array(
                        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                    )

                    try:
                        # Transform reference corners to current image space
                        corners_ref_transformed = cv2.perspectiveTransform(
                            corners_ref.reshape(-1, 1, 2).astype(np.float32),
                            homography.astype(np.float32),
                        ).reshape(-1, 2)

                        # Create masks for overlapping area
                        # For reference image: which keypoints are in the area that overlaps with current
                        ref_kpts = np.array([kp.pt for kp in kp_ref_cv])
                        cur_kpts = np.array([kp.pt for kp in kp_cur_cv])

                        # Simple approximation: keypoints that are within image bounds after transformation
                        # Count keypoints in reference that map within current image bounds
                        ref_kpts_transformed = cv2.perspectiveTransform(
                            ref_kpts.reshape(-1, 1, 2).astype(np.float32),
                            homography.astype(np.float32),
                        ).reshape(-1, 2)

                        ref_in_overlap = np.sum(
                            (ref_kpts_transformed[:, 0] >= 0)
                            & (ref_kpts_transformed[:, 0] < w)
                            & (ref_kpts_transformed[:, 1] >= 0)
                            & (ref_kpts_transformed[:, 1] < h)
                        )

                        # Count keypoints in current that map within reference bounds when back-projected
                        homography_inv = np.linalg.inv(homography)
                        cur_kpts_transformed = cv2.perspectiveTransform(
                            cur_kpts.reshape(-1, 1, 2).astype(np.float32),
                            homography_inv.astype(np.float32),
                        ).reshape(-1, 2)

                        cur_in_overlap = np.sum(
                            (cur_kpts_transformed[:, 0] >= 0)
                            & (cur_kpts_transformed[:, 0] < w)
                            & (cur_kpts_transformed[:, 1] >= 0)
                            & (cur_kpts_transformed[:, 1] < h)
                        )

                        # Average keypoints in overlapping area
                        avg_keypoints_in_overlap = (
                            ref_in_overlap + cur_in_overlap
                        ) / 2.0

                        # MS: Correct matches divided by average keypoints in overlapping area
                        matching_score = (
                            correct_matches / avg_keypoints_in_overlap
                            if avg_keypoints_in_overlap > 0
                            else 0.0
                        )

                    except (cv2.error, np.linalg.LinAlgError):
                        # Fallback: use simple approximation
                        avg_keypoints = (len(kp_ref_cv) + len(kp_cur_cv)) / 2.0
                        matching_score = (
                            correct_matches / avg_keypoints
                            if avg_keypoints > 0
                            else 0.0
                        )

                # Compute homography accuracy using pre-computed matches
                homography_accuracy = 0.0
                num_inliers = 0

                if len(matches) >= 4 and kp_ref_cv and kp_cur_cv:
                    try:
                        # Get matched points
                        pts1 = np.float32(
                            [kp_ref_cv[m[0]].pt for m in matches]
                        ).reshape(-1, 1, 2)
                        pts2 = np.float32(
                            [kp_cur_cv[m[1]].pt for m in matches]
                        ).reshape(-1, 1, 2)

                        # Estimate homography using RANSAC
                        estimated_H, mask = cv2.findHomography(
                            pts1,
                            pts2,
                            cv2.RANSAC,
                            ransacReprojThreshold=threshold,
                            maxIters=2000,
                            confidence=0.99,
                        )

                        if estimated_H is not None:
                            num_inliers = np.sum(mask) if mask is not None else 0

                            # Define image corners (standard HPatches approach)
                            h, w = img_shape
                            corners = np.float32(
                                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                            ).reshape(-1, 1, 2)

                            # Transform corners using ground truth homography
                            gt_corners = cv2.perspectiveTransform(
                                corners, homography.astype(np.float32)
                            )

                            # Transform corners using estimated homography
                            est_corners = cv2.perspectiveTransform(
                                corners, estimated_H.astype(np.float32)
                            )

                            # Compute corner accuracy
                            corner_errors = np.sqrt(
                                np.sum((gt_corners - est_corners) ** 2, axis=2)
                            ).flatten()
                            correct_corners = np.sum(corner_errors < threshold)
                            homography_accuracy = correct_corners / 4.0

                    except (cv2.error, np.linalg.LinAlgError):
                        homography_accuracy = 0.0

                # Store threshold-specific results
                result[f"repeatability_{threshold}"] = float(repeatability)
                result[f"matching_score_{threshold}"] = float(matching_score)  # MS
                result[f"matching_accuracy_{threshold}"] = float(
                    matching_accuracy
                )  # MMA
                result[f"homography_accuracy_{threshold}"] = float(homography_accuracy)
                result[f"correct_matches_{threshold}"] = int(correct_matches)
                result[f"total_matches_{threshold}"] = int(total_matches)
                result[f"num_inliers_{threshold}"] = int(num_inliers)

            results.append(result)

        except Exception as e:
            print(f"Error processing {seq_name}_{img_idx}: {e}")
            # Return default results on error
            result = {
                "sequence_name": seq_name,
                "img_idx": img_idx,
                "keypoints": len(kp_cur) if kp_cur is not None else 0,
            }
            for threshold in thresholds:
                result[f"repeatability_{threshold}"] = 0.0
                result[f"matching_score_{threshold}"] = 0.0  # MS
                result[f"matching_accuracy_{threshold}"] = 0.0  # MMA
                result[f"homography_accuracy_{threshold}"] = 0.0
                result[f"correct_matches_{threshold}"] = 0
                result[f"total_matches_{threshold}"] = 0
                result[f"num_inliers_{threshold}"] = 0
            results.append(result)

    return results


class HPatchesBenchmark:
    """HPatches benchmark for keypoint detection and description evaluation."""

    def __init__(
        self,
        data_path: str = "benchmarks/hpatches/data/hpatches-sequences-release",
        max_kpts: int = 2048,
        thresholds: List[float] = [1, 2, 3, 4, 5],
        n_jobs: int = -1,
        mnn_min_score: float = 0.0,
        mnn_ratio_test: float = 1.0,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.sequences = []
        self.results = {}
        self.max_kpts = max_kpts
        self.thresholds = thresholds
        self.n_jobs = n_jobs if n_jobs != -1 else int(mp.cpu_count() * 0.8)
        self.mnn_min_score = mnn_min_score
        self.mnn_ratio_test = mnn_ratio_test
        self.seed = seed

        # Matcher params
        self.matcher_params = {"min_score": mnn_min_score, "ratio_test": mnn_ratio_test}

    def prepare_data(self):
        """Prepare HPatches sequences for benchmarking."""
        if not self.data_path.exists():
            logger.error(f"HPatches data path does not exist: {self.data_path}")
            return False

        # Load all sequences
        sequence_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]

        for seq_dir in sorted(sequence_dirs):
            sequence_name = seq_dir.name

            # Load reference image (image 1)
            ref_img_path = seq_dir / "1.ppm"
            if not ref_img_path.exists():
                continue

            sequence_data = {
                "name": sequence_name,
                "ref_image": str(ref_img_path),
                "images": [],
                "homographies": [],
            }

            # Load all other images in sequence
            for i in range(2, 7):  # HPatches typically has 6 images per sequence
                img_path = seq_dir / f"{i}.ppm"
                h_path = seq_dir / f"H_1_{i}"

                if img_path.exists() and h_path.exists():
                    # Load homography matrix
                    homography = np.loadtxt(h_path)

                    sequence_data["images"].append(str(img_path))
                    sequence_data["homographies"].append(homography)

            if sequence_data["images"]:
                self.sequences.append(sequence_data)

        logger.info(f"Loaded {len(self.sequences)} HPatches sequences")
        return True

    @torch.no_grad()
    def extract_features_with_wrapper(self, wrapper):
        """Extract features for all unique images using the wrapper."""
        print("\nExtracting features using wrapper...")

        features_dict = {}

        for sequence in tqdm(self.sequences, desc="Extracting features"):
            sequence_name = sequence["name"]

            # Extract features for reference image
            ref_img = io.imread(sequence["ref_image"])
            H, W, _ = ref_img.shape
            ref_img = wrapper.img_from_numpy(ref_img)

            if ref_img is None:
                continue

            with torch.no_grad():
                out_ref = wrapper.extract(ref_img, max_kpts=self.max_kpts)

            ref_key = f"{sequence_name}_ref"
            features_dict[ref_key] = {
                "kpts": (
                    out_ref.kpts.cpu().numpy() if out_ref.kpts is not None else None
                ),
                "desc": out_ref.des.cpu() if out_ref.des is not None else None,
                "img_shape": ref_img.shape[:2],
            }

            # Extract features for all images in sequence
            for img_idx, (img_path, homography) in enumerate(
                zip(sequence["images"], sequence["homographies"])
            ):
                img = io.imread(img_path)
                img = wrapper.img_from_numpy(img)

                if img is None:
                    continue

                with torch.no_grad():
                    out_cur = wrapper.extract(img, max_kpts=self.max_kpts)

                cur_key = f"{sequence_name}_{img_idx + 2}"
                features_dict[cur_key] = {
                    "kpts": (
                        out_cur.kpts.cpu().numpy() if out_cur.kpts is not None else None
                    ),
                    "desc": out_cur.des.cpu() if out_cur.des is not None else None,
                    "img_shape": img.shape[:2],
                    "homography": homography,
                }

                gc.collect()
                torch.cuda.empty_cache()

        return features_dict

    def batch_match_all_pairs(self, features_dict, device="cuda"):
        """Match all pairs in batch mode."""
        print("\nPerforming matching...")

        # Create matcher
        matcher = MNN(**self.matcher_params)

        # Prepare match data
        matches_dict = {}

        for sequence in tqdm(self.sequences, desc="Matching pairs"):
            sequence_name = sequence["name"]
            ref_key = f"{sequence_name}_ref"

            if ref_key not in features_dict:
                continue

            ref_features = features_dict[ref_key]

            for img_idx in range(len(sequence["images"])):
                cur_key = f"{sequence_name}_{img_idx + 2}"

                if cur_key not in features_dict:
                    continue

                cur_features = features_dict[cur_key]

                # Get descriptors
                desc_ref = ref_features["desc"].to(device)
                desc_cur = cur_features["desc"].to(device)

                # Match
                matches = matcher.match([desc_ref], [desc_cur])[0].matches.cpu().numpy()

                # Store match data
                pair_key = (sequence_name, img_idx + 2)
                matches_dict[pair_key] = {
                    "matches": [(int(m[0]), int(m[1])) for m in matches],
                    "kp_ref": ref_features["kpts"],
                    "kp_cur": cur_features["kpts"],
                    "homography": cur_features["homography"],
                    "img_shape": cur_features["img_shape"],
                }

        return matches_dict

    def batch_compute_metrics(self, matches_dict):
        """Perform metrics computation in parallel batches."""
        print("\nStarting parallel metrics computation...")

        # Prepare data for parallel processing
        batch_data = []
        batch_size = len(matches_dict) // self.n_jobs

        current_batch = []
        for pair_key, data in matches_dict.items():
            current_batch.append(
                (
                    pair_key,
                    data["matches"],
                    data["kp_ref"],
                    data["kp_cur"],
                    data["homography"],
                    data["img_shape"],
                )
            )

            if len(current_batch) >= batch_size:
                batch_data.append(current_batch)
                current_batch = []

        if current_batch:  # Add remaining pairs
            batch_data.append(current_batch)

        # Process batches in parallel with proper seeding
        metrics_computation_partial = partial(
            process_metrics_batch, thresholds=self.thresholds
        )

        # Generate seeds for each worker
        base_seed = self.seed
        worker_seeds = [base_seed + i for i in range(len(batch_data))]

        # Use joblib with loky backend for better process isolation
        from joblib import parallel_backend

        with parallel_backend("loky", n_jobs=self.n_jobs):
            batch_results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(metrics_computation_partial)(batch, worker_seed=seed)
                for batch, seed in zip(batch_data, worker_seeds)
            )

        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    def benchmark(self, wrapper, device="cuda"):
        """Run HPatches benchmark on given wrapper."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fix_rng(seed=self.seed)

        if not self.sequences:
            self.prepare_data()

        # Phase 1: Extract all features
        features_dict = self.extract_features_with_wrapper(wrapper)

        # Phase 2: Batch matching
        matches_dict = self.batch_match_all_pairs(features_dict, device)

        # Phase 3: Parallel metrics computation
        all_results = self.batch_compute_metrics(matches_dict)

        # Step 4: Organize results by sequence type (i-, v-, overall)
        i_results = []  # Illumination sequences
        v_results = []  # Viewpoint sequences

        for result in all_results:
            seq_name = result["sequence_name"]
            if seq_name.startswith("i_"):
                i_results.append(result)
            elif seq_name.startswith("v_"):
                v_results.append(result)

        # Helper function to compute statistics for a set of results
        def compute_stats_for_results(results_list, category_name):
            if not results_list:
                return {
                    "category": category_name,
                    "num_sequences": 0,
                    "num_image_pairs": 0,
                    "mean_keypoints": 0.0,
                }

            stats = {
                "category": category_name,
                "num_sequences": len(set([r["sequence_name"] for r in results_list])),
                "num_image_pairs": len(results_list),
                "mean_keypoints": float(
                    np.mean([r["keypoints"] for r in results_list])
                ),
            }

            # Compute statistics for each threshold
            for threshold in self.thresholds:
                rep_scores = [r[f"repeatability_{threshold}"] for r in results_list]
                match_scores = [
                    r[f"matching_score_{threshold}"] for r in results_list
                ]  # MS
                match_accuracy = [
                    r[f"matching_accuracy_{threshold}"] for r in results_list
                ]  # MMA
                homo_scores = [
                    r[f"homography_accuracy_{threshold}"] for r in results_list
                ]
                total_matches = [r[f"total_matches_{threshold}"] for r in results_list]
                correct_matches = [
                    r[f"correct_matches_{threshold}"] for r in results_list
                ]

                stats[f"repeatability_{threshold}"] = {
                    "mean": float(np.mean(rep_scores)),
                    "std": float(np.std(rep_scores)),
                    "median": float(np.median(rep_scores)),
                }

                # MS: Matching Score = correct matches / avg keypoints in overlap
                stats[f"matching_score_{threshold}"] = {
                    "mean": float(np.mean(match_scores)),
                    "std": float(np.std(match_scores)),
                    "median": float(np.median(match_scores)),
                }

                # MMA: Mean Matching Accuracy = percentage of correct matches
                stats[f"matching_accuracy_{threshold}"] = {
                    "mean": float(np.mean(match_accuracy)),
                    "std": float(np.std(match_accuracy)),
                    "median": float(np.median(match_accuracy)),
                }

                stats[f"homography_accuracy_{threshold}"] = {
                    "mean": float(np.mean(homo_scores)),
                    "std": float(np.std(homo_scores)),
                    "median": float(np.median(homo_scores)),
                }

                stats[f"match_stats_{threshold}"] = {
                    "total_matches_mean": float(np.mean(total_matches)),
                    "correct_matches_mean": float(np.mean(correct_matches)),
                    "pairs_with_matches": sum(1 for tm in total_matches if tm > 0),
                }

            return stats

        # Compute statistics for each category
        illumination_stats = compute_stats_for_results(i_results, "illumination")
        viewpoint_stats = compute_stats_for_results(v_results, "viewpoint")
        overall_stats = compute_stats_for_results(all_results, "overall")

        # Create final results structure
        results = {
            "method": wrapper.name,
            "max_keypoints": self.max_kpts,
            "thresholds": self.thresholds,
            "matcher_config": {
                "type": "mnn",
                "min_score": self.mnn_min_score,
                "ratio_test": self.mnn_ratio_test,
            },
            "timestamp": timestamp,
            "seed": self.seed,
            "overall": overall_stats,
            "illumination": illumination_stats,
            "viewpoint": viewpoint_stats,
        }

        return results, timestamp


if __name__ == "__main__":
    import json
    import argparse
    from wrappers_manager import wrappers_manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wrapper-name", type=str, default="disk")
    parser.add_argument("--max-kpts", type=int, default=2048)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--ratio-test", type=float, default=1.0)

    parser.add_argument("--custom-desc", type=str, default=None)
    parser.add_argument("--stats", type=bool, default=True)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1, 2, 3, 4, 5],  # reducing to 1 2 3
        help="List of thresholds for repeatability and matching score computation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    device = args.device
    wrapper_name = args.wrapper_name
    max_kpts = args.max_kpts
    custom_desc = args.custom_desc
    thresholds = args.thresholds
    n_jobs = args.n_jobs
    seed = args.seed

    # Fixed MNN configuration - using same defaults as MD1500
    mnn_min_score = args.min_score
    mnn_ratio_test = args.ratio_test

    # Define the wrapper
    wrapper = wrappers_manager(name=wrapper_name, device=device)

    if custom_desc is not None:
        weights = torch.load(custom_desc, weights_only=False)
        config = weights["config"]
        model_config = {
            "ch_in": config["model_config"]["unet_ch_in"],
            "kernel_size": config["model_config"]["unet_kernel_size"],
            "activ": config["model_config"]["unet_activ"],
            "norm": config["model_config"]["unet_norm"],
            "skip_connection": config["model_config"]["unet_with_skip_connections"],
            "spatial_attention": config["model_config"]["unet_spatial_attention"],
            "third_block": config["model_config"]["third_block"],
        }

        from sandesc_models.sandesc.network_descriptor import SANDesc

        network = SANDesc(**model_config).eval().to(device)
        weights = torch.load(custom_desc, weights_only=False)
        network.load_state_dict(weights["state_dict"])
        wrapper.add_custom_descriptor(network)
        wrapper.name = f"{wrapper.name}+SANDesc"
        print(f"Using custom descriptors from {custom_desc}.\n")

    # benchmark params - cleaner key format
    key = f"{wrapper.name}_{max_kpts}kpts"

    # add tag to the key
    if args.run_tag is not None:
        key = f"{key}_{args.run_tag}"

    print(f"\n>>> Running HPatches benchmark for {key}...<<<\n")
    print(f"Using thresholds: {thresholds}")
    print(f"Using {n_jobs} jobs for parallel computation")
    print(
        f"Using MNN matcher with min_score: {mnn_min_score}, ratio_test: {mnn_ratio_test}"
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # create if not exists
    results_path = Path("benchmarks/hpatches/results")
    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(results_path / "results.json"):
        with open(results_path / "results.json", "w") as f:
            json.dump({}, f)

    with open(results_path / "results.json", "r") as f:
        data = json.load(f)

    # Define the benchmark
    benchmark = HPatchesBenchmark(
        max_kpts=max_kpts,
        thresholds=thresholds,
        n_jobs=n_jobs,
        mnn_min_score=mnn_min_score,
        mnn_ratio_test=mnn_ratio_test,
        seed=seed,
    )

    # Run the benchmark
    s = time.time()
    results, timestamp = benchmark.benchmark(wrapper, device=device)
    total_time = time.time() - s

    # Add timing info to results
    results["benchmark_time_seconds"] = float(total_time)

    print(f"Total time: {total_time:.1f} seconds")

    # Print summary
    print(f"\n{'='*80}")
    print(f"HPatches Benchmark Results - {wrapper.name}")
    print(f"{'='*80}")
    print(f"Method: {results['method']}")
    print(f"Max keypoints: {results['max_keypoints']}")
    print(f"Benchmark time: {total_time:.1f}s")
    print(f"{'='*80}")

    # Print results for each category
    for category_name, category_key in [
        ("OVERALL", "overall"),
        # ("ILLUMINATION", "illumination"),
        # ("VIEWPOINT", "viewpoint"),
    ]:
        stats = results[category_key]
        if stats["num_image_pairs"] > 0:
            print(f"{category_name} RESULTS:")
            print(f"  Sequences: {stats['num_sequences']}")
            print(f"  Image pairs: {stats['num_image_pairs']}")
            print(f"  Mean keypoints: {stats['mean_keypoints']:.1f}")
            print(f"  {'-'*60}")

            for threshold in thresholds:
                print(f"  Threshold {threshold}px:")
                print(
                    f"    Repeatability:      {stats[f'repeatability_{threshold}']['mean']:.4f} ± {stats[f'repeatability_{threshold}']['std']:.4f}"
                )
                print(
                    f"    Matching Score:     {stats[f'matching_score_{threshold}']['mean']:.4f} ± {stats[f'matching_score_{threshold}']['std']:.4f}"
                )
                print(
                    f"    Matching Accuracy:  {stats[f'matching_accuracy_{threshold}']['mean']:.4f} ± {stats[f'matching_accuracy_{threshold}']['std']:.4f}"
                )
                print(
                    f"    Homography Acc:     {stats[f'homography_accuracy_{threshold}']['mean']:.4f} ± {stats[f'homography_accuracy_{threshold}']['std']:.4f}"
                )
                print(
                    f"    Avg Total Matches:  {stats[f'match_stats_{threshold}']['total_matches_mean']:.1f}"
                )
            print()

    print(f"{'='*80}")

    # Convert numpy types before saving to JSON
    serializable_results = convert_numpy_types(results)

    # Save the results
    data[f"{key}_{timestamp}"] = serializable_results
    with open(results_path / "results.json", "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"Results saved to {results_path/'results.json'}")
