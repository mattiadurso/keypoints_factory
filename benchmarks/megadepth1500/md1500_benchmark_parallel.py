# This code is based on Parskatt implementation in DeDoDe.
# source: https://github.com/Parskatt/DeDoDe/blob/main/DeDoDe/benchmarks/mega_pose_est_mnn.py

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings

warnings.filterwarnings("ignore")

import time
import gc
import torch
import numpy as np
import pandas as pd
from torchinfo import summary
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial
import argparse
import h5py

from matchers.mnn import MNN
from benchmarks.benchmark_utils import (
    fix_rng,
    fix_worker_rng,
    parse_pair,
    print_metrics,
    estimate_pose,
    compute_pose_error,
    pose_auc,
    load_depth_h5,
    get_depth_at_keypoints,
)


def process_pose_estimation_batch(pair_matches_data, th, worker_seed=None):
    """Process pose estimation for a batch of pairs."""
    # Fix randomness for this worker
    if worker_seed is not None:
        fix_worker_rng(worker_seed)

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
            e_pose = max(e_t, e_R)
            e_pose = 180 if e_pose > 10 else e_pose
            num_inliers = np.sum(mask)

            results.append((img1, img2, e_t, e_R, e_pose, num_inliers))

        except Exception as e:
            results.append((img1, img2, 180, 180, 180, 0))

    return results


class MegaDepthBatchBenchmark:
    def __init__(
        self,
        DATASET_PATH="benchmarks/megadepth1500/data/",
        th=0.5,
        min_score: float = 0.0,
        ratio_test: float = 1.0,
        max_kpts: int = 2048,
        n_jobs: int = -1,
        seed: int = 0,
        keypoints_path: str = None,
        descriptors_path: str = None,
        compute_repeatability: bool = False,
    ):

        self.DATASET_PATH = Path(DATASET_PATH)
        self.images_path = self.DATASET_PATH / "images"
        self.depth_path = self.DATASET_PATH / "depths"
        self.th = th
        self.max_kpts = max_kpts
        self.n_jobs = (
            n_jobs if n_jobs != -1 else int(mp.cpu_count() * 0.8)
        )  # use 80% of CPUs by default
        self.seed = seed

        # Load pairs
        with open(self.DATASET_PATH / "pairs_calibrated.txt", "r") as f:
            self.pairs_calibrated = f.read().splitlines()

        # Matcher params
        self.matcher_params = {"min_score": min_score, "ratio_test": ratio_test}

        # Load precomputed features if provided
        self.use_precomputed = (
            keypoints_path is not None and descriptors_path is not None
        )
        if self.use_precomputed:
            self.keypoints_dict = torch.load(
                keypoints_path, map_location="cpu", weights_only=False
            )
            self.descriptors_dict = torch.load(
                descriptors_path, map_location="cpu", weights_only=False
            )
            print(
                f"Using precomputed features from {keypoints_path} and {descriptors_path}"
            )
        else:
            self.keypoints_dict = None
            self.descriptors_dict = None
            print("Will extract features using wrapper")
        self.compute_repeatability = compute_repeatability

    def extract_features_with_wrapper(self, wrapper):
        """Extract features using the wrapper."""
        print("\nExtracting features using wrapper...")

        keypoints_dict = {}
        descriptors_dict = {}

        # Get all unique images
        unique_images = set()
        for pair in self.pairs_calibrated:
            img1, img2, _, _, _, _ = parse_pair(pair)
            unique_images.add(img1)
            unique_images.add(img2)

        # Extract features for each image
        for img_path in tqdm(unique_images, desc="Extracting features"):
            im_path = self.images_path / img_path
            img = Image.open(im_path)
            img = wrapper.img_from_numpy(np.array(img))

            with torch.no_grad():
                out = wrapper.extract(img, self.max_kpts)
                keypoints_dict[img_path] = out.kpts.cpu()
                descriptors_dict[img_path] = out.des.cpu()

            if self.compute_repeatability:
                # load depth
                depth_file = self.depth_path / f"{img_path}.h5"
                depth = torch.from_numpy(load_depth_h5(depth_file))
                depth_at_kpts = get_depth_at_keypoints(depth, keypoints_dict[img_path])
                keypoints_dict[img_path] = {
                    "kpts": keypoints_dict[img_path],
                    "depth": depth_at_kpts,  # neeeded for repeatability computation
                }

        return keypoints_dict, descriptors_dict

    def save_features_to_intermediate(self, keypoints_dict, descriptors_dict, key):
        """Save extracted features to intermediate directory.
        key: f"{wrapper_name}_kpts_{max_kpts}"
        """
        intermediate_path = Path("benchmarks/megadepth1500/intermediate") / key
        os.makedirs(intermediate_path, exist_ok=True)

        keypoints_file = intermediate_path / "keypoints.pt"
        descriptors_file = intermediate_path / "descriptors.pt"

        print(f"Saving features to {intermediate_path}")
        torch.save(keypoints_dict, keypoints_file)
        torch.save(descriptors_dict, descriptors_file)

        print(f"Features saved: {keypoints_file} and {descriptors_file}")
        return keypoints_file, descriptors_file

    def batch_match_all_pairs(self, wrapper, device="cuda", save_key=None):
        """Match all pairs in batch mode."""
        print("\nStarting unique images feature extraction...")

        # Get features (either precomputed or extract with wrapper)
        if self.use_precomputed:
            keypoints_dict = self.keypoints_dict
            descriptors_dict = self.descriptors_dict
        else:
            keypoints_dict, descriptors_dict = self.extract_features_with_wrapper(
                wrapper
            )

            # Save features to intermediate directory if key provided
            if save_key:
                self.save_features_to_intermediate(
                    keypoints_dict, descriptors_dict, save_key
                )

        # Prepare all pair data
        pair_data = []
        for pair in self.pairs_calibrated:
            img1, img2, K1, K2, R, t = parse_pair(pair)
            pair_data.append(((img1, img2), K1, K2, R, t))

        # Create matcher
        matcher = MNN(**self.matcher_params)

        # Batch matching
        matches_dict = {}
        print("\nPerforming matching...")

        for (img1, img2), K1, K2, R, t in tqdm(pair_data, desc="Matching pairs"):
            # Get features
            kpts1 = keypoints_dict[img1]
            kpts2 = keypoints_dict[img2]
            desc1 = descriptors_dict[img1].to(device)
            desc2 = descriptors_dict[img2].to(device)

            # Match
            matches = matcher.match([desc1], [desc2])[0].matches.cpu()

            # Store for pose estimation
            matches_dict[(img1, img2)] = {
                "matches": matches,
                "kpts1": kpts1.cpu().numpy(),
                "kpts2": kpts2.cpu().numpy(),
                "K1": K1,
                "K2": K2,
                "R": R,
                "t": t,
            }

        return matches_dict

    def batch_pose_estimation(self, matches_dict):
        """Perform pose estimation in parallel batches."""
        print("\nStarting parallel pose estimation...")

        # Prepare data for parallel processing
        batch_data = []
        batch_size = len(matches_dict) // self.n_jobs

        current_batch = []
        for (img1, img2), data in matches_dict.items():
            current_batch.append(
                (
                    (img1, img2),
                    data["matches"],
                    data["kpts1"],
                    data["kpts2"],
                    data["K1"],
                    data["K2"],
                    data["R"],
                    data["t"],
                )
            )

            if len(current_batch) >= batch_size:
                batch_data.append(current_batch)
                current_batch = []

        if current_batch:  # Add remaining pairs
            batch_data.append(current_batch)

        # Process batches in parallel with proper seeding
        pose_estimation_partial = partial(process_pose_estimation_batch, th=self.th)

        # Generate seeds for each worker to ensure reproducibility
        base_seed = self.seed
        worker_seeds = [base_seed + i for i in range(len(batch_data))]

        batch_results = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(pose_estimation_partial)(batch, worker_seed=seed)
            for batch, seed in zip(batch_data, worker_seeds)
        )

        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    @torch.no_grad()
    def benchmark(self, wrapper, device="cuda", save_key=None):
        """Run the complete benchmark."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Enhanced randomness fixing for parallel processing
        fix_rng(seed=self.seed)

        # Create save key for features with timestamp (similar to GHR pattern)
        features_save_key = None
        if save_key:
            features_save_key = f"{save_key}_{timestamp}"

        # Set joblib backend to use 'loky' for better process isolation
        from joblib import parallel_backend

        # Force loky backend for better isolation and reproducibility
        with parallel_backend("loky", n_jobs=self.n_jobs):
            # Phase 1: Batch matching (with feature extraction if needed)
            matches_dict = self.batch_match_all_pairs(
                wrapper, device, save_key=features_save_key
            )

            # Phase 2: Parallel pose estimation
            results = self.batch_pose_estimation(matches_dict)

        # Compute metrics
        tot_e_pose = np.array([r[4] for r in results])
        inliers = np.array([r[5] for r in results])

        thresholds = [5, 10, 20]
        auc = pose_auc(tot_e_pose, thresholds)
        acc_5 = (tot_e_pose < 5).mean()
        acc_10 = (tot_e_pose < 10).mean()
        acc_15 = (tot_e_pose < 15).mean()
        acc_20 = (tot_e_pose < 20).mean()

        return {
            "inlier": np.mean(inliers),
            "auc_5": auc[0],
            "auc_10": auc[1],
            "auc_20": auc[2],
            "map_5": acc_5,
            "map_10": np.mean([acc_5, acc_10]),
            "map_20": np.mean([acc_5, acc_10, acc_15, acc_20]),
        }, timestamp


if __name__ == "__main__":
    import json
    import argparse
    from wrappers_manager import wrappers_manager

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keypoints-path",
        default=None,
        help="Path to precomputed keypoints (optional)",
    )
    parser.add_argument(
        "--descriptors-path",
        default=None,
        help="Path to precomputed descriptors (optional)",
    )
    parser.add_argument("--device", default="cuda", help="Device to use for matching")
    parser.add_argument("--wrapper-name", type=str, default="disk", help="Wrapper name")
    parser.add_argument(
        "--n-jobs", type=int, default=8, help="Number of parallel jobs"
    )  # might slightly affect reproducibility and results
    parser.add_argument(
        "--ratio-test", type=float, default=1.0, help="Ratio test threshold"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0, help="Minimum match score"
    )
    parser.add_argument(
        "--th", type=float, default=0.5, help="Pose estimation threshold"
    )
    parser.add_argument("--max-kpts", type=int, default=2048, help="Maximum keypoints")
    parser.add_argument("--run-tag", type=str, default=None, help="Tag for this run")
    parser.add_argument(
        "--custom-desc", type=str, default=None, help="Path to custom descriptors"
    )
    parser.add_argument(
        "--stats", type=bool, default=True, help="Save statistics"
    )  # not used, kept for compatibility. Now I save kpts stats anyway
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    device = args.device
    wrapper_name = args.wrapper_name
    ratio_test = args.ratio_test
    min_score = args.min_score
    max_kpts = args.max_kpts
    th = args.th
    custom_desc = args.custom_desc
    n_jobs = args.n_jobs
    seed = args.seed

    # Define the wrapper
    wrapper = wrappers_manager(name=wrapper_name, device=args.device)

    if custom_desc is not None:
        #  Eventually add my descriptors
        weights = torch.load(custom_desc, weights_only=False)
        config = weights["config"]["model"]  # model_config for sandesc, model for dino
        model = {
            "ch_in": config["unet_ch_in"],
            "kernel_size": config["unet_kernel_size"],
            "activ": config["unet_activ"],
            "norm": config["unet_norm"],
            "skip_connection": config["unet_with_skip_connections"],
            "spatial_attention": config["unet_spatial_attention"],
            "third_block": config["third_block"],
        }

        from sandesc_models.sandesc.network_descriptor import SANDesc, SANDescD

        # network = SANDesc(**model).eval().to(device)
        network = SANDescD(**model).eval().to(device)
        summary(network)

        weights = torch.load(custom_desc, weights_only=False)
        network.load_state_dict(weights["state_dict"])

        wrapper.add_custom_descriptor(network)
        wrapper.name = f"{wrapper.name}+SANDesc"
        print(f"Using custom descriptors from {custom_desc}.")

    # matcher params
    key = f"{wrapper.name} ratio_test_{ratio_test}_min_score_{min_score}_th_{th}_mnn {max_kpts}"

    # add tag to the key
    if args.run_tag is not None:
        key = f"{key} {args.run_tag}"

    print(f"\n>>> Running parallel benchmark for {key}...<<<\n")

    # create if not exists
    results_path = Path("benchmarks/megadepth1500/results")
    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(results_path / "results.json"):
        with open(results_path / "results.json", "w") as f:
            json.dump({}, f)

    with open(results_path / "results.json", "r") as f:
        data = json.load(f)

    if key in data:
        results = data[key]
        import warnings

        warnings.warn("A similar run already exists.", UserWarning)

    # Define the benchmark
    benchmark = MegaDepthBatchBenchmark(
        DATASET_PATH="benchmarks/megadepth1500/data/",
        th=args.th,
        min_score=args.min_score,
        ratio_test=args.ratio_test,
        max_kpts=args.max_kpts,
        n_jobs=args.n_jobs,
        seed=args.seed,
        keypoints_path=args.keypoints_path,
        descriptors_path=args.descriptors_path,
    )

    # Run the benchmark
    s = time.time()
    # Create a simpler save key for features (wrapper_name + max_kpts)
    feature_save_key = f"{wrapper_name}_kpts_{max_kpts}"
    results, timestamp = benchmark.benchmark(
        wrapper, args.device, save_key=feature_save_key
    )
    print(f"Total time: {time.time()-s:.1f} seconds")
    print_metrics(wrapper, results)
    print("-------------------------------------------------------------")

    # Save the results
    data[f"{key} {timestamp}"] = results
    with open(results_path / "results.json", "w") as f:
        json.dump(data, f)

    print(f"Results saved to {results_path/'results.json'}\n\n")
