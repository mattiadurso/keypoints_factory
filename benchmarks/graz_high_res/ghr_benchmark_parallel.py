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
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp
from functools import partial
import argparse

from matchers.mnn import MNN
from benchmarks.benchmark_utils import (
    str2bool,
    fix_rng,
    fix_worker_rng,
    print_metrics,
    estimate_pose,
    compute_pose_error,
    compute_relative_pose,
    pose_auc,
)


def process_pose_estimation_batch(pair_matches_data, th, worker_seed=None):
    """Process pose estimation for a batch of pairs."""
    # Fix randomness for this worker
    if worker_seed is not None:
        fix_worker_rng(worker_seed)

    results = []

    for (
        (scene_name, pair_id, img1_name, img2_name),
        matches,
        kpts1,
        kpts2,
        K1,
        K2,
        R,
        t,
    ) in pair_matches_data:
        try:
            if len(matches) < 5:
                results.append(
                    (scene_name, pair_id, img1_name, img2_name, 180, 180, 180, 0)
                )
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

            results.append(
                (
                    scene_name,
                    pair_id,
                    img1_name,
                    img2_name,
                    e_t,
                    e_R,
                    e_pose,
                    num_inliers,
                )
            )

        except Exception as e:
            results.append(
                (scene_name, pair_id, img1_name, img2_name, 180, 180, 180, 0)
            )

    return results


class GrazHighResBatchBenchmark:
    def __init__(
        self,
        DATASET_PATH="benchmarks/graz_high_res/data/",
        th=0.75,
        min_score: float = -1.0,
        ratio_test: float = 1.0,
        max_kpts: int = 2048,
        n_jobs: int = -1,
        seed: int = 0,
        partial: bool = False,
        keypoints_path: str = None,
        descriptors_path: str = None,
    ):

        self.DATASET_PATH = Path(DATASET_PATH)
        self.th = th
        self.max_kpts = max_kpts
        self.n_jobs = n_jobs if n_jobs != -1 else int(mp.cpu_count() * 0.8)
        self.seed = seed
        self.partial = partial

        # Initialize scene data
        partial_scene_id = 0
        sample_rate = 10
        min_matches = 100
        max_matches = 1000

        pairs_path = "benchmarks/graz_high_res/data/pairs.npy"
        if Path(pairs_path).exists():
            self.scenes = np.load(pairs_path, allow_pickle=True).item()
        else:
            print(
                f"No pairs.npy found in {pairs_path}, generating pairs from viewgraph_30.txt"
            )
            self.scenes = {}
            for scene_name in os.listdir(self.DATASET_PATH):
                if not os.path.isdir(self.DATASET_PATH / scene_name):
                    continue
                with open(
                    f"{self.DATASET_PATH}/{scene_name}/colmap/viewgraph_30.txt", "r"
                ) as f:
                    lines = f.readlines()

                scene = np.array([line.split() for line in lines])
                matches = np.array(scene)[:, 2].astype(int)
                sampling = np.zeros_like(matches, dtype=bool)
                sampling[::sample_rate] = True
                mask = (matches > min_matches) & (matches < max_matches) & sampling
                scene = scene[mask, :2]
                self.scenes[scene_name] = scene

            self.scenes = dict(
                sorted(
                    self.scenes.items(), key=lambda item: len(item[1]), reverse=False
                )
            )
            np.save(pairs_path, self.scenes)
            print(f"Pairs saved to {pairs_path}")

        if self.partial:
            k = list(self.scenes.keys())[partial_scene_id]
            self.scenes = {k: self.scenes[k]}

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

        # Print stats
        pairs = [scene for scene in self.scenes.values()]
        total_pairs = np.vstack(pairs) if pairs else np.array([])
        total_images = np.unique(total_pairs.flatten()) if len(total_pairs) > 0 else []
        total_frames = len(glob.glob(f"{self.DATASET_PATH}/*/frames/*/*.jpg"))

        print(f'Scenes: {", ".join(list(self.scenes.keys()))}\n')
        print(f"Total pairs: {len(total_pairs):,}")
        print(
            f"Total images: {len(total_images):,}/{total_frames:,} "
            + f"({100*len(total_images)/total_frames:.2f}%)"
            if total_frames > 0
            else "0%"
        )

    def extract_features_with_wrapper(self, wrapper, factor=1):
        """Extract features using the wrapper."""
        print("\nExtracting features using wrapper...")

        keypoints_dict = {}
        descriptors_dict = {}

        # Get all unique images across all scenes
        unique_images = set()
        for scene_name, scene_pairs in self.scenes.items():
            for img1_name, img2_name in scene_pairs:
                unique_images.add((scene_name, img1_name))
                unique_images.add((scene_name, img2_name))

        # Extract features for each image
        for scene_name, img_name in tqdm(unique_images, desc="Extracting features"):
            images_path = self.DATASET_PATH / scene_name / "frames"
            im_path = images_path / img_name

            try:
                img = Image.open(im_path)

                if factor != 1:
                    W, H = img.size
                    img = img.resize((int(W / factor), int(H / factor)))

                img = wrapper.img_from_numpy(np.array(img))

                with torch.no_grad():
                    out = wrapper.extract(img, self.max_kpts)
                    keypoints_dict[f"{scene_name}_{img_name}"] = out.kpts.cpu()
                    descriptors_dict[f"{scene_name}_{img_name}"] = out.des.cpu()

            except Exception as e:
                print(f"Error processing {scene_name}/{img_name}: {e}")
                continue

        return keypoints_dict, descriptors_dict

    def save_features_to_intermediate(self, keypoints_dict, descriptors_dict, key):
        """Save extracted features to intermediate directory.
        key: f"{wrapper_name}_scale_{scale_factor}_kpts_{max_kpts}"
        """
        intermediate_path = Path("benchmarks/graz_high_res/intermediate") / key
        os.makedirs(intermediate_path, exist_ok=True)

        keypoints_file = intermediate_path / "keypoints.pt"
        descriptors_file = intermediate_path / "descriptors.pt"

        print(f"Saving features to {intermediate_path}")
        torch.save(keypoints_dict, keypoints_file)
        torch.save(descriptors_dict, descriptors_file)

        print(f"Features saved: {keypoints_file} and {descriptors_file}")
        return keypoints_file, descriptors_file

    def batch_match_all_pairs(self, wrapper, device="cuda", save_key=None, factor=1):
        """Match all pairs in batch mode."""
        print("\nStarting unique images feature extraction...")

        # Get features (either precomputed or extract with wrapper)
        if self.use_precomputed:
            keypoints_dict = self.keypoints_dict
            descriptors_dict = self.descriptors_dict
        else:
            keypoints_dict, descriptors_dict = self.extract_features_with_wrapper(
                wrapper, factor
            )

            # Save features to intermediate directory if key provided
            if save_key:
                self.save_features_to_intermediate(
                    keypoints_dict, descriptors_dict, save_key
                )

        # Prepare all pair data
        pair_data = []
        for scene_name, scene_pairs in self.scenes.items():
            # Load camera data for this scene
            Ks = np.load(
                self.DATASET_PATH / scene_name / "cameras/cameras.npz",
                allow_pickle=True,
            )
            poses = np.load(
                self.DATASET_PATH / scene_name / "cameras/poses.npz", allow_pickle=True
            )

            for pair_id, (img1_name, img2_name) in enumerate(scene_pairs):
                try:
                    # Get camera parameters
                    camera1 = poses[str(img1_name)].item()
                    T1 = camera1["P"].copy()
                    K1 = Ks[str(camera1["camera_id"])].item()["K"].copy()
                    R1, t1 = T1[:3, :3], T1[:3, 3]

                    camera2 = poses[str(img2_name)].item()
                    T2 = camera2["P"].copy()
                    K2 = Ks[str(camera2["camera_id"])].item()["K"].copy()
                    R2, t2 = T2[:3, :3], T2[:3, 3]

                    R, t = compute_relative_pose(R1, t1, R2, t2)

                    # Scale intrinsics if factor != 1
                    if factor != 1:
                        K1[:2, :3] /= factor
                        K2[:2, :3] /= factor

                    pair_data.append(
                        ((scene_name, pair_id, img1_name, img2_name), K1, K2, R, t)
                    )

                except Exception as e:
                    print(
                        f"Error loading camera data for {scene_name}/{img1_name}-{img2_name}: {e}"
                    )
                    continue

        # Create matcher
        matcher = MNN(**self.matcher_params)

        # Batch matching
        matches_dict = {}
        print("\nPerforming matching...")

        for (scene_name, pair_id, img1_name, img2_name), K1, K2, R, t in tqdm(
            pair_data, desc="Matching pairs"
        ):
            # Get features
            key1 = f"{scene_name}_{img1_name}"
            key2 = f"{scene_name}_{img2_name}"

            if key1 not in keypoints_dict or key2 not in keypoints_dict:
                continue

            kpts1 = keypoints_dict[key1]
            kpts2 = keypoints_dict[key2]
            desc1 = descriptors_dict[key1].to(device)
            desc2 = descriptors_dict[key2].to(device)

            # Match
            matches = matcher.match([desc1], [desc2])[0].matches.cpu()

            # Store for pose estimation
            matches_dict[(scene_name, pair_id, img1_name, img2_name)] = {
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
        for (scene_name, pair_id, img1_name, img2_name), data in matches_dict.items():
            current_batch.append(
                (
                    (scene_name, pair_id, img1_name, img2_name),
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
    def benchmark(
        self, wrapper, device="cuda", save_key=None, factor=1, save_stats=True
    ):
        """Run the complete benchmark."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Enhanced randomness fixing for parallel processing
        fix_rng(seed=self.seed)

        # Create save key for features with timestamp (similar to MD1500 pattern)
        features_save_key = None
        if save_key:
            features_save_key = f"{save_key}_{timestamp}"

        # Force loky backend for better isolation and reproducibility
        with parallel_backend("loky", n_jobs=self.n_jobs):
            # Phase 1: Batch matching (with feature extraction if needed)
            matches_dict = self.batch_match_all_pairs(
                wrapper, device, save_key=features_save_key, factor=factor
            )

            # Phase 2: Parallel pose estimation
            results = self.batch_pose_estimation(matches_dict)

        # Process results for statistics
        stats_df = {}
        tot_e_pose = []
        inliers = []

        for (
            scene_name,
            pair_id,
            img1_name,
            img2_name,
            e_t,
            e_R,
            e_pose,
            num_inliers,
        ) in results:
            tot_e_pose.append(e_pose)
            inliers.append(num_inliers)

            if save_stats:
                stats_df[f"{scene_name}_{pair_id}"] = {
                    "img1": img1_name,
                    "img2": img2_name,
                    "scene_name": scene_name,
                    "e_t": e_t,
                    "e_R": e_R,
                    "e_pose": e_pose,
                    "num_inliers": num_inliers,
                }

        # Save stats to CSV
        if save_stats:
            stats_df = pd.DataFrame.from_dict(stats_df, orient="index")
            path = Path("benchmarks/graz_high_res/stats")
            os.makedirs(path, exist_ok=True)
            stats_df.to_csv(
                path
                / f'{wrapper.name}_stats_scale_{factor}_{"partial" if self.partial else "full"}_{timestamp}.csv',
                index=False,
            )

        # Compute metrics
        tot_e_pose = np.array(tot_e_pose)
        thresholds = [5, 10, 20]
        auc = pose_auc(tot_e_pose, thresholds)
        acc_5 = (tot_e_pose < 5).mean()
        acc_10 = (tot_e_pose < 10).mean()
        acc_20 = (tot_e_pose < 20).mean()

        return {
            "inlier": np.mean(inliers),
            "auc_5": auc[0],
            "auc_10": auc[1],
            "auc_20": auc[2],
            "map_5": acc_5,
            "map_10": np.mean([acc_5, acc_10]),
            "map_20": np.mean([acc_5, acc_10, acc_20]),
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
    parser.add_argument(
        "--wrapper-name", type=str, default="aliked", help="Wrapper name"
    )
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of parallel jobs")
    parser.add_argument(
        "--ratio-test", type=float, default=1.0, help="Ratio test threshold"
    )
    parser.add_argument(
        "--min-score", type=float, default=-1.0, help="Minimum match score"
    )
    parser.add_argument(
        "--th", type=float, default=0.75, help="Pose estimation threshold"
    )
    parser.add_argument("--max-kpts", type=int, default=2048, help="Maximum keypoints")
    parser.add_argument("--run-tag", type=str, default=None, help="Tag for this run")
    parser.add_argument(
        "--custom-desc", type=str, default=None, help="Path to custom descriptors"
    )
    parser.add_argument("--stats", type=str2bool, default=False, help="Save statistics")
    parser.add_argument(
        "--partial", type=str2bool, default=False, help="Run partial benchmark"
    )
    parser.add_argument(
        "--scale-factor", type=float, default=1, help="Scale factor for resizing images"
    )
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
    is_partial = args.partial  # Rename to avoid conflict with functools.partial
    scale_factor = args.scale_factor

    # Define the wrapper
    wrapper = wrappers_manager(name=wrapper_name, device=args.device)

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
        print(f"Using custom descriptors from {custom_desc}.")

    # matcher params
    key = f"{wrapper.name} ratio_test_{ratio_test}_min_score_{min_score}_th_{th} scale_{scale_factor} {max_kpts}"
    key += " partial" if is_partial else " full"

    # add tag to the key
    if args.run_tag is not None:
        key = f"{key} {args.run_tag}"

    print(f"\n>>> Running parallel benchmark for {key}...<<<\n")

    # create if not exists
    results_path = Path("benchmarks/graz_high_res/results")
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
    benchmark = GrazHighResBatchBenchmark(
        DATASET_PATH="benchmarks/graz_high_res/data/",
        th=args.th,
        min_score=args.min_score,
        ratio_test=args.ratio_test,
        max_kpts=args.max_kpts,
        n_jobs=args.n_jobs,
        seed=args.seed,
        partial=is_partial,  # Use the renamed variable
        keypoints_path=args.keypoints_path,
        descriptors_path=args.descriptors_path,
    )

    # Run the benchmark
    s = time.time()
    # Create a simpler save key for features (wrapper_name + scale + max_kpts)
    feature_save_key = f"{wrapper_name}_scale_{scale_factor}_kpts_{max_kpts}"  # should work, but not tested
    results, timestamp = benchmark.benchmark(
        wrapper,
        args.device,
        save_key=feature_save_key,
        factor=args.scale_factor,
        save_stats=args.stats,
    )
    print(f"Total time: {time.time()-s:.1f} seconds")
    print_metrics(wrapper, results)
    print("-------------------------------------------------------------")

    # Save the results
    data[f"{key} {timestamp}"] = results
    with open(results_path / "results.json", "w") as f:
        json.dump(data, f)

    print(f"Results saved to {results_path/'results.json'}\n\n")
