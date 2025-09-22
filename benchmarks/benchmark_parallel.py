# This code is based on Parskatt implementation in DKM and DeDoDe.

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

abs_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(abs_root))

import warnings

warnings.filterwarnings("ignore")

import gc
import time
import torch
import logging
import argparse
import numpy as np

from tqdm.auto import tqdm
from datetime import datetime
from functools import partial
from joblib import Parallel, delayed, parallel_backend

from matchers.mnn import MNN
from benchmarks.benchmark_utils import (
    fix_rng,
    parse_pair,
    print_metrics,
    pose_auc,
    process_pose_estimation_batch,
    parse_poses,
    load_depth,
)
from benchmarks.repeatability_utils import compute_repeatabilities_from_kpts


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Benchmark:
    def __init__(
        self,
        benchmark_name: str,
        dataset_path: str,
        ransac_th: float = 1,
        min_score: float = 0.5,
        ratio_test: float = 1,
        max_kpts: int = 2048,
        njobs: int = 8,
        seed: int = 0,
        scaling_factor: float = 1.0,
        ghr_partial: bool = False,
        keypoints_path: str = None,
        descriptors_path: str = None,
        compute_repeatability: bool = False,
        px_thrs: list = [1, 3, 5],
    ):
        self.benchmark_name = benchmark_name
        self.dataset_path = dataset_path
        self.ransac_th = ransac_th
        self.min_score = min_score
        self.ratio_test = ratio_test
        self.max_kpts = max_kpts
        self.njobs = njobs
        self.seed = seed
        self.scaling_factor = scaling_factor  # mostly for GHR
        self.ghr_partial = ghr_partial
        self.keypoints_path = keypoints_path
        self.descriptors_path = descriptors_path
        if scaling_factor != 1 and compute_repeatability:
            logger.warning(
                "Repeatability computation might be incorrect when \
                scaling_factor != 1. Thus, it is disabled."
            )
        self.compute_repeatability = (
            compute_repeatability
            and benchmark_name
            in [
                "megadepth1500",
                "graz_high_res",
            ]
            and scaling_factor == 1
        )  # repeatability only for MegaDepth and GHR
        self.px_thrs = px_thrs

        s = " with repeatability computation" if self.compute_repeatability else ""
        logger.info(f"Benchmarking {self.benchmark_name}{s}.")

        # Load pairs and paths
        self.dataset_path = abs_root / self.dataset_path
        with open(self.dataset_path / "pairs_calibrated.txt", "r") as f:
            self.pairs_calibrated = f.read().splitlines()  # limit for quick testing

        if self.ghr_partial:
            scene = "graz_main_square"  # small scene for quick testing
            self.pairs_calibrated = [p for p in self.pairs_calibrated if scene in p]
            logger.info(f"Using only pairs from {scene} for GHR partial benchmark.")

        if dataset_name.lower() == "megadepth1500":
            self.images_path = self.dataset_path / "images"
            self.depths_path = self.dataset_path / "depths"
            self.views_path = self.dataset_path / "views.txt"
            self.views_dict = parse_poses(self.views_path, self.benchmark_name)

        elif dataset_name.lower() == "scannet1500":
            self.images_path = self.dataset_path

        elif dataset_name.lower() == "graz_high_res":
            self.images_path = self.dataset_path
            self.depths_path = self.dataset_path
            self.views_path = self.dataset_path / "views.txt"
            self.views_dict = parse_poses(self.views_path, self.benchmark_name)

        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

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
            logger.info(
                f"Using precomputed features from {keypoints_path} and {descriptors_path}"
            )
        else:
            self.keypoints_dict = None
            self.descriptors_dict = None
            logger.info("Extracting features using wrapper")

    def extract_features_with_wrapper(self, wrapper):
        """Extract features using the wrapper."""
        keypoints_dict = {}
        descriptors_dict = {}

        # Get all unique images
        unique_images = set()
        for pair in self.pairs_calibrated:
            img1, img2, _, _, _, _ = parse_pair(
                pair, benchmark_name=self.benchmark_name
            )
            unique_images.add(img1)
            unique_images.add(img2)

        # Extract features for each image
        s = " and depths" if self.compute_repeatability else ""
        for img_name in tqdm(unique_images, desc=f"Extracting features{s}"):
            img_path = self.images_path / img_name

            try:
                # img = Image.open(img_path)

                # if self.scaling_factor != 1:
                #     W, H = img.size
                #     img = img.resize(
                #         (int(W // self.scaling_factor), int(H // self.scaling_factor))
                #     )

                # img = wrapper.img_from_numpy(np.array(img))
                img = wrapper.load_image(img_path, scaling=self.scaling_factor)

                with torch.no_grad():
                    out = wrapper.extract(img, self.max_kpts)

                keypoints_dict[img_name] = {"kpts": out.kpts.cpu()}
                descriptors_dict[img_name] = out.des.cpu()

                if self.compute_repeatability:
                    # load depth
                    if self.benchmark_name == "megadepth1500":
                        Z_path = self.depths_path / f"{img_name.split('.')[0]}.h5"
                    elif self.benchmark_name == "graz_high_res":
                        scene, _, cam, image_name = img_name.split("/")
                        Z_path = (
                            self.depths_path
                            / scene
                            / "depth"
                            / cam
                            / f"{image_name.split('.')[0]}.h5"
                        )

                    Z = load_depth(
                        Z_path,
                        scale_factor=self.scaling_factor,
                        target=out.kpts,
                    )

                    Z_sampled, _ = wrapper.grid_sample_nan(
                        out.kpts[None], Z[None], mode="nearest"
                    )
                    keypoints_dict[img_name]["depth"] = Z_sampled[0]

            except Exception as e:
                logger.info(f"Error processing {img_name}: {e}")
                continue

        # free memory, this stuff is no longer needed
        if self.compute_repeatability:
            del Z, Z_sampled
        del wrapper, img, out
        gc.collect()
        torch.cuda.empty_cache()

        return keypoints_dict, descriptors_dict

    def save_features_to_intermediate(self, keypoints_dict, descriptors_dict, key):
        """Save extracted features to intermediate directory.
        key: f"{wrapper_name}_kpts_{max_kpts}"
        """
        intermediate_path = Path(f"benchmarks/{self.benchmark_name}/intermediate") / key
        os.makedirs(intermediate_path, exist_ok=True)

        keypoints_file = intermediate_path / "keypoints.pt"
        descriptors_file = intermediate_path / "descriptors.pt"

        torch.save(keypoints_dict, keypoints_file)
        torch.save(descriptors_dict, descriptors_file)

        logger.info(f"Features saved: {keypoints_file} and {descriptors_file}")
        return keypoints_file, descriptors_file

    def batch_match_all_pairs(self, wrapper, device="cuda", save_key=None):
        """Match all pairs in batch mode."""

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
            img1, img2, K1, K2, R, t = parse_pair(
                pair, benchmark_name=self.benchmark_name
            )
            pair_data.append(((img1, img2), K1, K2, R, t))

        # Create matcher
        matcher = MNN(**self.matcher_params)

        # Batch matching
        matches_dict = {}

        for (img1, img2), K1, K2, R, t in tqdm(pair_data, desc="Matching pairs"):
            # Get features
            kpts1 = keypoints_dict[img1]["kpts"]
            kpts2 = keypoints_dict[img2]["kpts"]
            desc1 = descriptors_dict[img1].to(device)
            desc2 = descriptors_dict[img2].to(device)

            # Scale intrinsics if scaling applied
            if self.scaling_factor != 1:
                K1[:2, :3] /= self.scaling_factor
                K2[:2, :3] /= self.scaling_factor

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

            # free memory, this stuff is no longer needed
            del desc1, desc2, matches
            torch.cuda.empty_cache()

        # Compute repeatability
        if self.compute_repeatability:
            rep_results = {
                **{f"rep_{int(pix)}": [] for pix in self.px_thrs},
                **{f"rep_mnn_{int(pix)}": [] for pix in self.px_thrs},
            }

            for pair in tqdm(
                pair_data, desc="Repeatability"
            ):  # runs in ~5s with 2048kpts, batching it might be even faster
                img1, img2 = pair[:1][0]

                kpts1 = keypoints_dict[img1]["kpts"]
                Z1 = keypoints_dict[img1]["depth"]
                K1 = self.views_dict[img1]["K"]
                P1 = self.views_dict[img1]["P"]
                img1_size = self.views_dict[img1]["image_size"]

                kpts2 = keypoints_dict[img2]["kpts"]
                Z2 = keypoints_dict[img2]["depth"]
                K2 = self.views_dict[img2]["K"]
                P2 = self.views_dict[img2]["P"]
                img2_size = self.views_dict[img2]["image_size"]

                rep = compute_repeatabilities_from_kpts(
                    kpts1[None].float().to(device),
                    kpts2[None].float().to(device),
                    K1[None].float().to(device),
                    K2[None].float().to(device),
                    Z1[None].float().to(device),
                    Z2[None].float().to(device),
                    P1[None].float().to(device),
                    P2[None].float().to(device),
                    img1_shape=img1_size,
                    img2_shape=img2_size,
                    px_thrs=self.px_thrs,
                )

                for b in rep:
                    for k in rep[b]:
                        rep_results[k].append(rep[b][k])
            # average over all pairs
            for k in rep_results:
                rep_results[k] = sum(rep_results[k]) / len(rep_results[k])
        else:
            rep_results = {}
        return matches_dict, rep_results

    def batch_pose_estimation(self, matches_dict):
        """Perform pose estimation in parallel batches."""
        logger.info("Starting parallel pose estimation...")

        # Prepare data for parallel processing
        batch_data = []
        batch_size = len(matches_dict) // self.njobs

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
        pose_estimation_partial = partial(
            process_pose_estimation_batch, th=self.ransac_th
        )

        # Generate seeds for each worker to ensure reproducibility
        base_seed = self.seed
        worker_seeds = [base_seed + i for i in range(len(batch_data))]

        batch_results = Parallel(n_jobs=self.njobs, verbose=1)(
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Enhanced randomness fixing for parallel processing
        fix_rng(seed=self.seed)

        # Create save key for features with timestamp (similar to GHR pattern)
        features_save_key = None
        if save_key:
            features_save_key = f"{save_key}_{timestamp}"

        # Force loky backend for better isolation and reproducibility
        with parallel_backend("loky", n_jobs=self.njobs):
            # Phase 1: Batch matching (with feature extraction if needed)
            matches_dict, rep_results = self.batch_match_all_pairs(
                wrapper, device, save_key=features_save_key
            )  # for some reason it uses GPU memory (it shouldn't), but freeing wrapper works

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

        out = {
            "inlier": np.mean(inliers),
            "auc_5": auc[0],
            "auc_10": auc[1],
            "auc_20": auc[2],
            "map_5": acc_5,
            "map_10": np.mean([acc_5, acc_10]),
            "map_20": np.mean([acc_5, acc_10, acc_15, acc_20]),
        }

        if self.compute_repeatability:
            out.update(rep_results)

        # Final rounding
        for k in out:
            out[k] = round(out[k], 1) if k == "inlier" else round(out[k] * 100, 1)

        return out, timestamp


if __name__ == "__main__":
    import json
    import argparse
    from wrappers_manager import wrappers_manager

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds", type=str, default="megadepth1500", help="Dataset (ds) name"
    )
    parser.add_argument("--device", default="cuda", help="Device to use for matching")
    parser.add_argument(
        "--wrapper-name", type=str, default="disk-kornia", help="Wrapper name"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset",
    )
    parser.add_argument(
        "--njobs", type=int, default=16, help="Number of parallel jobs"
    )  # might slightly affect reproducibility and results
    parser.add_argument(
        "--ratio-test", type=float, default=1.0, help="Ratio test threshold"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0, help="Minimum match score"
    )
    parser.add_argument(
        "--ransac-th", type=float, default=1.0, help="Pose estimation threshold"
    )
    parser.add_argument(
        "--max-kpts",
        type=int,
        choices=[2048, 8000],
        default=2048,
        help="Maximum keypoints (allowed values: 2048 or 8000)",
    )
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
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=1.0,
        help="Down-scaling factor for input images (e.g., 2 for half size)",
    )
    parser.add_argument(
        "--ghr-partial",
        action="store_true",
        help="Compute partial GHR benchmark (only graz_main_square scene)",
    )
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
    parser.add_argument(
        "--skip-repeatability", action="store_false", help="Don't compute repeatability"
    )
    args = parser.parse_args()

    device = args.device
    wrapper_name = args.wrapper_name
    dataset_name = args.ds

    if dataset_name.lower() in ["md", "md1500", "megadepth1500"]:
        dataset_name = "megadepth1500"
    elif dataset_name.lower() in ["sc", "scannet", "sc1500", "scannet1500"]:
        dataset_name = "scannet1500"
    elif dataset_name.lower() in ["graz", "ghr", "graz_high_res"]:
        dataset_name = "graz_high_res"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    data_path = (
        args.data_path
        if args.data_path is not None
        else f"benchmarks/{dataset_name}/data"
    )
    njobs = args.njobs
    ratio_test = args.ratio_test
    min_score = args.min_score
    ransac_th = args.ransac_th
    max_kpts = args.max_kpts
    run_tag = args.run_tag
    custom_desc = args.custom_desc
    stats = args.stats
    seed = args.seed
    scaling_factor = args.scaling_factor
    ghr_partial = args.ghr_partial
    keypoints_path = args.keypoints_path
    descriptors_path = args.descriptors_path
    compute_repeatability = args.skip_repeatability

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

        from sandesc_models.sandesc.network_descriptor import SANDesc

        network = SANDesc(**model).eval().to(device)

        weights = torch.load(custom_desc, weights_only=False)
        network.load_state_dict(weights["state_dict"])

        wrapper.add_custom_descriptor(network)
        wrapper.name = f"{wrapper.name}+SANDesc"
        logger.info(f"Using custom descriptors from {custom_desc}.")

    # matcher params

    if dataset_name == "graz_high_res" and ghr_partial:
        key = f"{wrapper.name} min_score_{min_score}_ratio_test_{ratio_test}_th_{ransac_th}_mnn_scale_{scaling_factor}_partial {max_kpts}"
    else:
        key = f"{wrapper.name} min_score_{min_score}_ratio_test_{ratio_test}_th_{ransac_th}_mnn_scale_{scaling_factor} {max_kpts}"

    # add tag to the key
    if args.run_tag is not None:
        key = f"{key} {args.run_tag}"

    logger.info(f"\n>>> Running parallel benchmark for {key}...<<<\n")

    # create if not exists
    results_path = Path(f"benchmarks/{dataset_name}/results")
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
    benchmark = Benchmark(
        benchmark_name=dataset_name,
        dataset_path=data_path,
        ransac_th=ransac_th,
        min_score=min_score,
        ratio_test=ratio_test,
        max_kpts=max_kpts,
        njobs=njobs,
        seed=seed,
        scaling_factor=scaling_factor,
        ghr_partial=ghr_partial,
        keypoints_path=keypoints_path,
        descriptors_path=descriptors_path,
        compute_repeatability=compute_repeatability,
    )

    # Run the benchmark
    s = time.time()
    # Create a simpler save key for features (wrapper_name + max_kpts)
    feature_save_key = f"{wrapper_name}_kpts_{max_kpts}"
    results, timestamp = benchmark.benchmark(
        wrapper, args.device, save_key=feature_save_key
    )
    print_metrics(wrapper, results)
    print("-------------------------------------------------------------")

    # Save the results
    data[f"{key} {timestamp}"] = results
    with open(results_path / "results.json", "w") as f:
        json.dump(data, f)

    logger.info(
        f"Results computed in {time.time()-s:.1f}s and saved to {results_path/'results.json'}\n\n"
    )
