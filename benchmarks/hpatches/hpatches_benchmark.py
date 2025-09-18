import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings

warnings.filterwarnings("ignore")

import time
import json
import torch
import logging
import argparse
import pandas as pd
import multiprocessing as mp
from typing import List
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime


from matchers.mnn import MNN
from benchmarks.benchmark_utils import (
    str2bool,
    fix_rng,
    convert_numpy_types,
    display_hpatches_results,
)
from benchmarks.hpatches.hpatches_benchmark_utils import (
    load_hpatches_in_memory,
    compute_matching_stats,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        feature_path: str = None,
        save_csv: bool = False,
    ):
        self.data_path = Path(data_path)
        self.sequences = []
        self.results = {}
        self.max_kpts = max_kpts
        self.thresholds = thresholds
        self.n_jobs = (
            n_jobs if n_jobs != -1 else int(mp.cpu_count() * 0.9)
        )  # might not be a good idea to use all cores
        self.mnn_min_score = mnn_min_score
        self.mnn_ratio_test = mnn_ratio_test
        self.seed = seed
        self.feature_path = feature_path
        self.save_csv = save_csv

        # Matcher params
        self.matcher_params = {"min_score": mnn_min_score, "ratio_test": mnn_ratio_test}
        self.matcher = MNN(**self.matcher_params)

    @torch.no_grad()
    def extract_features_with_wrapper(self, hpatches, wrapper):
        """Extract features for all unique images using the wrapper."""
        logger.info("Extracting features using wrapper...")

        # extract features for all images
        keypoints = {folder_name: [] for folder_name in hpatches}
        descriptors = {folder_name: [] for folder_name in hpatches}
        for folder_name, data in tqdm(hpatches.items(), f"Extracting keypoints"):
            for img_np in data["imgs"]:
                img = wrapper.img_from_numpy(img_np)
                output = wrapper.extract(img, max_kpts=self.max_kpts)
                keypoints[folder_name].append(output.kpts.cpu())
                descriptors[folder_name].append(output.des.cpu())

        # save keypoints and descriptors in a dict
        features_dict = {"keypoints": keypoints, "descriptors": descriptors}
        os.makedirs("benchmarks/hpatches/features", exist_ok=True)
        torch.save(
            features_dict,
            f"benchmarks/hpatches/features/{wrapper.name}_{self.max_kpts}kpts.pth",
        )
        logger.info(
            f"Features saved to benchmarks/hpatches/features/{wrapper.name}_{self.max_kpts}kpts.pth"
        )
        return keypoints, descriptors

    def benchmark(self, wrapper, device="cuda"):
        """Run HPatches benchmark on given wrapper."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fix_rng(seed=self.seed)

        # first load all images
        hpatches = load_hpatches_in_memory(self.data_path)

        # Phase 1: Extract all features
        if self.feature_path is None:
            keypoints, descriptors = self.extract_features_with_wrapper(
                hpatches, wrapper
            )
        else:
            features_dict = torch.load(self.feature_path, weights_only=False)
            keypoints = features_dict["keypoints"]
            descriptors = features_dict["descriptors"]
            logger.info(f"Loaded features from {self.feature_path}.")

        # match - FIXED VERSION
        wrapper.matcher = self.matcher
        matches = {folder_name: {} for folder_name in descriptors}
        for folder_name, des_folder in tqdm(descriptors.items(), f"Matching keypoints"):
            for i in range(2, 7):
                # Remove the extra list wrapping
                matches[folder_name][f"1_{i}"] = wrapper.match(
                    [des_folder[0]], [des_folder[i - 1]]
                )[0].matches

        # compute metrics
        (
            stats_df,
            aggregated_df,
            stats_homography_df,
            aggregated_homography_accuracy_df,
        ) = compute_matching_stats(keypoints, matches, hpatches, n_jobs=self.n_jobs)

        # csv
        if self.save_csv:
            results_dir = Path("benchmarks/hpatches/results/csv")
            results_dir.mkdir(parents=True, exist_ok=True)

            stats_df.to_csv(
                results_dir / f"{wrapper.name}_{self.max_kpts}_stats.csv", index=False
            )
            aggregated_df.to_csv(
                results_dir / f"{wrapper.name}_{self.max_kpts}_aggregated.csv",
                index=False,
            )
            stats_homography_df.to_csv(
                results_dir / f"{wrapper.name}_{self.max_kpts}_stats_homography.csv",
                index=False,
            )
            aggregated_homography_accuracy_df.to_csv(
                results_dir
                / f"{wrapper.name}_{self.max_kpts}_aggregated_homography_accuracy.csv",
                index=False,
            )

        results = {
            "method": wrapper.name,
            "max_keypoints": self.max_kpts,
            "matcher_params": self.matcher_params,
            "timestamp": timestamp,
        }

        for category in [
            "overall",
            "i",
            "v",
        ]:  # illumination (i), viewpoint (v), overall
            category_data = aggregated_df[aggregated_df["type"] == category]
            homography_data = aggregated_homography_accuracy_df[
                aggregated_homography_accuracy_df["type"] == category
            ]

            if len(category_data) == 0:
                continue

            results[category] = {
                "num_sequences": (
                    len(stats_df[stats_df["type"] == category]["scene"].unique())
                    if category != "overall"
                    else len(stats_df["scene"].unique())
                ),
                "num_image_pairs": (
                    len(stats_df[stats_df["type"] == category])
                    if category != "overall"
                    else len(stats_df[stats_df["type"].isin(["i", "v"])])
                ),
                "mean_keypoints": (
                    float(category_data["mean_n_keypoints"].iloc[0])
                    if len(category_data) > 0
                    else 0.0
                ),
            }

            # Add metrics for each threshold
            for (
                thr
            ) in self.thresholds:  # Use self.thresholds instead of hardcoded [1, 2, 3]
                thr_data = category_data[category_data["thr"] == thr]
                if len(thr_data) == 0:
                    continue

                row = thr_data.iloc[0]

                # Repeatability
                # Check if column exists properly
                results[category][f"repeatability_{thr}"] = {
                    "mean": (
                        float(row["mean_repeatability"])
                        if not pd.isna(row["mean_repeatability"])
                        else float("nan")
                    ),
                    "median": (
                        float(row["median_repeatability"])
                        if "median_repeatability" in row
                        and not pd.isna(row["median_repeatability"])
                        else float("nan")
                    ),
                }

                # Matching Score
                results[category][f"matching_score_{thr}"] = {
                    "mean": float(row["mean_matching_score"]),
                    "median": (
                        float(row["median_matching_score"])
                        if "median_matching_score" in row
                        else float(row["mean_matching_score"])
                    ),
                }

                # Matching Accuracy
                results[category][f"matching_accuracy_{thr}"] = {
                    "mean": float(row["mean_matching_accuracy"]),
                    "median": (
                        float(row["median_matching_accuracy"])
                        if "median_matching_accuracy" in row
                        else float(row["mean_matching_accuracy"])
                    ),
                }

                # Homography Accuracy - get from homography_data
                hom_data = homography_data[
                    (homography_data["accuracy_thr"] == float(thr))
                    & (
                        homography_data["ransac_thr"] == 3.0
                    )  # use 3.0 as default ransac threshold
                ]
                if len(hom_data) > 0:
                    results[category][f"homography_accuracy_{thr}"] = {
                        "mean": float(hom_data["homography_accuracy"].iloc[0]),
                        "median": float(hom_data["homography_accuracy"].iloc[0]),
                    }
                else:
                    results[category][f"homography_accuracy_{thr}"] = {
                        "mean": 0.0,
                        "median": 0.0,
                    }

                # Match statistics
                results[category][f"match_stats_{thr}"] = {
                    "total_matches_mean": float(row["mean_n_matches_proposed"]),
                    "inliers_mean": float(row["mean_n_inliers_nn_GT"]),
                    "precision_mean": float(row["mean_precision"]),
                    "recall_mean": float(row["mean_recall"]),
                }

        return results, timestamp


if __name__ == "__main__":
    from wrappers_manager import wrappers_manager

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wrapper-name", type=str, default="superpoint")
    parser.add_argument("--max-kpts", type=int, default=2048)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--ratio-test", type=float, default=1.0)
    parser.add_argument("--feature-path", type=str, default=None)
    parser.add_argument("--custom-desc", type=str, default=None)
    parser.add_argument("--stats", type=str2bool, default=False)
    parser.add_argument("--save-csv", type=str2bool, default=True)
    parser.add_argument("--n-jobs", type=int, default=-1)  # mt not implemented yet
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
    feature_path = args.feature_path
    save_csv = args.save_csv

    # Fixed MNN configuration - using same defaults as MD1500
    mnn_min_score = args.min_score
    mnn_ratio_test = args.ratio_test

    # Define the wrapper
    wrapper = wrappers_manager(name=wrapper_name, device=device)

    if custom_desc is not None:
        try:
            logger.info(f"Loading custom descriptors from {custom_desc}")

            if not os.path.exists(custom_desc):
                raise FileNotFoundError(
                    f"Custom descriptor file not found: {custom_desc}"
                )

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
            network.load_state_dict(weights["state_dict"])
            wrapper.add_custom_descriptor(network)
            logger.info("Custom descriptor added to wrapper")

            old_name = wrapper.name
            wrapper.name = f"{wrapper.name}+SANDesc"
            logger.info(f"Wrapper name changed from '{old_name}' to '{wrapper.name}'")

        except Exception as e:
            logger.error(f"Failed to load custom descriptors from {custom_desc}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback

            logger.error(traceback.format_exc())
            raise

    # benchmark params - cleaner key format
    key = f"{wrapper.name}_{max_kpts}kpts"

    # add tag to the key
    if args.run_tag is not None:
        key = f"{key}_{args.run_tag}"

    logger.info(f"\n\n>>> Running HPatches benchmark for {key}...<<<\n")
    logger.info(f"Using thresholds: {thresholds}")
    logger.info(f"Using {n_jobs} jobs for parallel computation")
    logger.info(
        f"Using MNN matcher with min_score: {mnn_min_score}, ratio_test: {mnn_ratio_test}"
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
        feature_path=feature_path,
        save_csv=save_csv,
    )

    # Run the benchmark
    s = time.time()
    results, timestamp = benchmark.benchmark(wrapper, device=device)
    total_time = time.time() - s

    # Add timing info to results
    results["benchmark_time_seconds"] = float(total_time)

    print(f"Total time: {total_time:.1f} seconds")

    # Convert numpy types before saving to JSON
    serializable_results = convert_numpy_types(results)

    # Save the results
    data[f"{key}_{timestamp}"] = serializable_results
    with open(results_path / "results.json", "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"Results saved to {results_path/'results.json'}")

    # Print summary
    print("=== HPatches Overall Results ===")
    _ = display_hpatches_results(
        results_file=results_path / "results.json",
        partition="overall",
        method=f"{key}_{timestamp}",
    )
    print(f"{'='*80}")
