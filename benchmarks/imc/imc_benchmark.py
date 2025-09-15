import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

abs_root_path = Path(__file__).parents[2]
sys.path.append(str(abs_root_path))

import warnings

warnings.filterwarnings("ignore")

import gc
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm

from matchers.mnn import MNN
from benchmarks.benchmark_utils import print_metrics
from benchmarks.imc.imc_benchmark_utils import (
    extract_image_matching_benchmark,
    match_features,
    import_data_to_benchmark,
    run_benchmark,
    parse_results,
)

logger = logging.getLogger(__name__)


class IMC21MNNBenchmark:
    def __init__(
        self,
        device: str = "cuda",
        data_path=abs_root_path / "benchmarks/imc/data/phototourism",
        th=1.0,
        min_score: float = 0.0,
        ratio_test: float = 1.0,
        max_kpts: int = 2048,
        overwrite_extraction: bool = False,
        njobs: int = 16,
    ) -> None:
        """
        Args:
            DATASET_PATH: path to the dataset
            th: threshold for the pose estimation (max reprojection error)
            min_score: minimum score for the matches
            ratio_test: ratio test for the matches
            max_kpts: maximum number of keypoints to be extracted
        """
        self.device = device
        self.data_path = data_path
        self.th = th
        self.min_score = min_score
        self.ratio_test = ratio_test
        self.max_kpts = max_kpts
        self.overwrite_extraction = overwrite_extraction
        self.njobs = njobs

        # Matcher
        self.matcher = MNN(ratio_test=ratio_test, min_score=min_score, device=device)

    @torch.no_grad()
    def benchmark(self, wrapper):
        """
        Run the IMC benchmark

        Args:
            wrapper: The feature extraction wrapper
            calibrated: Whether to use calibrated pose estimation

        Returns:
            tuple: (results_dict, timestamp)
        """
        method_name = f"{wrapper.name}_{self.max_kpts}kpts"

        # Step 1: Extract features and store them
        output_path = (
            abs_root_path
            / f"benchmarks/imc/to_import_imc/{wrapper.name}_{self.max_kpts}kpts"
        )
        os.makedirs(output_path, exist_ok=True)

        if self.overwrite_extraction or not any(output_path.iterdir()):
            if self.overwrite_extraction:
                for file in output_path.iterdir():
                    os.system(f"rm -rf {file}")
                logger.info(f"Features in {output_path} have been removed.")

            extract_image_matching_benchmark(
                wrapper, self.data_path, max_kpts=self.max_kpts, output_path=output_path
            )
        elif output_path.exists() and any(output_path.iterdir()):
            logger.info(
                f"Features already extracted in {output_path}. Skipping extraction."
            )

        # Step 2: Match them
        # If already matched, will return the path.
        matched_features_path = match_features(
            method_name,
            output_path,
            self.matcher,
            ransac_thr=self.th,
            device=self.device,
            njobs=self.njobs,
            overwrite_extraction=self.overwrite_extraction,
        )

        # Step 3: Import to the IMC benchmark
        import_data_to_benchmark(
            matched_features_path,
            scenes_set="test",
            matcher_name=self.matcher.name,
        )
        logger.info("Data imported to the IMC benchmark.")

        method_name_json = run_benchmark(
            method_name, self.matcher.name, scenes_set="test"
        )
        logger.info(f"Method name for the json: {method_name_json}")

        # results = parse_results(method_name_json, scenes_set="test")
        # logger.info(f"Results for {method_name}: {results}")

        return {}, ""


if __name__ == "__main__":
    import json
    import argparse
    from wrappers_manager import wrappers_manager

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wrapper-name", type=str, default="superpoint")
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--ratio-test", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-kpts", type=int, default=2048)
    parser.add_argument("--th", type=float, default=0.5)
    parser.add_argument("--custom-desc", type=str, default=None)
    parser.add_argument("--njobs", type=int, default=16)
    parser.add_argument(
        "--overwrite-extraction",
        action="store_true",
        help="Overwrite existing features",
    )
    args = parser.parse_args()

    device = args.device
    wrapper_name = args.wrapper_name
    ratio_test = args.ratio_test
    min_score = args.min_score
    max_kpts = args.max_kpts
    th = args.th
    custom_desc = args.custom_desc
    njobs = args.njobs

    # Define the wrapper
    wrapper = wrappers_manager(name=wrapper_name, device=args.device)

    if custom_desc is not None:
        #  Eventually add my descriptors
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

    # matcher params
    key = f"{wrapper.name} ratio_test_{ratio_test}_min_score_{min_score}_th_{th}_mnn {max_kpts}"

    # add tag to the key
    if args.run_tag is not None:
        key = f"{key} {args.run_tag}"

    print(f"\n>>> Running benchmark for {key} <<<\n")

    # create if not exists
    results_path = Path("benchmarks/imc/results")
    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(results_path / "results.json"):
        with open(results_path / "results.json", "w") as f:
            json.dump({}, f)
        f.close()

    with open(results_path / "results.json", "r") as f:
        data = json.load(f)

    if key in data:
        results = data[key]
        warnings.warn("A similar run already exists.", UserWarning)

    f.close()

    # Define the benchmark
    benchmark = IMC21MNNBenchmark(
        ratio_test=ratio_test,
        min_score=min_score,
        max_kpts=max_kpts,
        th=th,
        overwrite_extraction=args.overwrite_extraction,
        device=device,
        njobs=njobs,
    )

    # Run the benchmark
    results, timestamp = benchmark.benchmark(wrapper)
    print_metrics(wrapper, results)
    print("-------------------------------------------------------------")

    # Save the results
    data[f"{key} {timestamp}"] = results
    with open(results_path / "results.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {results_path / 'results.json'}")
