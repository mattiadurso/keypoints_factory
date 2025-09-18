import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

abs_root_path = Path(__file__).parents[2]
sys.path.append(str(abs_root_path))

import gc
import torch
import logging

from matchers.mnn import MNN
from benchmarks.imc.imc_benchmark_utils import (
    extract_image_matching_benchmark,
    match_features,
    import_data_to_benchmark,
    run_benchmark,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
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
        scene_set: str = "test",
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
        self.scene_set = scene_set

        # Matcher
        self.matcher = MNN(
            ratio_test=ratio_test,
            min_score=min_score,
            device="cpu",  # one can also use cuda here. With cpu, gpu stays free.
        )

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
            / f"benchmarks/imc/to_import_imc/{self.scene_set}/{wrapper.name}_{self.max_kpts}kpts"
        )
        os.makedirs(output_path, exist_ok=True)

        if self.overwrite_extraction or not any(output_path.iterdir()):
            if self.overwrite_extraction:
                for file in output_path.iterdir():
                    os.system(f"rm -rf {file}")
                logger.info(f"Features in {output_path} have been removed.")

            extract_image_matching_benchmark(
                wrapper=wrapper,
                data_path=self.data_path,
                max_kpts=self.max_kpts,
                output_path=output_path,
                scene_set=self.scene_set,
            )
        elif output_path.exists() and any(output_path.iterdir()):
            logger.info(
                f"Features already extracted in {output_path}. Skipping extraction."
            )

        # free gpu memory, from here wtrapper is not needed anymore
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()

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
            scenes_set=self.scene_set,
            matcher_name=self.matcher.name,
        )
        logger.info("Data imported to the IMC benchmark.")

        method_name_json = run_benchmark(
            method_name, self.matcher.name, scenes_set=self.scene_set
        )

        # Step 4: Copy results to the results folder
        results_path = abs_root_path / f"benchmarks/imc/results/{self.scene_set}"
        os.makedirs(results_path, exist_ok=True)
        os.system(
            f"cp \
                  benchmarks/imc/image-matching-benchmark/packed-{self.scene_set}/{method_name_json}.json \
                  {results_path / method_name_json}.json"
        )

        logger.info(
            f"{method_name_json} results copyed to benchmarks/imc/results. \n\n"
        )


if __name__ == "__main__":
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
    parser.add_argument("--max-kpts", type=int, default=2048)
    parser.add_argument("--th", type=float, default=1.0)
    parser.add_argument("--ratio-test", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--custom-desc", type=str, default=None)
    parser.add_argument("--njobs", type=int, default=18)
    parser.add_argument("--scene-set", type=str, default="test")
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
        wrapper.name = f"{wrapper.name}SANDesc"
        print(f"Using custom descriptors from {custom_desc}.\n")

    # matcher params
    key = f"{wrapper.name} ratio_test_{ratio_test}_min_score_{min_score}_th_{th}_mnn {max_kpts}"

    # add tag to the key
    if args.run_tag is not None:
        key = f"{key} {args.run_tag}"

    print(f"\n>>> Running benchmark for {key} <<<\n")

    # Define the benchmark
    benchmark = IMC21MNNBenchmark(
        ratio_test=ratio_test,
        min_score=min_score,
        max_kpts=max_kpts,
        th=th,
        overwrite_extraction=args.overwrite_extraction,
        device=device,
        njobs=njobs,
        scene_set=args.scene_set,
    )

    # Run the benchmark
    benchmark.benchmark(wrapper)
    print("-------------------------------------------------------------")
