# This code is based on Parskatt implementation in DeDoDe.
# source: https://github.com/Parskatt/DeDoDe/blob/main/DeDoDe/benchmarks/mega_pose_est_mnn.py

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust as needed
sys.path.insert(0, str(PROJECT_ROOT))  # contains the 'methods' package

import warnings

warnings.filterwarnings("ignore")

import gc
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm
from matchers.mnn import MNN
from benchmarks.benchmark_utils import (
    str2bool,
    fix_rng,
    print_metrics,
    estimate_pose,
    compute_pose_error,
    compute_relative_pose,
    pose_auc,
)

DATASET_PATH = Path("benchmarks/graz_high_res/data/")
if not DATASET_PATH.exists():
    exit("Dataset not found")


class GrazHighResMNNBenchmark:
    def __init__(
        self,
        DATASET_PATH=DATASET_PATH,
        th=0.5,
        min_score: float = -1.0,
        ratio_test: float = 1.0,
        max_kpts: int = 2048,
        partial: bool = False,
    ) -> None:
        """
        Args:
            DATASET_PATH: path to the dataset
            scene_names: list of scene names to be used for the benchmark
            th: threshold for the pose estimation (max reprojection error)
            min_score: minimum score for the matches
            ratio_test: ratio test for the matches
            max_kpts: maximum number of keypoints to be extracted
            matcher: matcher to be used for the computation mnn or dual_softmax
        """
        partial_scene_id = 0  # 0 for graz_hauptplatz, 1 for graz_eggenberg

        sample_rate = 10
        min_matches = 100
        max_matches = 1000
        self.partial = partial

        pairs_path = "benchmarks/graz_high_res/data/pairs.npy"
        if Path(pairs_path).exists():
            self.scenes = np.load(pairs_path, allow_pickle=True).item()
        else:
            print(
                f"No pairs.npy found in {pairs_path}, generating pairs from viewgraph_30.txt"
            )
            self.scenes = {}
            for scene_name in os.listdir(DATASET_PATH):
                if not os.path.isdir(DATASET_PATH / scene_name):
                    continue
                with open(
                    f"{DATASET_PATH}/{scene_name}/colmap/viewgraph_30.txt", "r"
                ) as f:
                    lines = f.readlines()  # [[img1, img2, num of matches], ...]

                scene = np.array([line.split() for line in lines])
                matches = np.array(scene)[:, 2].astype(int)
                sampling = np.zeros_like(matches, dtype=bool)
                sampling[::sample_rate] = True
                mask = (
                    (matches > min_matches) & (matches < max_matches) & sampling
                )  # covers almost all images
                scene = scene[mask, :2]  # number of matches not needed anymore
                self.scenes[scene_name] = scene

            # sort by number of pairs
            self.scenes = dict(
                sorted(
                    self.scenes.items(), key=lambda item: len(item[1]), reverse=False
                )
            )
            np.save(pairs_path, self.scenes)
            print(f"Pairs saved to {pairs_path}")

        if self.partial:
            k = list(self.scenes.keys())[
                partial_scene_id
            ]  # only one scene for partial benchmarking
            self.scenes = {
                k: self.scenes[k]
            }  # only one small scene for quick benchmarking

        # Set the parameters
        self.data_root = Path(DATASET_PATH)
        self.max_kpts = max_kpts
        self.th = th

        # load matcher
        self.matcher = MNN(min_score=min_score, ratio_test=ratio_test)

        # printing stats
        pairs = [scene for scene in self.scenes.values()]
        total_pairs = np.vstack(pairs)
        total_images = np.unique(total_pairs.flatten())
        total_frames = len(glob.glob(f"{DATASET_PATH}/*/frames/*/*.jpg"))

        print(f'Scenes: {", ".join(list(self.scenes.keys()))}\n')
        print(f"Total pairs: {len(total_pairs):,}")
        print(
            f"Total images: {len(total_images):,}/{total_frames:,} "
            + f"({100*len(total_images)/total_frames:.2f}%)"
        )

    @torch.inference_mode()
    def benchmark(self, wrapper, factor=1, save_stats=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fix_rng()

        tot_e_t, tot_e_R, tot_e_pose, inlier = [], [], [], []
        thresholds = [5, 10, 20]
        stats_df = {}
        for scene_name in tqdm(self.scenes.keys(), desc="Processing scenes"):
            scene = self.scenes[scene_name]
            images_path = DATASET_PATH / scene_name / "frames"
            Ks = np.load(
                DATASET_PATH / scene_name / "cameras/cameras.npz", allow_pickle=True
            )
            poses = np.load(
                DATASET_PATH / scene_name / "cameras/poses.npz", allow_pickle=True
            )

            for pair_id, (img1_name, img2_name) in enumerate(
                tqdm(scene, desc=f"Processing scene {scene_name}")
            ):
                try:
                    # Data preparation
                    camera1 = poses[str(img1_name)].item()
                    T1 = camera1["P"].copy()
                    K1 = Ks[str(camera1["camera_id"])].item()["K"].copy()
                    R1, t1 = T1[:3, :3], T1[:3, 3]

                    camera2 = poses[str(img2_name)].item()
                    T2 = camera2["P"].copy()
                    K2 = Ks[str(camera2["camera_id"])].item()["K"].copy()
                    R2, t2 = T2[:3, :3], T2[:3, 3]

                    R, t = compute_relative_pose(R1, t1, R2, t2)

                    im_A_path = images_path / img1_name
                    im_B_path = images_path / img2_name
                    img_A = Image.open(im_A_path)
                    img_B = Image.open(im_B_path)

                except Exception as e:
                    print(
                        f"Error loading images {img1_name} and {img2_name} in scene {scene_name}: {e}"
                    )
                    continue

                if factor != 1:
                    W1, H1 = img_A.size
                    W2, H2 = img_B.size
                    img_A = img_A.resize((int(W1 / factor), int(H1 / factor)))
                    img_B = img_B.resize((int(W2 / factor), int(H2 / factor)))

                    # scale intrinsics
                    K1[:2, :3] /= factor
                    K2[:2, :3] /= factor

                img_A = wrapper.img_from_numpy(np.array(img_A))
                img_B = wrapper.img_from_numpy(np.array(img_B))

                # Local features extraction
                with torch.inference_mode():
                    out_A = wrapper.extract(img_A, self.max_kpts)
                    keypoints_A, description_A = out_A.kpts.cpu(), out_A.des.cpu()
                    # gc.collect()
                    # torch.cuda.empty_cache()

                    out_B = wrapper.extract(img_B, self.max_kpts)
                    keypoints_B, description_B = out_B.kpts.cpu(), out_B.des.cpu()
                    # gc.collect()
                    # torch.cuda.empty_cache()

                # matcher
                matches = self.matcher.match([description_A], [description_B])[
                    0
                ].matches
                kpts1 = keypoints_A[matches[:, 0]]
                kpts2 = keypoints_B[matches[:, 1]]

                # Pose Estimation
                shuffling = np.random.permutation(np.arange(len(kpts1)))
                kpts1 = kpts1[shuffling]
                kpts2 = kpts2[shuffling]
                try:
                    threshold = self.th
                    norm_threshold = threshold / (
                        np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2]))
                    )
                    R_est, t_est, mask = estimate_pose(
                        kpts1.cpu().numpy(),
                        kpts2.cpu().numpy(),
                        K1,
                        K2,
                        norm_threshold,
                        conf=0.99999,
                    )  # None if no pose is found

                    T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)
                    e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                    e_pose = max(e_t, e_R)
                    e_pose = (
                        180 if e_pose > 10 else e_pose
                    )  # cap at 10 degrees, even though it impact only AUC20
                    num_inliers = np.sum(mask)

                except Exception as e:
                    # print(repr(e)) # use to debug
                    e_t, e_R, e_pose = 180, 180, 180
                    num_inliers = 0

                tot_e_t.append(e_t)
                tot_e_R.append(e_R)
                tot_e_pose.append(e_pose)
                inlier.append(num_inliers)

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

                # gc.collect()
                # torch.cuda.empty_cache()

        # stats to csv and save
        if save_stats:
            stats_df = pd.DataFrame.from_dict(stats_df, orient="index")
            path = Path("benchmarks/graz_high_res/stats")
            os.makedirs(path, exist_ok=True)
            stats_df.to_csv(
                path
                / f'{wrapper.name}_stats_scale_{scale_factor}_\
                            {"partial" if self.partial is True else "full"}_\
                            {timestamp}.csv',
                index=False,
            )

        # Compute the metrics
        tot_e_pose = np.array(tot_e_pose)
        auc = pose_auc(tot_e_pose, thresholds)
        acc_5 = (tot_e_pose < 5).mean()
        acc_10 = (tot_e_pose < 10).mean()
        acc_20 = (tot_e_pose < 20).mean()
        map_5 = acc_5
        map_10 = np.mean([acc_5, acc_10])
        map_20 = np.mean([acc_5, acc_10, acc_20])

        return {
            "inlier": np.mean(inlier),
            "auc_5": auc[0],
            "auc_10": auc[1],
            "auc_20": auc[2],
            "map_5": map_5,
            "map_10": map_10,
            "map_20": map_20,
        }, timestamp


if __name__ == "__main__":
    import json
    import argparse
    from wrappers_manager import wrappers_manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wrapper-name", default="aliked")
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--ratio-test", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--max-kpts", type=int, default=2048)
    parser.add_argument("--th", type=float, default=0.75)
    parser.add_argument(
        "--custom-desc", type=str, default=None, help="Path to custom descriptor model"
    )
    parser.add_argument("--stats", type=str2bool, default=False)
    parser.add_argument("--partial", type=str2bool, default=False)
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1,
        help="Scale factor for resizing images. Default 1 is 4K.",
    )
    args = parser.parse_args()

    # Set the parameters
    device = args.device
    wrapper_name = args.wrapper_name
    ratio_test = args.ratio_test
    min_score = args.min_score
    max_kpts = args.max_kpts
    th = args.th
    partial = args.partial
    scale_factor = args.scale_factor
    custom_desc = args.custom_desc

    # Define the wrapper
    wrapper = wrappers_manager(name=wrapper_name, device=args.device)

    if custom_desc is not None:  # & it is a SANDesc model
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

        network = SANDesc(
            ch_in=model_config["ch_in"],
            kernel_size=model_config["kernel_size"],
            activ=model_config["activ"],
            norm=model_config["norm"],
            skip_connection=model_config["skip_connection"],
            spatial_attention=model_config["spatial_attention"],
            third_block=model_config["third_block"],
        )
        network = SANDesc(**model_config).eval().to(device)
        weights = torch.load(custom_desc, weights_only=False)
        network.load_state_dict(weights["state_dict"])
        wrapper.add_custom_descriptor(network)
        wrapper.name = f"{wrapper.name}+SANDesc"  # eventually change name
        print(f"Using custom descriptor: {custom_desc} with {wrapper.name} wrapper\n")

    # matcher
    key = f"{wrapper.name} ratio_test_{ratio_test}_min_score_{min_score}_th_{th} scale_{scale_factor} {max_kpts}"

    # adding partial or full and run_tag
    key += " partial" if partial is True else " full"
    key = f"{key} {args.run_tag}" if args.run_tag is not None else key
    print(f"\n\n>>> Running benchmark for {key} <<<\n\n")

    # create if not exists
    results_path = Path("benchmarks/graz_high_res/results")
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
    benchmark = GrazHighResMNNBenchmark(
        ratio_test=ratio_test,
        min_score=min_score,
        max_kpts=max_kpts,
        th=th,
        partial=partial,
    )

    # Run the benchmark
    results, timestamp = benchmark.benchmark(
        wrapper, save_stats=args.stats, factor=scale_factor
    )
    print_metrics(wrapper, results)
    print("------------------------------------------------------------------")

    # Save the results
    data[f"{key} {timestamp}"] = results
    with open(results_path / "results.json", "w") as f:
        json.dump(data, f)
    f.close()
