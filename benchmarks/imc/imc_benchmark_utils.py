import sys
from pathlib import Path

abs_root_path = Path(__file__).parent.parent
sys.path.append(str(abs_root_path))

import os
import glob
import h5py
import json
import logging
import subprocess
import pydegensac
import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from tqdm.auto import tqdm
from functools import partial
from typing import Union, Dict
from itertools import combinations
from joblib import Parallel, delayed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

validation_scenes = ["reichstag", "sacre_coeur", "st_peters_square"]
test_scenes = [
    "british_museum",
    "florence_cathedral_side",
    "lincoln_memorial_statue",
    "london_bridge",
    "milan_cathedral",
    "mount_rushmore",
    "piazza_san_marco",
    "sagrada_familia",
    "st_pauls_cathedral",
]


def save_h5(dict_to_save: Dict[str, Tensor], filename: str) -> None:
    """Saves dictionary to HDF5 file"""
    with h5py.File(filename, "w") as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])


def load_h5(filename: str) -> Dict[str, Tensor]:
    """Load dictionary from HDF5 file"""
    output = {}
    with h5py.File(filename, "r") as f:
        keys = list(f.keys())
        for key in keys:
            output[key] = th.tensor(f[key][()])
    return output


def extract_image_matching_benchmark(
    wrapper,
    data_path: str = "benchmarks/imc/data/phototourism",
    max_kpts: int = 2048,
    output_path: str = None,
    scene_set: str = "test",
) -> Union[Path, None]:

    data_path = Path(data_path)
    assert data_path.exists(), f"Dataset path {data_path} does not exist."

    scenes = validation_scenes if scene_set == "val" else test_scenes
    scenes_paths = sorted([data_path / scene for scene in scenes])

    # Create the phototourism subdirectory to match expected structure
    phototourism_path = output_path / "phototourism"
    phototourism_path.mkdir(parents=True, exist_ok=True)

    # ? extract features for each scene in the dataset
    scenes_bar = tqdm(scenes_paths, position=0)
    for scene_path in scenes_bar:
        scenes_bar.set_description(f"Extracting scene {scene_path.name}")
        scene_output_path = phototourism_path / scene_path.name  # Changed this line
        scene_output_path.mkdir(parents=True, exist_ok=True)

        scene_imgs_path = scene_path / "set_100" / "images"
        imgs_paths = sorted(
            [
                path
                for path in scene_imgs_path.iterdir()
                if not path.stem.startswith(".")  # ignore hidden files
            ]
        )

        keypoints, descriptors = {}, {}

        imgs_bar = tqdm(imgs_paths, position=1)
        for img_path in imgs_bar:
            img_name = img_path.stem
            img_np = np.array(Image.open(img_path))
            with th.no_grad():
                img = wrapper.img_from_numpy(img_np)
                output = wrapper.extract(img, max_kpts=max_kpts)

            keypoints[img_name] = output["kpts"].cpu()
            descriptors[img_name] = output["des"].cpu()
            imgs_bar.set_description(
                f"extracted keypoints: {keypoints[img_name].shape[0]}"
            )

        # ? export keypoints
        save_h5(keypoints, scene_output_path / "keypoints.h5")
        # ? export descriptors
        save_h5(descriptors, scene_output_path / "descriptors.h5")

    return output_path


def filter_matches(
    pair_name: str, pair_matches: Tensor, kpts: Dict[str, Tensor], ransac_thr: float
) -> Dict:
    # ? geometrical filter the pairs
    img0_name, img1_name = pair_name.split("-")

    filtering_params = {
        "method": "cmp-degensac-f",
        "threshold": ransac_thr,
        "confidence": 0.999999,
        "max_iter": 100_000,
        "error_type": "sampson",
        "degeneracy_check": True,
    }
    kpts0 = kpts[img0_name]
    kpts1 = kpts[img1_name]
    kpts0_matched = kpts0[pair_matches[:, 0]].cpu().numpy()
    kpts1_matched = kpts1[pair_matches[:, 1]].cpu().numpy()

    if kpts0_matched.shape[0] > 7 and kpts1_matched.shape[0] > 7:
        F, inlier_mask = pydegensac.findFundamentalMatrix(
            kpts0_matched,
            kpts1_matched,
            px_th=filtering_params["threshold"],
            conf=filtering_params["confidence"],
            max_iters=filtering_params["max_iter"],
            laf_consistensy_coef=0,
            error_type=filtering_params["error_type"],
            symmetric_error_check=True,
            enable_degeneracy_check=filtering_params["degeneracy_check"],
        )

        return {pair_name: pair_matches[inlier_mask].T}
    else:
        return {pair_name: np.zeros((0, 2))}


def match_features(
    method_name: str,
    path: Path,
    matcher,
    ransac_thr: float,
    device: str,
    njobs: int = 12,
    overwrite_extraction=False,
):
    # if debugging jobs =1
    if sys.gettrace() is not None:
        njobs = 1

    # create output path or use the existing one
    matcher.name = f"{matcher.name}_Ransac-{ransac_thr}"
    output_path = Path(path).parent / f"{method_name}_Matcher-{matcher.name}"

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Matches will be saved in {output_path}")
    else:
        logger.info(
            f"Matches might already exist in {output_path}. Found {len(list(output_path.iterdir()))} scenes."
        )
        if not overwrite_extraction:
            return output_path
        else:
            logger.info(f"Overwriting matches in {output_path}.")
            for file in output_path.iterdir():
                os.system(f"rm -rf {file}")

    # load all the descriptors
    logger.info(
        f"Geometrically filtering the matches with RANSAC threshold {ransac_thr} with {njobs} jobs"
    )
    path = Path(path) / "phototourism"
    bar = tqdm(sorted(list(path.iterdir())), position=0)
    for scene_path in bar:
        bar.set_description(f"Processing scene {scene_path.name}")
        # load descriptors and keypoints
        desc_path = scene_path / "descriptors.h5"
        kpts_path = scene_path / "keypoints.h5"
        des_all = load_h5(desc_path)
        kpts_all = load_h5(kpts_path)

        # match all possible pairs
        pairs = list(combinations(list(des_all.keys()), 2))
        matches = {}
        for img1_name, img0_name in tqdm(
            pairs, position=1, desc="Matching descriptors"
        ):
            des0 = des_all[img0_name]
            des1 = des_all[img1_name]
            matches_all = matcher.match([des0.to(device)], [des1.to(device)])[
                0
            ].matches.cpu()
            matches[f"{img0_name}-{img1_name}"] = matches_all

        # Geometrically filter the matches
        filter_matches_partial = partial(
            filter_matches, kpts=kpts_all, ransac_thr=ransac_thr
        )

        matches_filtered = Parallel(n_jobs=njobs)(
            delayed(filter_matches_partial)(pair_name, pair_matches)
            for pair_name, pair_matches in tqdm(
                matches.items(), position=1, desc="Geometric filtering the matches"
            )
        )
        matches_filtered = {k: v for x in matches_filtered for k, v in x.items()}

        # save matches
        output_path_scene = output_path / "phototourism" / scene_path.name
        output_path_scene.mkdir(parents=True, exist_ok=True)
        # copy keypoints and descriptors
        os.system(f"cp {kpts_path} {output_path_scene / 'keypoints.h5'}")
        os.system(f"cp {desc_path} {output_path_scene / 'descriptors.h5'}")
        # save matches
        save_h5(matches, output_path_scene / "matches_multiview.h5")
        save_h5(matches_filtered, output_path_scene / "matches_stereo.h5")

    return output_path


def import_data_to_benchmark(
    extracted_path: Path, scenes_set: str, matcher_name: str = None
):
    logger.info("Importing into benchmark")

    # Convert Path to string for subprocess
    extracted_path_str = str(extracted_path)

    if matcher_name is not None:
        subprocess.call(
            [
                sys.executable,
                "import_features.py",
                "--path_features",
                extracted_path_str,  # Convert to string
                "--kp_name",
                extracted_path.name,  # Remove f-string
                "--desc_name",
                extracted_path.name,  # Remove f-string
                "--match_name",
                matcher_name,
                "--subset",
                scenes_set,
                "--datasets",
                "phototourism",
            ],
            cwd=abs_root_path / "imc/image-matching-benchmark",
        )
    else:
        subprocess.call(
            [
                sys.executable,
                "import_features.py",
                "--path_features",
                extracted_path_str,  # Convert to string
                "--kp_name",
                extracted_path.name,  # Remove f-string
                "--desc_name",
                extracted_path.name,  # Remove f-string
                "--subset",
                scenes_set,
                "--datasets",
                "phototourism",
            ],
            cwd=abs_root_path / "imc/image-matching-benchmark",
        )


def generate_json(method_name: str, matcher_name: str = None, num_kpts: int = 2048):
    method_name_converted = method_name.replace("_", "-").lower()

    if matcher_name is not None:
        matcher_name_converted = matcher_name.replace("_", "-").lower()
        method_name_json = f"{method_name_converted}-matcher-{matcher_name_converted}"
        logger.info(matcher_name_converted)
        config = {
            "config_common": {
                "json_label": method_name_json,
                "keypoint": method_name_json,
                "descriptor": method_name_json,
                "num_keypoints": num_kpts,
            },
            "config_phototourism_stereo": {
                "use_custom_matches": True,
                # 'custom_matches_name': f'{matcher_name_converted}-stereo',
                "custom_matches_name": f"{matcher_name_converted}",
                "geom": {"method": "cv2-8pt"},
            },
        }

    else:
        method_name_json = method_name_converted
        config = {
            "config_common": {
                "json_label": method_name_json,
                "keypoint": method_name_converted,
                "descriptor": method_name_converted,
                "num_keypoints": 2048,
            },
            "config_phototourism_stereo": {
                "use_custom_matches": False,
                "matcher": {
                    "method": "nn",
                    "distance": "L2",
                    "flann": False,
                    "num_nn": 1,
                    "filtering": {"type": "none"},
                    "symmetric": {"enabled": True, "reduce": "both"},
                },
                "outlier_filter": {"method": None},
                "geom": {
                    "method": "cmp-degensac-f",
                    "threshold": 1.0,
                    "confidence": 0.999999,
                    "max_iter": 100000,
                    "error_type": "sampson",
                    "degeneracy_check": True,
                },
            },
        }

    # ? save config as .json
    output_path = Path("/tmp/") / "config.json"
    with open(output_path, "w") as f:
        json.dump([config], f, indent=4)
    return output_path, method_name_json


def run_benchmark(
    method_name: str,
    matcher_name: str,
    scenes_set: str,
    num_kpts: int = 2048,
    multiview: bool = False,
) -> str:
    json_path, method_name_json = generate_json(
        method_name, matcher_name, num_kpts=num_kpts
    )
    subprocess.call(
        [
            sys.executable,
            "run.py",
            "--json_method",
            json_path,
            "--subset",
            scenes_set,
            "--run_mode",
            "interactive",
            "--eval_multiview",
            str(multiview).lower(),
            "--parallel",
            "1",
        ],
        cwd=abs_root_path / "imc/image-matching-benchmark",
    )
    return method_name_json


def load_imc_results(
    res_path="imc/image-matching-benchmark/packed-test",
    return_df=False,
    all_scenes=False,
    auc_th=5,  # can be 5 or 10
):
    """Load IMC results from given JSON files and return a summary dictionary.
    Args:
        res_path (str): Path to the directory containing JSON result files.
        return_df (bool): If True, returns results as a pandas DataFrame. Default is False.
        all_scenes (bool): If True, includes all scenes in the summary. Default is False.
        auc_th (int): Threshold for AUC metrics (5 or 10). Default is 5.
    Returns:
        dict or pd.DataFrame: Summary of results either as a dictionary or DataFrame.
    """
    scene_short = {
        "allseq": "all",
    }
    scenes = {
        "british_museum": "BM",
        "florence_cathedral_side": "FC",
        "lincoln_memorial_statue": "LM",
        "london_bridge": "LB",
        "milan_cathedral": "MC",
        "mount_rushmore": "MR",
        "piazza_san_marco": "PS",
        "sagrada_familia": "SA",
        "st_pauls_cathedral": "SPC",
        "reichstag": "RE",
        "sacre_coeur": "SC",
        "st_peters_square": "SPS",
    }
    if all_scenes:
        scene_short.update(scenes)

    df_results = {}

    paths = glob.glob(res_path + "/*.json")
    if len(paths) == 0:
        print(f"No JSON files found in {res_path}.")
        return

    for path in paths:
        with open(path) as f:
            # load results
            res = json.load(f)
            res_phototourism = res["phototourism"]["results"]

            # method name
            method_config = (
                res["config"]["config_common"]["keypoint"]
                .replace("dedode-g", "dedodeG")
                .replace("sandesc", "+SANDesc")
            )

            method_data = {}

            for scene_name, scene_short_name in scene_short.items():
                if scene_name not in res_phototourism:
                    continue

                res_scene = res_phototourism[scene_name]["stereo"]["run_avg"]

                # Add metrics for this scene - multiply by 100 and round to 1 decimal except inliers
                rep = (
                    res_scene.get("repeatability", {}).get("mean", [0, 0, 0])[2]
                    if res_scene.get("repeatability", {}).get("mean")
                    else 0
                )

                # Select AUC metric based on auc_th parameter
                if auc_th == 5:
                    auc = res_scene.get("qt_auc_05", {}).get("mean", 0)
                    auc_col_name = f"{scene_short_name}_auc5"
                elif auc_th == 10:
                    auc = res_scene.get("qt_auc_10", {}).get("mean", 0)
                    auc_col_name = f"{scene_short_name}_auc10"
                else:
                    raise ValueError(f"auc_th must be 5 or 10, got {auc_th}")

                inliers = res_scene.get("num_matches_geom_th_0.1", {}).get("mean", 0)

                method_data[f"{scene_short_name}_rep"] = round(rep * 100, 1)
                method_data[auc_col_name] = round(auc * 100, 1)
                method_data[f"{scene_short_name}_inliers"] = round(
                    inliers, 0
                )  # Keep inliers as whole numbers

            df_results[method_config] = method_data

    if return_df:
        df = pd.DataFrame(df_results).T

        # Split the index into method, kpts_budget, and params
        df_split = df.reset_index()
        df_split["index_parts"] = df_split["index"].str.split("-")

        # Extract components with safe indexing
        df_split["method"] = df_split["index_parts"].apply(
            lambda x: x[0] if len(x) > 0 else ""
        )
        df_split["custom_desc"] = df_split["index_parts"].apply(
            lambda x: (
                "SANDesc"
                if any("sandesc" in part.lower() for part in x)
                else ("G" if any("G" in part for part in x) else "")
            )
        )
        # resome +sandesc from method name
        df_split["method"] = df_split["method"].str.replace("+sandesc", "", case=False)
        df_split["method"] = df_split["method"].str.replace("+SANDesc", "", case=False)
        df_split["method"] = df_split["method"].str.replace("G", "", case=False)

        df_split["kpts_budget"] = df_split["index_parts"].apply(
            lambda x: x[1] if len(x) > 1 else ""
        )
        df_split["params"] = df_split["index_parts"].apply(
            lambda x: "-".join(x[2:]) if len(x) > 2 else ""
        )

        # Drop unnecessary columns and reorder
        df = df_split.drop(["index"], axis=1)

        # Group columns by metric type
        metric_cols = [
            col
            for col in df.columns
            if col not in ["method", "custom_desc", "kpts_budget", "params"]
        ]

        # Sort scenes for consistent ordering
        scene_order = sorted(scene_short.values())

        # Group columns by metric type
        rep_cols = [
            f"{scene}_rep" for scene in scene_order if f"{scene}_rep" in metric_cols
        ]
        auc_suffix = f"auc{auc_th}"
        auc_cols = [
            f"{scene}_{auc_suffix}"
            for scene in scene_order
            if f"{scene}_{auc_suffix}" in metric_cols
        ]
        inliers_cols = [
            f"{scene}_inliers"
            for scene in scene_order
            if f"{scene}_inliers" in metric_cols
        ]

        # Reorder columns: method info first, then grouped metrics
        column_order = (
            ["method", "custom_desc", "kpts_budget", "params"]
            + rep_cols
            + inliers_cols
            + auc_cols
        )
        df = df[column_order]

        # Set simple integer index
        df.index = range(len(df))

        return df.sort_values(
            by=["method", "custom_desc"], ascending=[True, True]
        ).reset_index(drop=True)

    return df_results
