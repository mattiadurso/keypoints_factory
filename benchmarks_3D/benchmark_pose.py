from __future__ import annotations

import os

import pycolmap
import logging
import argparse

import glob
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from tqdm.auto import tqdm
from pathlib import Path
from itertools import combinations
from utils_benchmark_pose import compute_AUC, evaluate_R_t

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_scene(target_rec, input_rec, deg=True):
    """
    Given two dictionaries {"image_idx":{qvec, tvec}}, evaluate the relative pose between all the possible pairs of images.
    Args:
        images_target_dict:   dictionary with the ground truth poses.
        images_pred_dict: dictionary with the predicted poses.
        deg: if True, the errors are returned in degrees else in radians. Default is True.
    Returns:
        df: dataframe with the relative pose errors with keys {image1, image2, q_error, t_error}.
    """

    df = {
        "image1": [],
        "image2": [],
        "q_error": [],
        "t_error": [],
        "max_error": [],
    }
    target_images = np.array(
        sorted([img.name for img in target_rec.images.values()])
    )  # remove eventual subdirectory in the image name (e.g. camera calibration folder)
    input_images = np.array(sorted([img.name for img in input_rec.images.values()]))

    # for each pair of images in the ground truth
    for image_1_path, image_2_path in combinations(target_images, 2):
        if not (
            (image_1_path in input_images) and (image_2_path in input_images)
        ):  # working?
            q_err, t_err, max_error = np.inf, np.inf, np.inf
            logger.info(f"Image {image_1_path} or {image_2_path} not in input model.")
        else:
            # get the rotation and translation for two images (target)
            R1_target, t1_target = (
                target_rec.find_image_with_name(
                    image_1_path
                ).cam_from_world.rotation.matrix(),
                target_rec.find_image_with_name(
                    image_1_path
                ).cam_from_world.translation,
            )
            R2_target, t2_target = (
                target_rec.find_image_with_name(
                    image_2_path
                ).cam_from_world.rotation.matrix(),
                target_rec.find_image_with_name(
                    image_2_path
                ).cam_from_world.translation,
            )

            # Be careful here: image names in input and target might contain subdirectories. I am not accounting for that.
            # since VGGT read images from
            R1_input, t1_input = (
                input_rec.find_image_with_name(
                    image_1_path
                ).cam_from_world.rotation.matrix(),
                input_rec.find_image_with_name(image_1_path).cam_from_world.translation,
            )
            R2_input, t2_input = (
                input_rec.find_image_with_name(
                    image_2_path
                ).cam_from_world.rotation.matrix(),
                input_rec.find_image_with_name(image_2_path).cam_from_world.translation,
            )

            # compute the relative pose between the two images (target)
            R_target = R2_target @ R1_target.T
            t_target = t2_target - R_target @ t1_target

            # compute the relative pose between the two images (input)
            R_pred = R2_input @ R1_input.T
            t_pred = t2_input - R_pred @ t1_input

            # compute the error
            q_err, t_err = evaluate_R_t(R_pred, t_pred, R_target, t_target, deg=deg)
            max_error = max(q_err, t_err)

        # append to the dataframe
        df["image1"].append(image_1_path)
        df["image2"].append(image_2_path)
        df["q_error"].append(q_err)
        df["t_error"].append(t_err)
        df["max_error"].append(max_error)  # if max_error < 10 else np.inf)

    return pd.DataFrame(df)


def eval_colmap_model(
    model_path, target_path, thrs=[1, 3, 5], return_df=False, AUC_col="max_error"
):
    """
    Given a scene path, evaluate the model in the given folder versus the ground truth in the target_folder.
    Args:
        model_path: path to the model to evaluate.
        target_path:    path to the ground truth model.
        thrs:       list of thresholds for the AUC computation.
        return_df:  if True, return the dataframe with the errors.
        AUC_col:    column to compute the AUC. Default is "max_error". Other options are "q_error" and "t_error".
    Returns:
        df_AUC: dataframe with the AUC values for the model.

    """

    if not os.path.exists(model_path):
        raise Exception(f"Path {model_path} does not exist.")

    if not os.path.exists(target_path):
        raise Exception(f"Path {target_path} does not exist.")

    # read models
    rec_input = pycolmap.Reconstruction(model_path)
    rec_target = pycolmap.Reconstruction(target_path)

    # evaluate scene (each pair of images) and compute the AUC
    df = evaluate_scene(rec_target, rec_input)
    AUC_score_max = np.array(compute_AUC(df[AUC_col], thrs))

    if return_df:
        return AUC_score_max, df

    return AUC_score_max


def eval_colmap_model_all_scenes(
    input_path,
    target_path,
    input_folder="colmap/sparse/0",
    target_folder="sparse",
    thrs=[0.5, 1, 3, 5, 10],
    AUC_col="max_error",
    n_jobs=-1,
) -> pd.DataFrame:
    """
    Evaluate the model on all the scenes in the data_path using parallel processing.
    These must be in COLMAP format. The model is evaluated at the specified thresholds.
    """

    # Get scene names from both directories
    input_scene_names = set(os.listdir(input_path))
    target_scene_names = set(os.listdir(target_path))

    # Keep only common scenes
    common_scenes = sorted(input_scene_names & target_scene_names)

    print(f"Found {len(common_scenes)} common scenes.")

    if len(common_scenes) == 0:
        logger.warning("No common scenes found!")
        return pd.DataFrame()

    # Build paths for common scenes only
    input_paths = [
        os.path.join(input_path, scene, input_folder) for scene in common_scenes
    ]
    target_paths = [
        os.path.join(target_path, scene, target_folder) for scene in common_scenes
    ]

    # Verify paths exist
    valid_pairs = []
    valid_scenes = []
    for inp, tgt, scene in zip(input_paths, target_paths, common_scenes):
        if os.path.exists(inp) and os.path.exists(tgt):
            valid_pairs.append((inp, tgt))
            valid_scenes.append(scene)
        else:
            logger.warning(f"Skipping {scene}: paths don't exist")

    print(f"Evaluating {len(valid_pairs)} valid scenes.")

    # Use joblib to parallelize the evaluation of each scene
    results = Parallel(n_jobs=n_jobs)(
        delayed(eval_colmap_model)(
            input, target, thrs=thrs, return_df=False, AUC_col=AUC_col
        )
        for input, target in tqdm(
            valid_pairs,
            desc="Evaluating scenes",
            total=len(valid_pairs),
        )
    )

    # Process results and create the DataFrame
    res = {}
    for auc_scores, scene_name in zip(results, valid_scenes):
        if auc_scores is not None:
            res[scene_name] = auc_scores

    # Creating the DataFrame and transposing it to have the scenes as rows
    df_res_colmap = pd.DataFrame(res, index=thrs).transpose()
    # sort by scene name
    df_res_colmap = df_res_colmap.sort_index()

    # Rename the columns as {model}@{thrs}
    df_res_colmap.columns = [f"auc@{thr}" for thr in thrs]
    return df_res_colmap.round(2)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-model",
        type=str,
        required=True,
        help="Path to the input 3D model file or directorys. E.g., scenes/scene_name/colmap/sparse/0 or scenes/",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Path to the target 3D model file or directorys. E.g., scenes/scene_name/colmap/sparse/0 or scenes/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks_3D/results",
        help="Directory to save the output results.",
    )
    parser.add_argument(
        "--many-scenes",
        action="store_true",
        help="If set, evaluate all scenes in the input directory.",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="colmap/sparse/0",
        help="Subfolder in each scene folder where the input model is located. Used only if --many-scenes is set.",
    )
    parser.add_argument(
        "--target-folder",
        type=str,
        default="sparse",
        help="Subfolder in each scene folder where the ground truth model is located. Used only if --many-scenes is set.",
    )
    parser.add_argument(
        "--mapper",
        type=str,
        default="colmap",
        help="Name of the mapper used to generate the input model. Used only if --many-scenes is set.",
    )
    parser.add_argument(
        "--thrs",
        type=float,
        nargs="+",
        default=[1, 3, 5, 10, 15, 30],
        help="Thresholds for AUC computation.",
    )
    args = parser.parse_args()

    s = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.many_scenes:
        df_res = eval_colmap_model_all_scenes(
            args.input_model,
            args.target_model,
            thrs=args.thrs,
            n_jobs=16,
            input_folder=args.input_folder,
            target_folder=args.target_folder,
            mapper=args.mapper,
        )
        df_res.to_csv(
            os.path.join(args.output_dir, f"results_all_scenes_{args.mapper}.csv"),
            index=True,
        )
        print(df_res)
    else:
        auc_scores, df = eval_colmap_model(
            args.input_model,
            args.target_model,
            thrs=args.thrs,
            return_df=True,
        )
        print(f"AUC scores at {args.thrs}: {auc_scores}")
        df.to_csv(
            os.path.join(args.output_dir, f"results_single_scene_{args.mapper}.csv"),
            index=False,
        )
        print(df)

    print(f"Total time: {time.time() - s:.2f} seconds.")
