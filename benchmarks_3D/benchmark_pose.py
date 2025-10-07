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
        images_gt_dict:   dictionary with the ground truth poses.
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
    target_images = [img.name for img in target_rec.images.values()]
    input_images = [img.name for img in input_rec.images.values()]

    # for each pair of images in the ground truth
    for image_1_path, image_2_path in combinations(target_images, 2):
        # set to inf if both images have not been registered (= in *_pred_dict)
        if not (
            image_1_path in input_images and image_2_path in input_images
        ):  # working?
            q_err, t_err, max_error = np.inf, np.inf, np.inf
        else:
            # get the rotation and translation for two images (target)
            R1_gt, t1_gt = (
                target_rec.find_image_with_name(image_1_path)
                .cam_from_world()
                .rotation.matrix(),
                target_rec.find_image_with_name(image_1_path)
                .cam_from_world()
                .translation,
            )
            R2_gt, t2_gt = (
                target_rec.find_image_with_name(image_2_path)
                .cam_from_world()
                .rotation.matrix(),
                target_rec.find_image_with_name(image_2_path)
                .cam_from_world()
                .translation,
            )

            # get the rotation and translation for two images (input)
            R1_input, t1_input = (
                input_rec.find_image_with_name(image_1_path)
                .cam_from_world()
                .rotation.matrix(),
                input_rec.find_image_with_name(image_1_path)
                .cam_from_world()
                .translation,
            )
            R2_input, t2_input = (
                input_rec.find_image_with_name(image_2_path)
                .cam_from_world()
                .rotation.matrix(),
                input_rec.find_image_with_name(image_2_path)
                .cam_from_world()
                .translation,
            )

            # compute the relative pose between the two images (target)
            R_gt = R2_gt @ R1_gt.T
            t_gt = t2_gt - R_gt @ t1_gt

            # compute the relative pose between the two images (input)
            R_pred = R2_input @ R1_input.T
            t_pred = t2_input - R_pred @ t1_input

            # compute the error
            q_err, t_err = evaluate_R_t(R_pred, t_pred, R_gt, t_gt, deg=deg)
            max_error = max(q_err, t_err)

        # append to the dataframe
        df["image1"].append(image_1_path)
        df["image2"].append(image_2_path)
        df["q_error"].append(q_err)
        df["t_error"].append(t_err)
        df["max_error"].append(max_error if max_error < 10 else np.inf)

    return pd.DataFrame(df)


def eval_colmap_model(
    model_path, gt_path, thrs=[1, 3, 5], return_df=False, AUC_col="max_error"
):
    """
    Given a scene path, evaluate the model in the given folder versus the ground truth in the gt_folder.
    Args:
        model_path: path to the model to evaluate.
        gt_path:    path to the ground truth model.
        thrs:       list of thresholds for the AUC computation.
        return_df:  if True, return the dataframe with the errors.
        AUC_col:    column to compute the AUC. Default is "max_error". Other options are "q_error" and "t_error".
    Returns:
        df_AUC: dataframe with the AUC values for the model.

    """

    if not os.path.exists(model_path):
        raise Exception(f"Path {model_path} does not exist.")

    if not os.path.exists(gt_path):
        raise Exception(f"Path {gt_path} does not exist.")

    # read models
    rec_input = pycolmap.Reconstruction(model_path)
    rec_gt = pycolmap.Reconstruction(gt_path)

    # evaluate scene (each pair of images) and compute the AUC
    df = evaluate_scene(rec_gt, rec_input)
    AUC_score_max = np.array(compute_AUC(df[AUC_col], thrs))

    if return_df:
        return AUC_score_max, df

    return AUC_score_max


def eval_colmap_model_all_scenes(
    input_path,
    gt_path,
    input_folder="colmap/sparse/0",
    gt_folder="sparse_gt",
    mapper="colmap",
    thrs=[0.5, 1, 3, 5, 10],
    AUC_col="max_error",
    n_jobs=-1,
) -> pd.DataFrame:
    """
    Evaluate the model on all the scenes in the data_path using parallel processing.
    These must be in COLMAP format. The model is evaluated at the specified thresholds.

    Args:
        scene_path (Path): Path to the directory containing the COLMAP models for each scene.
        gt_path (Path, optional): Path to the ground truth models for each scene. Defaults to "./sparse".
        thrs (List[int], optional): List of thresholds for AUC computation.
        AUC_col (str, optional): Column to compute the AUC from in the evaluation DataFrame.
                                 Defaults to "max_error". Other options are "q_error" and "t_error".
        n_jobs (int, optional): Number of parallel jobs to run. -1 means using all available CPU cores. Defaults to -1. # max=16

    Returns:
        pd.DataFrame: A DataFrame with the AUC values for each scene, indexed by scene name.
    """

    input_paths = sorted(glob.glob(f"{input_path}/*/{input_folder}"))
    target_paths = sorted(glob.glob(f"{gt_path}/*/{gt_folder}"))

    print(f"Found {len(input_paths)} scenes in {input_path}.")
    print(f"Found {len(target_paths)} scenes in {gt_path}.")

    # chek they have same number of scenes
    if len(input_paths) != len(target_paths):
        raise Exception(
            f"Number of scenes in {input_path} ({len(input_paths)}) and {gt_path} ({len(target_paths)}) do not match."
        )

    # check scene names match
    input_scene_names = [os.path.basename(os.path.dirname(p)) for p in input_paths]
    target_scene_names = [os.path.basename(os.path.dirname(p)) for p in target_paths]
    if input_scene_names != target_scene_names:
        raise Exception(f"Scene names in {input_path} and {gt_path} do not match.")

    s = time.time()
    # Use joblib to parallelize the evaluation of each scene
    results = Parallel(n_jobs=n_jobs)(
        delayed(eval_colmap_model)(
            input, target, thrs=thrs, return_df=False, AUC_col=AUC_col
        )
        for input, target in tqdm(
            zip(input_paths, target_paths),
            desc="Evaluating scenes",
            total=len(input_paths),
        )
    )
    print(results, input_scene_names)
    # Process results and create the DataFrame
    res = {}
    for auc_scores, scene_name in zip(results, input_scene_names):
        if auc_scores is not None:
            res[scene_name] = auc_scores

    # Creating the DataFrame and transposing it to have the scenes as rows
    df_res_colmap = pd.DataFrame(res, index=thrs).transpose()

    # Rename the columns as {model}@{thrs}
    df_res_colmap.columns = [f"{mapper}@{thr}" for thr in thrs]
    print(f"Evaluation completed in {time.time() - s:.2f} seconds.")
    return df_res_colmap.round(2)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-model",
        type=str,
        required=True,
        help="Path to the input 3D model file or directory.",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Path to the target 3D model file or directory.",
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
        "--thrs",
        type=float,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Thresholds for AUC computation.",
    )
    args = parser.parse_args()

    s = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    ### stuff

    print(f"Total time: {time.time() - s:.2f} seconds.")
