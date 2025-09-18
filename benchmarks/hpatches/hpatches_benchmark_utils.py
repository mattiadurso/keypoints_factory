import cv2
import json
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from typing import Tuple, Dict
from joblib import Parallel, delayed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hpatches_in_memory(base_path):
    """load the hpatches dataset in memory"""
    base_path = Path(base_path)

    if not base_path.exists():
        raise ValueError(f"hpatches folder does not exist: {base_path}")

    i_folders = sorted(list(base_path.glob("i_*")))
    v_folders = sorted(list(base_path.glob("v_*")))

    # ? order the folders such there is always one v and one i
    folders = {}
    while v_folders or i_folders:
        if v_folders:
            folders[v_folders.pop(0)] = "v"
        if i_folders:
            folders[i_folders.pop(0)] = "i"
    hpatches = {}
    for i, folder in enumerate(tqdm(folders, "Loading hpatches")):
        hpatches[folder.name] = {"imgs": [], "homs": []}
        img_paths = [Path(f"{k}.ppm") for k in range(1, 7)]
        hom_paths = [Path(f"H_1_{k}") for k in range(1, 7)]
        for j, (img_path, hom_path) in enumerate(zip(img_paths, hom_paths)):
            img = np.array(Image.open(folder / img_path))
            hom = np.loadtxt(folder / hom_path) if j != 0 else np.eye(3)

            hpatches[folder.name]["imgs"].append(img)
            hpatches[folder.name]["homs"].append(hom)
    return hpatches


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.array):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def two_vector_idxs_to_matching_matrix(
    matches: Tensor, n_kpts0: int, n_kpts1: int
) -> Tensor:
    """convert the input two vector idx notation to matching matrix
    Args:
        matches: the input matches in the two vector idx notation
            n,2
        n_kpts0: the number of keypoints in the first image
        n_kpts1: the number of keypoints in the second image
    output
        matching_matrix   n_kpts x n_kpts bool
    """
    matching_matrix = matches.new_zeros(
        n_kpts0, n_kpts1, dtype=th.bool, device=matches.device
    )
    matching_matrix[matches[:, 0], matches[:, 1]] = True
    return matching_matrix


def filter_outside(
    xy: Tensor, shape: tuple[int, int] | Tensor | np.ndarray, border: int = 0
) -> Tensor:
    """set as nan all the points that are not inside rectangle defined with shape HxW
    Args:
        xy: keypoints with coordinate (x, y)
            (B)xnx2
        shape: shape where the keypoints should be contained (H, W)
            2
        border: the minimum border to apply when masking
    Returns:
        Tensor: input keypoints with 'nan' where one of the two coordinates was not contained inside shape
        xy_filtered     (B)xnx2
    """
    assert xy.shape[-1] == 2, f"xy must have last dimension of size 2, got {xy.shape}"
    assert len(shape) == 2, f"shape must be a tuple of 2 elements, got {shape}"
    assert border < max(
        shape
    ), f"border must be smaller than the smallest shape dimension, got {border} and {shape}"

    xy = xy.clone()
    outside_mask = (
        (xy[..., 0] < border)
        + (xy[..., 0] >= shape[1] - border)
        + (xy[..., 1] < border)
        + (xy[..., 1] >= shape[0] - border)
    )  # (B)xn
    xy[outside_mask] = float("nan")
    return xy


def warp_points(
    xy: Tensor,
    H: Tensor,
    img_shape: tuple[float, float] | Tensor | np.ndarray | None = None,
    border: int = 0,
) -> Tensor:
    """warp the points using the provided homography matrix
    Args:
        xy: input points coordinates with order (x, y)
           B,n,2
        H: input homography
           B,3,3
        img_shape: if provided, the points that project out of shape are set to 'nan'
        border: together with img_shape, set to nan the points that are closer to the border than this
    Returns:
        Tensor: the projected points
           B,n,2
    """
    assert (
        xy.shape[0] == H.shape[0]
    ), f"xy.shape[0] = {xy.shape[0]} != H.shape[0] = {H.shape[0]}"
    assert xy.shape[2] == 2, f"xy must be B,n,2 but is {xy.shape}"
    assert H.shape[1] == 3 and H.shape[2] == 3, f"H must be B,3,3 but is {H.shape}"
    if img_shape is not None:
        assert len(img_shape) == 2

    # xy_hom = geom.convert_points_to_homogeneous(xy.to(H.dtype))  # B,n,3
    xy_hom = torch.cat(
        (
            xy,
            torch.ones((xy.shape[0], xy.shape[1], 1), dtype=xy.dtype, device=xy.device),
        ),
        dim=2,
    ).to(
        H.dtype
    )  # B,n,3
    xy_proj_hom = xy_hom @ H.to(xy.device).permute(0, 2, 1)  # B,n,3
    # xy_proj = geom.convert_points_from_homogeneous(xy_proj_hom)   # B
    xy_proj = xy_proj_hom[:, :, 0:2] / xy_proj_hom[:, :, 2:3]  # B,n,2

    if img_shape is not None:
        xy_proj = filter_outside(xy_proj, img_shape, border)

    xy_proj = xy_proj.to(xy.dtype)

    return xy_proj


def compute_homography_corner_error(
    H0_1_GT: Tensor,
    H0_1_estimated: Tensor,
    img0_shape: tuple[int, int] | Tensor,
    img1_shape: tuple[int, int] | Tensor | None = None,
) -> Tensor:
    """compute the corner error in pixel between the GT homography and the estimated one projecting the corners of
        img0 and img1 (if provided)
    Args:
        H0_1_GT: the GT homography warping between img0 and img1
            Bx3x3
        H0_1_estimated: the estimated homography between img0 and img1
            Bx3x3
        img0_shape: the shape of img0 used to project the four corners
            (H, W)
        img1_shape: the shape of img1 used to project the four corners, if not provided only one direction is used
            (H, W)
    Returns:
        corner_error: the computed (symmetric if img1_shape is provided) corner error
            B
    """
    assert H0_1_GT.dim() == 3
    assert H0_1_estimated.dim() == 3
    assert H0_1_GT.shape == H0_1_estimated.shape
    assert len(img0_shape) == 2
    assert img1_shape is None or len(img1_shape) == 2
    B = H0_1_estimated.shape[0]
    device = H0_1_GT.device

    corners0 = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, img0_shape[0]],
            [img0_shape[1], img0_shape[0]],
            [img0_shape[1], 0.0],
        ],
        device=device,
    )[None, :, :].repeat(
        B, 1, 1
    )  # B,4,2

    # ? compute the corner error in the projected frame
    projected_corners0_GT = warp_points(corners0, H0_1_GT.to(th.double))  # B,4,2
    projected_corners0 = warp_points(corners0, H0_1_estimated.to(th.double))  # B,4,2
    corner_error = torch.norm(projected_corners0_GT - projected_corners0, dim=2).mean(
        1
    )  # B
    # # ? compute the corner error in the img0 frame
    # projected_corners0_GT = warp_points(corners0, H0_1_GT.to(th.double))  # B,4,2
    # unprojected_corners0 = warp_points(projected_corners0_GT, torch.inverse(H0_1_estimated.to(th.double)))  # B,4,2
    # corner_error = torch.norm(corners0 - unprojected_corners0, dim=2).mean(1)  # B

    if img1_shape is not None:
        corners1 = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, img1_shape[0]],
                [img1_shape[1], img1_shape[0]],
                [img1_shape[1], 0.0],
            ],
            device=device,
        )[None, :, :].repeat(
            B, 1, 1
        )  # B,4,2
        # ? compute the corner error in the projected frame
        projected_corners1_GT = warp_points(
            corners1, torch.inverse(H0_1_GT.to(th.double))
        )  # B,4,2
        projected_corners1 = warp_points(
            corners1, torch.inverse(H0_1_estimated.to(th.double))
        )  # B,4,2
        corner_error_1 = torch.norm(
            projected_corners1_GT - projected_corners1, dim=2
        ).mean(
            1
        )  # B
        # # ? compute the corner error in the img1 frame
        # projected_corners1_GT = warp_points(corners1, torch.inverse(H0_1_GT.to(th.double)))  # B,4,2
        # unprojected_corners1 = warp_points(projected_corners1_GT, H0_1_estimated.to(th.double))  # B,4,2
        # corner_error_1 = torch.norm(corners1 - unprojected_corners1, dim=2).mean(1)  # B

        corner_error = 0.5 * (corner_error + corner_error_1)  # B

    # # ? remove all the values smaller than a threshold
    # corner_error = torch.div(corner_error, 1e-3, rounding_mode='floor') * 1e-3

    return corner_error


def compute_corner_error(
    xy0_matched: Tensor,
    xy1_matched: Tensor,
    H0_1: Tensor,
    img0_shape: tuple[int, int] | Tensor,
    img1_shape: tuple[int, int] | Tensor,
    mode: str,
    ransac_homography_threshold: list[float] = None,
    ransac_max_iters: int = 10_000,
) -> list[dict]:
    assert xy0_matched.shape == xy1_matched.shape
    assert xy0_matched.shape[1] == 2
    assert len(img0_shape) == 2
    assert len(img1_shape) == 2
    assert H0_1.shape == (3, 3)

    if ransac_homography_threshold is None:
        # ransac_homography_threshold = [
        #     0.125, 0.25, 0.5, 0.75,
        #     1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        #     12.0, 15.0, 20.0, 30.0, 50.0
        # ]
        ransac_homography_threshold = [3.0]

    device = xy0_matched.device

    def estimate_homography(
        _xy0_matched: np.ndarray, _xy1_matched: np.ndarray, _thr: float
    ) -> tuple[np.ndarray, np.ndarray]:
        _H0_1_estimated, mask = cv2.findHomography(
            _xy0_matched,
            _xy1_matched,
            method=cv2.RANSAC,
            # method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=_thr,
            maxIters=ransac_max_iters,
            confidence=0.99999999,
        )

        # H0_1_estimated, mask = pydegensac.findHomography(
        #     _xy0_matched,
        #     _xy1_matched,
        #     thr,
        #     conf=0.99999999,
        #     max_iters=ransac_max_iters
        # )
        return _H0_1_estimated, mask

    homography_stats_list = []
    if xy0_matched.shape[0] < 4:
        for ransac_thr in ransac_homography_threshold:
            homography_stats_dict = {
                "valid_homography": False,
                "ransac_thr": ransac_thr,
                "corner_error": float("inf"),
                "homography": torch.ones((3, 3), device=device) * float("nan"),
                "mask": torch.zeros(xy0_matched.shape[0], dtype=th.bool, device=device),
                "n_matched_keypoints": xy0_matched.shape[0],
                "n_ransac_inliers": 0,
            }
            homography_stats_list.append(homography_stats_dict)
        return homography_stats_list

    n_parallel_ransac = 16
    xy0_matched_list = [xy0_matched.cpu().numpy() for _ in range(n_parallel_ransac)]
    xy1_matched_list = [xy1_matched.cpu().numpy() for _ in range(n_parallel_ransac)]
    for ransac_thr in ransac_homography_threshold:
        homography_stats_dict = {
            "valid_homography": False,
            "ransac_thr": ransac_thr,
            "corner_error": float("inf"),
            "homography": torch.ones((3, 3), device=device) * float("nan"),
            "mask": torch.zeros(xy0_matched.shape[0], dtype=th.bool, device=device),
            "n_ransac_inliers": 0,
        }

        output = Parallel(n_jobs=n_parallel_ransac)(
            delayed(estimate_homography)(xy0_matched, xy1_matched, ransac_thr)
            for xy0_matched, xy1_matched in zip(xy0_matched_list, xy1_matched_list)
        )
        H0_1_estimated_list, mask_list = zip(
            *output
        )  # (n_parallel_ransac) (n_parallel_ransac)
        # ? get only the homographies that could be recovered
        H0_1_valid_list = [
            H0_1 for H0_1 in H0_1_estimated_list if H0_1 is not None
        ]  # n_valid
        if H0_1_valid_list:
            mask_list: list[Tensor] = [
                torch.tensor(mask_list[i], dtype=th.bool, device=device)
                for i in range(len(mask_list))
                if H0_1_estimated_list[i] is not None
            ]  # list of n_valid Tensors of different shapes
            H0_1_estimated: Tensor = torch.tensor(
                np.array(H0_1_estimated_list), dtype=th.float, device=device
            )  # n_valid,3,3
            if mode == "symmetric":
                error = compute_homography_corner_error(
                    H0_1[None].to(device).repeat(H0_1_estimated.shape[0], 1, 1),
                    H0_1_estimated,
                    img0_shape,
                    img1_shape,
                )  # n_valid
            elif mode == "hpatches":
                error = compute_homography_corner_error(
                    H0_1[None].to(device).repeat(H0_1_estimated.shape[0], 1, 1),
                    H0_1_estimated,
                    img0_shape,
                )  # n_valid
            else:
                raise ValueError(f"Unknown mode {mode}")
            best_error, best_error_idx = error.min(0)
            homography_stats_dict = {
                "valid_homography": True,
                "ransac_thr": ransac_thr,
                "corner_error": best_error.item(),
                "homography": H0_1_estimated[best_error_idx].cpu().numpy(),
                "mask": mask_list[best_error_idx].flatten().cpu().numpy(),
                "n_ransac_inliers": mask_list[best_error_idx].flatten().sum().item(),
            }

        homography_stats_list.append(homography_stats_dict)

    return homography_stats_list


def find_distance_matrices_between_points_and_their_projections(
    xy0: Tensor, xy1: Tensor, xy0_proj: Tensor, xy1_proj: Tensor
) -> (Tensor, Tensor):
    """find the mutual nearest neighbors between two sets of keypoints and their projections
    Args:
        xy0: first set of keypoints
            n0,2
        xy1: second set of keypoints
            n1,2
        xy0_proj: first set of projected keypoints
            n0_proj,2
        xy1_proj: second set of projected keypoints
            n1_proj,2
    Returns:
        dist0: distance matrix between xy0 and xy1_proj
            n0,n1
        dist1: distance matrix between xy0_proj and xy1
            n0,n1
    """

    assert (
        xy0.ndim == 2 and xy1.ndim == 2
    ), f"xy0 and xy1 must be 2D tensors, got {xy0.ndim} and {xy1.ndim}"
    assert (
        xy0_proj.ndim == 2 and xy1_proj.ndim == 2
    ), f"xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.ndim} and {xy1_proj.ndim}"

    # ? compute the distance between all the reprojected points
    # # ? low memory usage, slow but correct
    # dist0 = torch.cdist(xy0.to(th.float), xy1_proj,         compute_mode='donot_use_mm_for_euclid_dist')  # n0,n1
    # dist1 = torch.cdist(xy0_proj,         xy1.to(th.float), compute_mode='donot_use_mm_for_euclid_dist')  # n0,n1
    # ? high memory usage, fast and correct
    dist0 = (xy0[:, None, :] - xy1_proj[None, :, :]).norm(dim=2)  # n0,n1
    dist1 = (xy0_proj[:, None, :] - xy1[None, :, :]).norm(dim=2)  # n0,n1
    # # ? low memory usage, fast but non-deterministic
    # dist0 = torch.cdist(xy0.to(th.float), xy1_proj)  # n0,n1
    # dist1 = torch.cdist(xy0_proj,         xy1.to(th.float))  # n0,n1
    dist0[dist0.isnan()] = float("+inf")
    dist1[dist1.isnan()] = float("+inf")
    return dist0, dist1


def find_mutual_nearest_neighbors_from_keypoints_and_their_projections(
    xy0: Tensor,
    xy1: Tensor,
    xy0_proj: Tensor,
    xy1_proj: Tensor,
    dist0: Tensor | None = None,
    dist1: Tensor | None = None,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
    """find the mutual nearest neighbors between two sets of keypoints and their projections
    Args:
        xy0: first set of keypoints
            n0,2
        xy1: second set of keypoints
            n1,2
        xy0_proj: first set of projected keypoints
            n0_proj,2
        xy1_proj: second set of projected keypoints
            n1_proj,2
        dist0: optional distance matrix between xy0 and xy1_proj, if not provided it will be computed
            n0,n1
        dist1: optional distance matrix between xy0_proj and xy1, if not provided it will be computed
            n0,n1
    Returns:
        mnn_mask: binary mask of mutual nearest neighbors
            n0,n1
        xy0_closest_dist_mnn: for each xy0 that has a mutual nearest neighbor, the distance to the closest xy1_proj in img0
            n0_mnn
        xy1_closest_dist_mnn: for each xy1 that has a mutual nearest neighbor, the distance to the closest xy0_proj in img1
            n1_mnn
        xy0_closest_dist: for each xy0, the distance to the closest xy1_proj in img0
            n0
        xy1_closest_dist: for each xy1, the distance to the closest xy0_proj in img1
            n1
    """
    assert (
        xy0.ndim == 2 and xy1.ndim == 2
    ), f"xy0 and xy1 must be 2D tensors, got {xy0.ndim} and {xy1.ndim}"
    assert (
        xy0_proj.ndim == 2 and xy1_proj.ndim == 2
    ), f"xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.ndim} and {xy1_proj.ndim}"
    if dist0 is not None:
        assert dist0.shape == (
            xy0.shape[0],
            xy1_proj.shape[0],
        ), f"dist0 must be a matrix of shape ({xy0.shape[0]}, {xy1.shape[0]}), got {dist0.shape}"
    if dist1 is not None:
        assert dist1.shape == (
            xy0.shape[0],
            xy1_proj.shape[0],
        ), f"dist1 must be a matrix of shape ({xy0.shape[0]}, {xy1.shape[0]}), got {dist1.shape}"

    device = xy0.device

    n0 = xy0.shape[0]
    n1 = xy1.shape[0]

    if dist0 is None and dist1 is None:
        dist0, dist1 = find_distance_matrices_between_points_and_their_projections(
            xy0, xy1, xy0_proj, xy1_proj
        )

    if n1 > 0:
        # ? find the closest point in the image between each xy0 and xy1_proj
        xy0_closest_dist, closest0 = dist0.min(1)
    else:
        xy0_closest_dist, closest0 = torch.zeros((0,), device=device), torch.zeros(
            (0,), dtype=th.long, device=device
        )  # n0
    if n0 > 0:
        # ? find the closest point in the image between each xy1 and xy0_proj
        xy1_closest_dist, closest1 = dist1.min(0)
    else:
        xy1_closest_dist, closest1 = torch.zeros((0,), device=device), torch.zeros(
            (0,), dtype=torch.long, device=device
        )  # n1

    xy0_closest_matrix = torch.zeros(dist0.shape, dtype=torch.bool, device=device)
    xy1_closest_matrix = torch.zeros(dist0.shape, dtype=torch.bool, device=device)
    if n1 > 0:
        xy0_closest_matrix[th.arange(len(xy0)), closest0] = True
    if n0 > 0:
        xy1_closest_matrix[closest1, torch.arange(len(xy1))] = True
    # ? fink the keypoints that are mutual nearest neighbors (using only x,y coordinates) in both images
    mnn_mask = xy0_closest_matrix & xy1_closest_matrix
    mnn_idx = mnn_mask.nonzero()
    xy0_closest_dist_mnn = torch.ones_like(xy0_closest_dist) * float("inf")
    xy1_closest_dist_mnn = torch.ones_like(xy1_closest_dist) * float("inf")
    xy0_closest_dist_mnn[mnn_idx[:, 0]] = xy0_closest_dist[mnn_idx[:, 0]]
    xy1_closest_dist_mnn[mnn_idx[:, 1]] = xy1_closest_dist[mnn_idx[:, 1]]
    return (
        mnn_mask,
        xy0_closest_dist_mnn,
        xy1_closest_dist_mnn,
        xy0_closest_dist,
        xy1_closest_dist,
    )


def compute_coverages(
    xy0: Tensor,
    xy1: Tensor,
    xy0_proj: Tensor,
    xy1_proj: Tensor,
    img0_shape: Tensor,
    img1_shape: Tensor,
    px_thrs: float | list[float],
    coverage_kernel_size: int,
) -> tuple[float, float] | tuple[Tensor, Tensor]:
    assert (
        xy0.ndim == 2 and xy1.ndim == 2
    ), f"xy0 and xy1 must be 2D tensors, got {xy0.shape} and {xy1.shape}"
    assert (
        xy0_proj.ndim == 2 and xy1_proj.ndim == 2
    ), f"xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.shape} and {xy1_proj.shape}"
    device = xy0.device

    # ? number of keypoints that are in the overlap area
    n_xy0_overlap = int((~th.isnan(xy0_proj[:, 0])).sum())
    n_xy1_overlap = int((~th.isnan(xy1_proj[:, 0])).sum())

    dist0, dist1 = find_distance_matrices_between_points_and_their_projections(
        xy0, xy1, xy0_proj, xy1_proj
    )

    if not isinstance(px_thrs, list):
        px_thrs = [px_thrs]

    coverage = torch.zeros(len(px_thrs), device=device)
    coverage_per_kpt = torch.zeros(len(px_thrs), device=device)
    for i, px_thr in enumerate(px_thrs):
        xy0_inlier = xy0[(dist1 <= px_thr).any(1)]
        xy1_inlier = xy1[(dist0 <= px_thr).any(0)]
        ij0 = torch.flip(xy0_inlier.to(th.long), [-1])
        ij1 = torch.flip(xy1_inlier.to(th.long), [-1])
        img0 = torch.zeros(img0_shape, device=device)
        img1 = torch.zeros(img1_shape, device=device)
        img0[ij0[:, 0], ij0[:, 1]] = 1
        img1[ij1[:, 0], ij1[:, 1]] = 1
        img0 = torch.max_pool2d(
            img0[None, None, ...],
            (coverage_kernel_size, coverage_kernel_size),
            stride=1,
            padding=coverage_kernel_size // 2,
        )[0, 0, ...]
        img1 = torch.max_pool2d(
            img1[None, None, ...],
            (coverage_kernel_size, coverage_kernel_size),
            stride=1,
            padding=coverage_kernel_size // 2,
        )[0, 0, ...]
        coverage[i] = 0.5 * (img0.mean() + img1.mean())

        if n_xy0_overlap > 0 and n_xy1_overlap > 0:
            coverage_per_kpt[i] = (
                1000 * coverage[i] / (0.5 * (n_xy0_overlap + n_xy1_overlap))
            )
        else:
            coverage_per_kpt[i] = 0.0

    coverage = coverage[0].item() if coverage.shape[0] == 1 else coverage
    coverage_per_kpt = (
        coverage_per_kpt[0].item()
        if coverage_per_kpt.shape[0] == 1
        else coverage_per_kpt
    )

    return coverage, coverage_per_kpt


def compute_matching_stats_homography(
    xy0: Tensor,
    xy1: Tensor,
    H0_1: Tensor,
    img0_shape: np.ndarray | Tensor,
    img1_shape: np.ndarray | Tensor,
    mode: str,
    matches: Tensor = None,
    px_thrs: float | list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    compute_coverage: bool = True,
    coverage_kernel_size: int = 9,
    evaluate_corner_error: bool = True,
    evaluate_corner_error_keypoints: bool = False,
    device: str = "cpu",
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]]]:
    """compute multiple matching statistics at multiple threshold
        - the following three metrics are computed without generating the GT matches
         (and can be computed symmetrically or not)
            matching_accuracy
                n_matched_with_max(dist0,dist1)_lower_then_threshold / n_matched_proposed
            matching_score
                0.5*good_matches/n_keypoints_overlap_img0 + 0.5*good_matches/n_keypoints_overlap_img1
            repeatability
                (n_close_kpts0_proj + n_close_kpts1_proj) / (n_kpts0_proj_overlap + n_kpts1_proj_overlap)

        - the following metrics are computed generating the GT matches are mutual nn (GT always computed symmetrically)
            precision
                n_matches_inlier_nn / n_matches_proposed
            recall
                n_matches_inlier_nn / n_matches_GT

        - additionally the corner error is computed as the distance between the corners of img0 projected in img1 using
            the GT homography and projected back using the estimated one (with ransac and threshold depending on the
            threshold used for the metrics) and the original img0 corners
            The corner error is NaN if the number of matches < 4
    Args:
        xy0: coordinates of the keypoints in img0
            n0 x 2  (x, y)
        xy1: coordinates of the keypoints in img1
            n1 x 2  (x, y)
        H0_1: homography that warp img0 in img1
            3 x 3
        img0_shape: the shape of img0
            (height, width)
        img1_shape: the shape of img1
            (height, width)
        mode: 'symmetric' 'oneway' or 'hpatches' how to compute
            [matching_accuracy, matching_score and repeatability]
            symmetric -> compute everything back and forward
            oneway -> compute everything only forward
            hpatches -> mma forward and the rest symmetric
        matches: index of xy0 and xy1 that were matched together
            nm x 2  (xy0_idx, xy1_idx)
        px_thrs: the threshold at which to compute the metrics
        compute_coverage: if True, compute the coverage
        coverage_kernel_size: if provided, the kernel size for the computation of the coverage
        evaluate_corner_error: if True, compute the corner error with different ransac thresholds
        evaluate_corner_error_keypoints: if True, compute the corner error with different ransac thresholds using only the keypoints (matches does not come from descriptors)
        device: the device where to compute the metrics
    Returns:
        list of dict with keys:
        {
            'thr'
            'matching_accuracy'
            'matching_score'
            'repeatability'
            'repeatability_mnn'
            'precision'
            'recall'
            'n_matches_inliers'
            'n_keypoints'
            [Optional] 'coverage'
            [Optional] 'coverage_per_kp'
        }

        list of dict with keys:  (empty list if evaluate_corner_error is False)
        {
            'valid_homography':
            'ransac_thr':
            'corner_error':
            'homography':
            'mask':
        }

        list of dict with keys:  (empty list if evaluate_corner_error_keypoints is False)
            'valid_homography_keypoints':
            'ransac_thr_keypoints':
            'corner_error_keypoints':
            'homography_keypoints':
            'mask_keypoints':
        }
    Raises:
        None
    """
    assert mode in ["symmetric", "hpatches"]
    assert (
        len(img0_shape) == 2 and len(img1_shape) == 2
    ), "img shape must be (height, width)"

    xy0 = xy0.to(device)
    xy1 = xy1.to(device)
    H0_1 = H0_1.to(device)
    try:
        matches = (
            matches.to(device)
            if matches is not None
            else torch.zeros((0, 2), dtype=th.long, device=device)
        )
    except:
        matches = (
            matches[0].to(device)
            if matches is not None
            else torch.zeros((0, 2), dtype=th.long, device=device)
        )

    if not isinstance(px_thrs, list):
        px_thrs = [px_thrs]

    n0 = xy0.shape[0]
    n1 = xy1.shape[0]
    n_matches_proposed = matches.shape[0]

    # ? project the points from one image to the other, nan if a keypoints project out of the image
    xy0_proj = warp_points(xy0[None], H0_1[None], img1_shape)[0].to(th.float)  # n0 x 2
    xy1_proj = warp_points(xy1[None], torch.inverse(H0_1)[None], img0_shape)[0].to(
        torch.float
    )  # n1 x 2

    # ? number of keypoints that are in the overlap area
    n_xy0_overlap = (~xy0_proj.isnan().any(-1)).sum().item()
    n_xy1_overlap = (~xy1_proj.isnan().any(-1)).sum().item()

    dist0, dist1 = find_distance_matrices_between_points_and_their_projections(
        xy0, xy1, xy0_proj, xy1_proj
    )

    dist_max = torch.max(dist0, dist1)
    mnn_mask, xy0_dist_mnn, xy1_dist_mnn, xy0_dist, xy1_dist = (
        find_mutual_nearest_neighbors_from_keypoints_and_their_projections(
            xy0, xy1, xy0_proj, xy1_proj, dist0, dist1
        )
    )

    # ? matched keypoints
    xy0_matched = xy0[matches[:, 0]]  # nm x 2
    xy0_proj_matched = xy0_proj[matches[:, 0]]  # nm x 2
    xy1_matched = xy1[matches[:, 1]]  # nm x 2
    xy1_proj_matched = xy1_proj[matches[:, 1]]  # nm x 2

    # ? distance between the matched keypoints only
    dist_matched0 = torch.linalg.norm(xy0_matched - xy1_proj_matched, dim=1)  # nm
    dist_matched1 = torch.linalg.norm(xy0_proj_matched - xy1_matched, dim=1)  # nm
    dist_matched0[dist_matched0.isnan()] = float("+inf")
    dist_matched1[dist_matched1.isnan()] = float("+inf")
    dist_matched_max = torch.max(dist_matched0, dist_matched1)  # nm

    matching_matrix_proposed = two_vector_idxs_to_matching_matrix(matches, n0, n1)

    stats_list = []
    for px_thr in px_thrs:
        stats_dict = {
            "thr": float(px_thr),
            "n_keypoints": (n0 + n1) / 2,
            "repeatability": 0.0,
            "repeatability_mnn": 0.0,
            "n_matches_proposed": n_matches_proposed,
            "matching_score": 0.0,
            "matching_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_inliers_nn_GT": 0,
        }

        n_xy0_close = (xy0_dist <= px_thr).sum()
        n_xy1_close = (xy1_dist <= px_thr).sum()

        if mode == "symmetric":
            n_matches_inlier = int((dist_matched_max <= px_thr).sum().item())
        else:
            n_matches_inlier = int((dist_matched1 <= px_thr).sum().item())

        if n_xy0_overlap > 0 and n_xy1_overlap > 0:
            stats_dict["repeatability"] = (
                (n_xy0_close + n_xy1_close) / (n_xy0_overlap + n_xy1_overlap)
            ).item()
            stats_dict["repeatability_mnn"] = (
                ((xy0_dist_mnn <= px_thr).sum() + (xy1_dist_mnn <= px_thr).sum())
                / (n_xy0_overlap + n_xy1_overlap)
            ).item()
            stats_dict["matching_score"] = (
                0.5 * n_matches_inlier * (1 / n_xy0_overlap + 1 / n_xy1_overlap)
            )
        else:
            print("THERE WERE 0 n_kpts0_overlap or n_kpts1_overlap")

        # ? the mutual nn is used to compute precision and accuracy
        matching_matrix_GT = mnn_mask * (dist_max <= px_thr)
        # ? the inliers here are computed in a slightly different way, where a GT match needs to be mutual nn
        n_inliers_nn = (matching_matrix_proposed * matching_matrix_GT).sum()
        n_matches_GT = matching_matrix_GT.sum()
        stats_dict["n_inliers_nn_GT"] = n_inliers_nn.item()

        if n_matches_GT > 0:
            stats_dict["recall"] = (n_inliers_nn / n_matches_GT).item()

        if n_matches_proposed > 0:
            stats_dict["matching_accuracy"] = n_matches_inlier / n_matches_proposed
            stats_dict["precision"] = (n_inliers_nn / n_matches_proposed).item()
        # else:
        #     print('NO MATCHES PROPOSED')

        stats_list.append(stats_dict)

        # ? compute the coverage
        if compute_coverage:
            stats_dict["coverage"], stats_dict["coverage_per_kpt"] = compute_coverages(
                xy0,
                xy1,
                xy0_proj,
                xy1_proj,
                img0_shape=img0_shape,
                img1_shape=img1_shape,
                px_thrs=px_thr,
                coverage_kernel_size=coverage_kernel_size,
            )
            stats_dict["coverage_kernel"] = coverage_kernel_size

    # ? compute AUC values
    if evaluate_corner_error:
        homography_stats_list = compute_corner_error(
            xy0_matched, xy1_matched, H0_1, img0_shape, img1_shape, mode
        )
    else:
        homography_stats_list = []

    if evaluate_corner_error_keypoints:
        mnn_idx = mnn_mask.nonzero()
        xy0_matched_mnn = xy0[mnn_idx[:, 0]]  # n_mnn,2
        xy1_matched_mnn = xy1[mnn_idx[:, 1]]  # n_mnn,2
        dist_max_mnn = dist_max[mnn_mask]  # n_mnn

        homography_stats_list_keypoints = []
        for px_thr in px_thrs:
            distance_mask = dist_max_mnn <= px_thr
            xy0_matched_kpts_thr = xy0_matched_mnn[distance_mask]
            xy1_matched_kpts_thr = xy1_matched_mnn[distance_mask]
            homography_stats_at_thr = compute_corner_error(
                xy0_matched_kpts_thr,
                xy1_matched_kpts_thr,
                H0_1,
                img0_shape,
                img1_shape,
                mode,
            )
            for stats in homography_stats_at_thr:
                stats["thr_mnn_keypoints"] = px_thr
            homography_stats_list_keypoints += homography_stats_at_thr
    else:
        homography_stats_list_keypoints = []

    return stats_list, homography_stats_list, homography_stats_list_keypoints


def compute_matching_stats_sequential(
    keypoints: Dict,
    matches: Dict,
    hpatches: Dict,
    max_kpts: int = 999999,
    px_thrs: float | list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Original sequential implementation - renamed for clarity"""
    stats_df = pd.DataFrame()
    stats_homography_df = pd.DataFrame()
    for folder_name in tqdm(hpatches, "Computing matching stats"):
        homographies = hpatches[folder_name]["homs"]
        images = hpatches[folder_name]["imgs"]
        kpts = keypoints[folder_name]
        # select only the top max_kpts
        kpts = [k[: min(k.shape[0], max_kpts)] for k in kpts]

        for i in range(1, 6):
            matches_single_pair = matches[folder_name][f"1_{i+1}"]
            list_of_stats_dict, list_of_homography_stats_dict, _ = (
                compute_matching_stats_homography(
                    kpts[0],
                    kpts[i],
                    torch.from_numpy(homographies[i]),
                    images[0].shape[:2],
                    images[i].shape[:2],
                    mode="hpatches",
                    matches=matches_single_pair,
                    px_thrs=px_thrs,  # Pass through px_thrs parameter
                    evaluate_corner_error_keypoints=False,
                )
            )
            stats_single_pair_df = pd.DataFrame(list_of_stats_dict)
            # add additional information
            stats_single_pair_df["type"] = folder_name[0]
            stats_single_pair_df["scene"] = folder_name
            stats_single_pair_df["pair"] = f"1-{i+1}"
            stats_single_pair_df["n_keypoints"] = 0.5 * (
                kpts[0].shape[0] + kpts[i].shape[0]
            )

            stats_homography_single_pair_df = pd.DataFrame(
                list_of_homography_stats_dict
            )
            # add additional information
            stats_homography_single_pair_df["type"] = folder_name[0]
            stats_homography_single_pair_df["scene"] = folder_name
            stats_homography_single_pair_df["pair"] = f"1-{i+1}"
            stats_homography_single_pair_df["n_keypoints"] = 0.5 * (
                kpts[0].shape[0] + kpts[i].shape[0]
            )

            stats_df = pd.concat([stats_df, stats_single_pair_df], axis=0)
            stats_homography_df = pd.concat(
                [stats_homography_df, stats_homography_single_pair_df], axis=0
            )

    # duplicate the dataframe and call the type as overall
    stats_df_overall = stats_df.copy()
    stats_df_overall["type"] = "overall"
    stats_df = pd.concat([stats_df, stats_df_overall])

    # mean statistics for the 3 classes (v, i, overall)
    metric_names = [
        "repeatability",
        "repeatability_mnn",
        "n_keypoints",
        "coverage",
        "coverage_per_kpt",
        "matching_accuracy",
        "matching_score",
        "precision",
        "recall",
        "n_matches_proposed",
        "n_inliers_nn_GT",
    ]
    mean_renaming_dict = {name: f"mean_{name}" for name in metric_names}
    median_renaming_dict = {name: f"median_{name}" for name in metric_names}

    numeric_cols = stats_df.select_dtypes(include="number").columns
    mean_stats_df = stats_df.groupby(["thr", "type"], as_index=False)[
        numeric_cols
    ].mean()
    mean_stats_df.rename(columns=mean_renaming_dict, inplace=True)
    numeric_cols = stats_df.select_dtypes(include="number").columns
    median_stats_df = stats_df.groupby(["thr", "type"], as_index=False)[
        numeric_cols
    ].median()
    median_stats_df.rename(columns=median_renaming_dict, inplace=True)
    aggregated_df = pd.merge(mean_stats_df, median_stats_df, "outer")

    # compute the homography accuracy@x
    # duplicate the dataframe and call the type as overall
    stats_homography_df_overall = stats_homography_df.copy()
    stats_homography_df_overall["type"] = "overall"
    stats_homography_df = pd.concat([stats_homography_df, stats_homography_df_overall])

    aggregated_homography_accuracy_df = pd.DataFrame()
    HOMOGRAPHY_ACCURACY_THRS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for homography_accuracy_thr in HOMOGRAPHY_ACCURACY_THRS:
        temp_df = stats_homography_df.copy()
        temp_df["homography_accuracy"] = (
            temp_df["corner_error"] <= homography_accuracy_thr
        )
        stats_homography_at_thr_df = (
            temp_df.groupby(["ransac_thr", "type"])["homography_accuracy"]
            .mean()
            .reset_index()
        )
        stats_homography_at_thr_df["accuracy_thr"] = homography_accuracy_thr
        aggregated_homography_accuracy_df = pd.concat(
            [aggregated_homography_accuracy_df, stats_homography_at_thr_df]
        )

    return (
        stats_df,
        aggregated_df,
        stats_homography_df,
        aggregated_homography_accuracy_df,
    )


# Keep the original function for backwards compatibility
def compute_matching_stats(
    keypoints: Dict,
    matches: Dict,
    hpatches: Dict,
    max_kpts: int = 999999,
    n_jobs: int = 1,  # Default to sequential for backwards compatibility
    px_thrs: float | list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute matching statistics with optional parallelization.
    This is now a wrapper that calls either the parallel or original implementation.
    """
    if n_jobs == 1:
        # Use original sequential implementation
        return compute_matching_stats_sequential(
            keypoints, matches, hpatches, max_kpts, px_thrs=px_thrs
        )
    else:
        logger.info(f"Multi-processing not enabled, using single-threaded execution.")
        return compute_matching_stats_sequential(
            keypoints, matches, hpatches, max_kpts, px_thrs=px_thrs
        )


def display_hpatches_results(
    results_file="hpatches/results/results.json",
    partition="overall",
    method=None,
    ths=None,  # Allow None as default
    tostring=True,
):
    """
    Display HPatches benchmark results in a formatted DataFrame.

    Args:
        results_file: Path to the results JSON file
        partition: Which partition to display ('overall', 'i', 'v')
        method: Specific method to display (if None, displays all methods)
        ths: List of thresholds to display (if None, auto-detect from results)
        tostring: If True, print the DataFrame as a string

    Returns:
        pandas.DataFrame: Formatted results table
    """
    # Load results
    if isinstance(results_file, str):
        results_file = Path(results_file)

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    if not data:
        print("No results found in the file.")
        return pd.DataFrame()

    # Filter by specific method if provided
    if method is not None:
        if method in data:
            data = {method: data[method]}
        else:
            print(f"Method '{method}' not found in results.")
            return pd.DataFrame()

    # Auto-detect thresholds if not provided
    if ths is None:
        ths = []
        for method_key, results in data.items():
            if partition in results:
                partition_data = results[partition]
                for key in partition_data.keys():
                    if key.startswith("repeatability_"):
                        thr = key.split("_")[-1]
                        try:
                            thr_val = int(float(thr))
                            if thr_val not in ths:
                                ths.append(thr_val)
                        except ValueError:
                            continue
        ths = sorted(ths) if ths else [1, 2, 3]  # Default fallback

    # Create DataFrame
    rows = []

    for method_key, results in data.items():
        if partition not in results:
            continue

        partition_data = results[partition]

        # Base row info
        if "+" in method_key:
            method_name, custom_desc_part = method_key.split("+", 1)
            custom_desc = custom_desc_part.split("_")[0]
        else:
            method_name = method_key.split("_")[0]
            custom_desc = ""

        row = {
            "Method": method_name,
            "Custom Desc": custom_desc,
        }

        # Add metrics for all thresholds - GROUPED BY METRIC TYPE
        for thr in ths:
            # Repeatability
            rep_key = f"repeatability_{thr}"
            if rep_key in partition_data:
                rep_data = partition_data[rep_key]
                row[f"Rep@{thr}"] = f"{rep_data.get('mean', 0.0)*100:.1f}"

        for thr in ths:
            # Matching Accuracy
            ma_key = f"matching_accuracy_{thr}"
            if ma_key in partition_data:
                ma_data = partition_data[ma_key]
                row[f"MA@{thr}"] = f"{ma_data.get('mean', 0.0)*100:.1f}"

        for thr in ths:
            # Matching Score
            ms_key = f"matching_score_{thr}"
            if ms_key in partition_data:
                ms_data = partition_data[ms_key]
                row[f"MS@{thr}"] = f"{ms_data.get('mean', 0.0)*100:.1f}"

        for thr in ths:
            # Homography Accuracy
            ha_key = f"homography_accuracy_{thr}"
            if ha_key in partition_data:
                ha_data = partition_data[ha_key]
                row[f"HA@{thr}"] = f"{ha_data.get('mean', 0.0)*100:.1f}"

        rows.append(row)

    if not rows:
        print(f"No results found for partition '{partition}'")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by method name for consistent ordering
    if len(df) > 1:
        df = df.sort_values(["Method", "Custom Desc"], ascending=[True, True])

    # Reset index
    df.reset_index(drop=True, inplace=True)

    if tostring:
        print(df.to_string(index=False))
    return df
