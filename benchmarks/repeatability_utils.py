import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor


def compute_repeatabilities_from_kpts(
    kpts1: Tensor,
    kpts2: Tensor,
    K1: Tensor,
    K2: Tensor,
    Z1: Tensor,
    Z2: Tensor,
    P1: Tensor,
    P2: Tensor,
    img1_shape: tuple[int, int] | None = None,
    img2_shape: tuple[int, int] | None = None,
    image_names_list: list[str] | None = None,
    px_thrs: float | list[float] = [1.0, 3.0, 5.0],
) -> tuple[float, float] | tuple[Tensor, Tensor]:
    """compute the repeatability between two sets of keypoints given the intrinsics, depthmaps and extrinsics
    Args:
        kpts1: keypoints in image 1
            B,n,2
        kpts2: keypoints in image 2
            B,n,2
        K1: intrinsics of camera 1
            B,3,3
        K2: intrinsics of camera 2
            B,3,3
        Z1: depthmap of image 1
            B,H,W
        Z2: depthmap of image 2
            B,H,W
        P1: extrinsics of camera 1 (if P_rel is not provided)
            B,4,4
        P2: extrinsics of camera 2 (if P_rel is not provided)
            B,4,4
        P_rel: relative extrinsics from camera 1 to camera 2 (if P1 and P2 are not provided)
            B,4,4
        px_thrs: pixel thresholds to compute the repeatability. Can be a single float or a list of floats.
    Returns:
        rep: the repeatability at the provided thresholds
            float or Tensor (if multiple thresholds are provided)
        rep_mnn: the repeatability at the provided thresholds considering only mutual nearest neighbors
            float or Tensor (if multiple thresholds are provided)
    """
    assert kpts1.ndim == 3 and kpts1.shape[-1] == 2
    assert kpts2.ndim == 3 and kpts2.shape[-1] == 2
    # all need to have the same batch size
    assert (
        kpts1.shape[0]
        == kpts2.shape[0]
        == K1.shape[0]
        == K2.shape[0]
        == Z1.shape[0]
        == Z2.shape[0]
        == P1.shape[0]
        == P2.shape[0]
    ), f"All the inputs must have the same batch size, got {kpts1.shape[0]}, {kpts2.shape[0]}, {K1.shape[0]}, {K2.shape[0]}, {Z1.shape[0]}, {Z2.shape[0]}, {P1.shape[0]}, {P2.shape[0]}. Images size are assumed to be the same of the depthmaps."
    if image_names_list is None:
        image_names_list = range(kpts1.shape[0])
    assert (
        len(image_names_list) == kpts1.shape[0]
    ), f"image_names_list must have the same length as the batch size, got {len(image_names_list)} and {kpts1.shape[0]}."

    kpts12 = reproject_2D_2D(kpts1, Z1, P1, P2, K1, K2, img2_shape)  # B,n,2
    kpts21 = reproject_2D_2D(kpts2, Z2, P2, P1, K2, K1, img1_shape)  # B,n,2

    reps = {}
    for b, img_name in enumerate(image_names_list):
        rep, rep_mnn = compute_repeatabilities(
            kpts1[b], kpts2[b], kpts12[b], kpts21[b], px_thrs
        )
        rep, rep_mnn = rep.cpu(), rep_mnn.cpu()
        reps[img_name] = {
            **{f"rep_{int(pix)}": rep[i].item() for i, pix in enumerate(px_thrs)},
            **{
                f"rep_mnn_{int(pix)}": rep_mnn[i].item()
                for i, pix in enumerate(px_thrs)
            },
        }

    return reps


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
    # dist0 = torch.cdist(xy0.to(torch.float), xy1_proj,         compute_mode='donot_use_mm_for_euclid_dist')  # n0,n1
    # dist1 = torch.cdist(xy0_proj,         xy1.to(torch.float), compute_mode='donot_use_mm_for_euclid_dist')  # n0,n1
    # ? high memory usage, fast and correct
    dist0 = (xy0[:, None, :] - xy1_proj[None, :, :]).norm(dim=2)  # n0,n1
    dist1 = (xy0_proj[:, None, :] - xy1[None, :, :]).norm(dim=2)  # n0,n1
    # # ? low memory usage, fast but non-deterministic
    # dist0 = torch.cdist(xy0.to(torch.float), xy1_proj)  # n0,n1
    # dist1 = torch.cdist(xy0_proj,         xy1.to(torch.float))  # n0,n1
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
            (0,), dtype=torch.long, device=device
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
        xy0_closest_matrix[torch.arange(len(xy0)), closest0] = True
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


def compute_repeatabilities(
    xy0: Tensor,
    xy1: Tensor,
    xy0_proj: Tensor,
    xy1_proj: Tensor,
    px_thrs: float | list[float],
) -> tuple[float, float] | tuple[Tensor, Tensor]:
    assert (
        xy0.ndim == 2 and xy0.shape[1] == 2
    ), f"the shape of xy0, xy1 should be (n0,2), but got {xy0.shape}, {xy1.shape}"
    assert (
        xy0_proj.ndim == 2 and xy1_proj.ndim == 2
    ), f"xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.shape} and {xy1_proj.shape}"
    device = xy0.device

    n_valid0 = (~xy0_proj.isnan().any(1)).sum()
    n_valid1 = (~xy1_proj.isnan().any(1)).sum()
    n_valid = n_valid0 + n_valid1
    (
        mnn_mask,
        xy0_closest_dist_mnn,
        xy1_closest_dist_mnn,
        xy0_closest_dist,
        xy1_closest_dist,
    ) = find_mutual_nearest_neighbors_from_keypoints_and_their_projections(
        xy0, xy1, xy0_proj, xy1_proj
    )

    if not isinstance(px_thrs, list):
        px_thrs = [px_thrs]

    rep = torch.zeros(len(px_thrs), device=device)
    rep_mnn = torch.zeros(len(px_thrs), device=device)
    for i, px_thr in enumerate(px_thrs):
        n_close0 = (
            (xy0_closest_dist <= px_thr).sum().item()
            if (xy0_closest_dist.shape[0] > 0)
            else 0
        )
        n_close1 = (
            (xy1_closest_dist <= px_thr).sum().item()
            if (xy1_closest_dist.shape[0] > 0)
            else 0
        )

        n_close0_mnn = (
            (xy0_closest_dist_mnn <= px_thr).sum().item()
            if (xy0_closest_dist_mnn.shape[0] > 0)
            else 0
        )
        n_close1_mnn = (
            (xy1_closest_dist_mnn <= px_thr).sum().item()
            if (xy1_closest_dist_mnn.shape[0] > 0)
            else 0
        )

        rep[i] = (n_close0 + n_close1) / n_valid if n_valid > 0 else 0
        rep_mnn[i] = (n_close0_mnn + n_close1_mnn) / n_valid if n_valid > 0 else 0

    rep = rep[0].item() if rep.shape[0] == 1 else rep
    rep_mnn = rep_mnn[0].item() if rep_mnn.shape[0] == 1 else rep_mnn

    return rep, rep_mnn


def grid_sample_nan(xy: Tensor, img: Tensor, mode="nearest") -> tuple[Tensor, Tensor]:
    """pytorch grid_sample with embedded coordinate normalization and grid nan handling (if a nan is present in xy,
    the output will be nan). Works both with input with shape B,n,2 and B,n0,n1,2
    xy point that fall outside the image are treated as nan (those which are really close are interpolated using
    border padding mode)
    Args:
        xy: input coordinates (with the convention top-left pixel center at (0.5, 0.5))
            B,n,2 or B,n0,n1,2
        img: the image where the sampling is done
            BxCxHxW or BxHxW
        mode: the interpolation mode
    Returns:
        sampled: the sampled values
            BxCxN or BxCxN0xN1 (if no C dimension in input BxN or BxN0xN1)
        mask_img_nan: mask of the points that had a nan in the img. The points xy that were nan appear as false in the
            mask in the same way as point that had a valid img value. This is done to discriminate between invalid
            sampling position and valid sampling position with a nan value in the image
            BxN or BxN0xN1
    """
    assert img.dim() in {3, 4}
    if img.dim() == 3:
        # ? remove the channel dimension from the result at the end of the function
        squeeze_result = True
        img.unsqueeze_(1)
    else:
        squeeze_result = False

    assert xy.shape[-1] == 2
    assert xy.dim() == 3 or xy.dim() == 4
    B, C, H, W = img.shape

    xy_norm = normalize_pixel_coordinates(xy, img.shape[-2:])  # BxNx2 or BxN0xN1x2
    # ? set to nan the point that fall out of the second image
    xy_norm[(xy_norm < -1) + (xy_norm > 1)] = float("nan")
    if xy.ndim == 3:
        sampled = F.grid_sample(
            img,
            xy_norm[:, :, None, ...],
            align_corners=False,
            mode=mode,
            padding_mode="border",
        ).view(
            B, C, xy.shape[1]
        )  # BxCxN
    else:
        sampled = F.grid_sample(
            img, xy_norm, align_corners=False, mode=mode, padding_mode="border"
        )  # BxCxN0xN1
    # ? points xy that are not nan and have nan img. The sum is just to squash the channel dimension
    mask_img_nan = torch.isnan(sampled.sum(1))  # BxN or BxN0xN1
    # ? set to nan the sampled values for points xy that were nan (grid_sample consider those as (-1, -1))
    xy_invalid = xy_norm.isnan().any(-1)  # BxN or BxN0xN1
    if xy.ndim == 3:
        sampled[xy_invalid[:, None, :].repeat(1, C, 1)] = float("nan")
    else:
        sampled[xy_invalid[:, None, :, :].repeat(1, C, 1, 1)] = float("nan")

    if squeeze_result:
        img.squeeze_(1)
        sampled.squeeze_(1)

    return sampled, mask_img_nan


def normalize_pixel_coordinates(
    xy: Tensor, shape: tuple[int, int] | Tensor | np.ndarray
) -> Tensor:
    """normalize pixel coordinates from -1 to +1. Being (-1,-1) the exact top left corner of the image
    the coordinates must be given in a way that the center of pixel is at half coordinates (0.5,0.5)
    xy ordered as (x, y) and shape ordered as (H, W)
    Args:
        xy: input coordinates in order (x,y) with the convention top-left pixel center is at coordinates (0.5, 0.5)
            ...x2
        shape: shape of the image in the order (H, W)
    Returns:
        xy_norm: normalized coordinates between [-1, 1]
    """
    xy_norm = xy.clone()
    # ? the shape index are flipped because the coordinates are given as x,y but shape is H,W
    xy_norm[..., 0] = 2 * xy_norm[..., 0] / shape[1]
    xy_norm[..., 1] = 2 * xy_norm[..., 1] / shape[0]
    xy_norm -= 1
    return xy_norm


def to_homogeneous(xy: Tensor) -> Tensor:
    return torch.cat((xy, torch.ones_like(xy[..., 0:1])), dim=-1)


def from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    z_vec: Tensor = points[..., -1:]
    # set the results of division by zero/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))
    output = scale * points[..., :-1]
    return output


def unproject_to_virtual_plane(
    xy: Tensor, K: Tensor, cast_to_double: bool = True
) -> Tensor:
    """unproject points to the camera virtual plane at depth 1
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns:
        xyz: 3D points laying on the virtual plane
            B,n,3
    """
    xy_hom = to_homogeneous(xy)  # B,n,3
    if cast_to_double:
        original_type = xy.dtype
        # Bx3x3 * Bx3xn = Bx3xn  -> B,n,3 after permute
        xyz = (
            (
                torch.inverse(K.to(torch.double))
                @ (xy_hom.permute(0, 2, 1).to(torch.double))
            )
            .permute(0, 2, 1)
            .to(original_type)
        )
    else:
        # Bx3x3 * Bx3xn = Bx3xn  -> B,n,3 after permute
        xyz = (torch.inverse(K) @ (xy_hom.permute(0, 2, 1))).permute(0, 2, 1)

    return xyz


def unproject_to_3D(xy: Tensor, K: Tensor, depths: Tensor) -> Tensor:
    """unproject points to 3D in the camera ref system
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        K: intrinsics of the camera
            B,3,3
        depths: the points depth
            B,n
    Returns:
        xyz: unprojected 3D points in the camera reference system
            B,n,3
    """
    assert xy.shape[0] == K.shape[0] and xy.shape[0] == depths.shape[0]
    assert xy.shape[1] == depths.shape[1]
    assert xy.shape[2] == 2

    xyz = unproject_to_virtual_plane(xy, K)  # B,n,3
    xyz *= depths[:, :, None]  # B,n,3

    return xyz


def invert_P(P: Tensor) -> Tensor:
    """invert the extrinsics P matrix in a more stable way with respect to np.linalg.inv()
    Args:
        P: input extrinsics P matrix
            Bx4x4
    Return:
        P_inv: the inverse of the P matrix
            Bx4x4
    Raises:
        None
    """
    B = P.shape[0]
    R = P[:, 0:3, 0:3]
    t = P[:, 0:3, 3:4]
    P_inv = torch.cat((R.permute(0, 2, 1), -R.permute(0, 2, 1) @ t), dim=2)
    P_inv = torch.cat(
        (P_inv, P.new_tensor([[0.0, 0.0, 0.0, 1.0]])[None, ...].repeat(B, 1, 1)), dim=1
    )
    return P_inv


def change_reference_3D_points(
    xyz0: Tensor, P0: Tensor, P1: Tensor, cast_to_double: bool = True
) -> Tensor:
    """move 3D points from P0 to P1 reference systems
    Args:
        xyz0: the 3D points in the P0 coordinate system
            B,n,3
        P0: the source coordinate system
            B,4,4
        P1: the destination coordinate system
            B,4,4
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns
        xyz1: the 3D points in the P1 coordinate system
            B,n,3
    """
    assert (
        xyz0.shape[0] == P0.shape[0] and xyz0.shape[0] == P1.shape[0]
    ), f"Expected xyz0 and P0 to have the same batch size, got {xyz0.shape[0]} and {P0.shape[0]}"
    assert xyz0.shape[2] == 3, f"Expected xyz0 to have 3 channels, got {xyz0.shape[2]}"
    assert (
        P0.shape[1] == 4 and P0.shape[2] == 4
    ), f"Expected P0 to have shape Bx4x4, got {P0.shape}"
    assert (
        P1.shape[1] == 4 and P1.shape[2] == 4
    ), f"Expected P1 to have shape Bx4x4, got {P1.shape}"

    xyz0_hom = to_homogeneous(xyz0)  # B,n,4
    if cast_to_double:
        original_dtype = xyz0.dtype
        P0_inv = invert_P(P0.to(torch.double))
        xyz1_hom = (
            P1.to(torch.double) @ P0_inv @ xyz0_hom.permute(0, 2, 1).to(torch.double)
        )  # B,4,n
        xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1)).to(original_dtype)  # B,n,3
    else:
        P0_inv = invert_P(P0)
        xyz1_hom = P1 @ P0_inv @ xyz0_hom.permute(0, 2, 1)  # B,4,n
        xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1))  # B,n,3

    return xyz1


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


def project_to_2D(
    xyz: Tensor,
    K: Tensor,
    img_shape: tuple[int, int] | None = None,
    border: int = 0,
) -> Tensor | tuple[Tensor, Tensor]:
    """project 3D points to 2D using the provided intrinsics matrix K. If img_shape is provided, set to nan the points
    that project out of the img and additionally return mask_outside boolean tensor
    Args:
        xyz: the 3D points
            B,n,3
        K: the camera intrinsics matrix
            B,3,3
        img_shape: if provided, set to nan the points that map out of the image and additionally return mask_outside
        border: if img_shape is provided, set to nan the points that map out of the image border
    Returns
        xy_proj: the 2D projection of the 3D points
            B,n,2
        mask_outside: optional (if img_shape is provided). True where the point map outside img_shape
            B,n bool
    """
    original_dtype = xyz.dtype
    # B,3,3 * B,3,n =  B,3,n  -> B,n,3 after permutation
    xy_proj_hom = (K.to(torch.double) @ xyz.permute(0, 2, 1).to(torch.double)).permute(
        0, 2, 1
    )
    xy_proj = from_homogeneous(xy_proj_hom).to(original_dtype)  # B,n,2

    if img_shape is not None:
        # ? filter points that fall outside the second image but have depth valid
        # ? as the comparison of a 'nan' values with something else is always false, only the points that had valid
        # ? depth will appear in mask_outside
        mask_outside = (
            (xy_proj[..., 0] < border)
            + (xy_proj[..., 0] >= img_shape[1] - border)
            + (xy_proj[..., 1] < border)
            + (xy_proj[..., 1] >= img_shape[0] - border)
        )
        xy_proj = filter_outside(xy_proj, img_shape, border)
        return xy_proj, mask_outside
    else:
        return xy_proj


def reproject_2D_2D(
    xy0: Tensor,
    depthmap0: Tensor,
    P0: Tensor,
    P1: Tensor,
    K0: Tensor,
    K1: Tensor,
    img1_shape: tuple[int, int] | None = None,
    border: int = 0,
    mode: str = "nearest",
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """projects xy0 points from img0 to img1 using depth0. Points that have an invalid depth='nan' are
        set to 'nan' (if bilinear sampling is used, all the 4 closest depth values must be valid to get a valid projection).
        If img1_shape is provided, also the points that project out of the second image are set to Nan
    Args:
        xy0: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            B,n,2
        depthmap0: depthmap of img0
            B,H,W or B,n
        P0: camera0 extrinsics matrix
            B,4,4
        P1: camera1 extrinsics matrix
            B,4,4
        K0: camera0 intrinsics matrix
            B,3,3
        K1: camera1 intrinsics matrix
            B,3,3
        img1_shape: shape of img1 (H, W)
        border: if > 0, the points that project closer to the image borders are set to nan
        mode: depthmap interpolation mode, can be 'nearest' or 'bilinear'
    Returns:
        xy0_proj: the projected keypoints in img1
            B,n,2
        mask_invalid_depth: mask of points that had invalid depth
            B,n  bool
        mask_outside: optional (if img1_shape is provided) mask of points that had valid depth but project out of the
            second image
            B,n  bool
    """
    # ? interpolate depths
    if depthmap0.dim() == 3:
        selected_depths0, mask_invalid_depth0 = grid_sample_nan(
            xy0, depthmap0, mode=mode
        )  # Bxn, Bxn
    else:
        # pre-sampled depths
        assert (
            depthmap0.shape == xy0.shape[:2]
        ), f"If depthmap0 is not BxHxW, it must be Bxn, got {depthmap0.shape} and {xy0.shape}"
        selected_depths0 = depthmap0

    # ? use the depth to define the 3D coordinates of points in the ref system of camera0
    xyz0 = unproject_to_3D(xy0, K0, selected_depths0)  # B,n,3

    # ? change the ref system of the 3d point to camera1
    xyz0_proj = change_reference_3D_points(xyz0, P0, P1)  # B,n,3

    # ? project the point in the destination image
    if img1_shape is not None:
        xy0_proj, mask_outside0 = project_to_2D(
            xyz0_proj, K1, img1_shape, border
        )  # B,n,2, B,n,2
        return xy0_proj
    else:
        assert border == 0, "border must be 0 if img1_shape is not provided"
        xy0_proj = project_to_2D(xyz0_proj, K1)  # B,n,2, B,n,2
        return xy0_proj
