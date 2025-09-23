from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union, Tuple, List, Any, Optional

import cv2
import torch
import numpy as np
from torch import Tensor

# from torchvision import transforms
from torch.nn import functional as F
from abc import ABC, abstractmethod


@dataclass
class MethodOutput:
    kpts: Tensor
    kpts_scores: Optional[Tensor] = None
    kpts_sizes: Optional[Tensor] = None  # receptive field
    kpts_scales: Optional[Tensor] = (
        None  # at which resolution they have been extracted, for multiscale
    )
    kpts_angles: Optional[Tensor] = None
    des: Optional[Tensor] = None
    des_vol: Optional[Tensor] = None

    def __post_init__(self):
        assert (
            self.kpts.ndim == 2
        ), f"kpts must have shape (N, 2), got {self.kpts.shape}"

        # put emtpy stuff in place of None
        if self.kpts_scores is None:
            self.kpts_scores = torch.ones_like(self.kpts[:, 0])
        if self.kpts_sizes is None:
            self.kpts_sizes = torch.ones_like(self.kpts[:, 0])
        if self.kpts_scales is None:
            self.kpts_scales = torch.ones_like(self.kpts[:, 0])
        if self.kpts_angles is None:
            self.kpts_angles = torch.zeros_like(self.kpts[:, 0])

    def __getitem__(self, key: Union[str, None]) -> Tensor:
        return self.__dict__[key]

    def __contains__(self, item) -> Union[Any, None]:
        return item in self.__dict__

    def get(self, key) -> Union[Any, None]:
        return self[key] if key in self.__dict__ else None

    def mask(self, mask: Tensor):
        assert mask.dtype == torch.bool, "mask must be boolean"
        return MethodOutput(
            kpts=self.kpts.clone()[mask],
            kpts_scores=self.kpts_scores.clone()[mask],
            kpts_sizes=self.kpts_sizes.clone()[mask],
            kpts_scales=self.kpts_scales.clone()[mask],
            kpts_angles=self.kpts_angles.clone()[mask],
            des=self.des.clone()[mask] if self.des is not None else None,
            des_vol=self.des_vol.clone() if self.des_vol is not None else None,
        )


class MethodWrapper(ABC):
    def __init__(self, name: str, border: int = 0, device: str = "cpu", use_amp=True):
        self.name = name
        self.border = border
        self.device = device
        self.custom_descriptor = None
        self.matcher = None

        # amp
        self.use_amp = use_amp
        if self.use_amp:
            print("Using automatic mixed precision.")
        self.amp_dtype = torch.float16

    def load_image(self, path, scaling=1.0):
        """
        Load image from path, convert to float32 tensor in [0, 1], resize if needed,
        and crop to multiple of 16."""
        img = self.read_image_to_torch(str(path)) / 255.0  # 3, H, W, float32 [0, 1]
        # resize if needed
        if scaling != 1.0:
            img = F.interpolate(
                img.unsqueeze(0),
                scale_factor=scaling,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        # crop to multiple of 16
        img = self.crop_multiple_of(img, multiple_of=16)

        return img.to(self.device)

    def read_image_to_torch(self, path):
        """
        Read image with OpenCV and convert to RGB.
        Returns a tensor uint8 CxHxW in [0,255].
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # HxWxC (BGR) or HxW (gray)
        if img is None:
            raise FileNotFoundError(path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t = torch.from_numpy(img)  # HxWxC, uint8/uint16
        t = t.permute(2, 0, 1).contiguous()  # CxHxW
        return t

    def crop_multiple_of(self, img, multiple_of=16):
        if isinstance(img, np.ndarray):
            H, W = img.shape[:2]
            new_H = (H // multiple_of) * multiple_of
            new_W = (W // multiple_of) * multiple_of
            return img[:new_H, :new_W, :]

        elif isinstance(img, Tensor):
            H, W = img.shape[-2:]
            new_H = (H // multiple_of) * multiple_of
            new_W = (W // multiple_of) * multiple_of
            return img[..., :new_H, :new_W]
        else:
            raise TypeError("Unsupported image type")

    def add_custom_descriptor(self, model):
        # can be whatever model that takes (B, C, H, W) as input and returns (B, D, H, W)
        self.custom_descriptor = model

    def to_pixel_coords(self, flow, h1, w1):
        w_ = w1 * (flow[..., 0] + 1) / 2
        h_ = h1 * (flow[..., 1] + 1) / 2
        flow = torch.stack((w_, h_), axis=-1)
        return flow

    @abstractmethod
    def _extract(
        self, img: Union[Tensor, np.ndarray], max_kpts: Union[float, int]
    ) -> MethodOutput:
        raise NotImplementedError

    @torch.inference_mode()
    def extract(
        self,
        img: Union[Tensor, np.ndarray],
        max_kpts: Union[float, int],
        custom_kpts=None,
    ) -> MethodOutput:
        if not isinstance(img, Tensor):
            raise TypeError("Input image must be a Tensor")

        H, W = img.shape[-2:]  # images is supposed to be (C, H, W) or (B, C, H, W)
        output = self._extract(img, max_kpts, custom_kpts)
        # ? remove all the points in the border
        valid_mask = (
            (output.kpts[:, 0] > self.border)
            & (output.kpts[:, 0] < W - self.border)
            & (output.kpts[:, 1] > self.border)
            & (output.kpts[:, 1] < H - self.border)
        )
        output = output.mask(valid_mask)
        return output

    def grid_sample_nan(
        self, xy: Tensor, img: Tensor, mode="nearest"
    ) -> Tuple[Tensor, Tensor]:
        """pytorch grid_sample with embedded coordinate normalization and grid nan handling (if a nan is present in xy,
        the output will be nan). Works both with input with shape Bxnx2 and B x n0 x n1 x 2
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
        assert (
            xy.dim() == 3 or xy.dim() == 4
        ), f"xy must have 3 or 4 dimensions, got {xy.dim()}"
        B, C, H, W = img.shape

        xy_norm = self.normalize_pixel_coordinates(
            xy, img.shape[-2:]
        )  # BxNx2 or BxN0xN1x2
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

    def normalize_pixel_coordinates(self, xy: Tensor, shape: Tuple[int, int]) -> Tensor:
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
        # ! the shape index are flipped because the coordinates are given as x,y but shape is H,W
        xy_norm[..., 0] = 2 * xy_norm[..., 0] / shape[1]
        xy_norm[..., 1] = 2 * xy_norm[..., 1] / shape[0]
        xy_norm -= 1
        return xy_norm

    def match(self, des0: List[Tensor], des1: List[Tensor]):
        if self.matcher is None:
            raise ValueError("No matcher defined for this wrapper")
        return self.matcher.match(des0, des1)

    def extract_multiscale(
        self,
        img: Union[Tensor, np.ndarray],
        max_kpts: Union[float, int],
        scales: List[float],
    ) -> List[MethodOutput]:
        raise NotImplementedError
