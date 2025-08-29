from pathlib import Path
import sys
file_path = Path(__file__).resolve().parent
sys.path.append(str(file_path))
sys.path.append(str(file_path.parent))  # Add parent directory to sys.path

import gc
import numpy as np
from typing import Union, List

# pycolmap
import pycolmap as pcm

# Torch only used for input compatibility with your existing code
import torch
from torch import Tensor

# Your utils
from libutils.utils_2D import grid_sample_nan  # unused here but kept for interface parity
from utils.method_wrapper import MethodOutput

class SIFTPyColmapWrapper():
    def __init__(self, device: str = 'cuda', sift_opts: dict = None):
        """
        device: 'cuda' | 'cpu' | 'auto' (pycolmap Device; CUDA build required for GPU)
        sift_opts: dict of pycolmap.SiftExtractionOptions fields (optional)
        """
        self.name = 'sift_pycolmap'
        dev = device.lower()
        # if dev == 'cuda':
        #     self.device = pcm.Device.cuda
        # elif dev == 'cpu':
        #     self.device = pcm.Device.cpu
        # else:
        #     self.device = pcm.Device.auto

        # Build SIFT options (defaults are good; expose a dict for tweaks)
        if sift_opts is None:
            sift_opts = {}
        self.options = pcm.SiftExtractionOptions(**sift_opts)
        self.sift = pcm.Sift(self.options, device=pcm.Device.auto)

        self.descriptor_network = None           # for parity with your RIPE wrapper
        self.amp_enabled = False                 # not used here
        self.use_ripe_desc = False               # not used here

    def add_custom_descriptors(self, model):
        # Not supported with SIFT path; keep for API compatibility.
        self.descriptor_network = model
        print(f'Custom descriptors are not applied in {self.name} (SIFT already computes descriptors).')

    def _to_gray_uint8(self, img_chw: Tensor) -> np.ndarray:
        """
        Convert a torch CHW float tensor in [0,1] or [0,255] to HxW uint8 grayscale.
        Accepts either 1xHxW or 3xHxW.
        """
        if img_chw.dtype.is_floating_point:
            arr = img_chw.clamp(0, 1).detach().cpu().numpy()
            if arr.max() <= 1.0:
                arr = (arr * 255.0).round()
        else:
            arr = img_chw.detach().cpu().numpy().astype(np.float32)

        if arr.shape[0] == 3:
            # simple RGB->gray
            r, g, b = arr[0], arr[1], arr[2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif arr.shape[0] == 1:
            gray = arr[0]
        else:
            raise ValueError(f"Expected CHW with C=1 or 3, got {arr.shape}")

        gray = np.ascontiguousarray(gray.astype(np.uint8))
        return gray

    def img_from_numpy(self, img: np.ndarray) -> Union[Tensor, np.ndarray]:
        # Keep your cropping-to-multiple-of-16 (not required for SIFT, but harmless)
        factor = 16
        img = img[:img.shape[0] // factor * factor, :img.shape[1] // factor * factor]
        # Return CHW torch tensor on CPU for consistency with your pipeline
        img_out = torch.from_numpy(img.copy()).permute(2, 0, 1) / 255.0
        return img_out

    @torch.inference_mode()
    def extract(self, x, max_kpts: int = 2048, kpts=None) -> MethodOutput:
        """
        x: CHW torch tensor in [0,1] or [0,255] (H,W multiples of 16 OK).
        Returns MethodOutput with:
          - kpts: Tensor [N,2] (x,y)
          - kpts_scores: None (pycolmap SIFT doesn’t expose scores via extract)
          - des: Tensor [N,128]
        """
        # Ensure 4D -> 3D CHW
        x = x if x.dim() == 3 else x[0]
        gray = self._to_gray_uint8(x)

        # pycolmap.Sift.extract expects 2D np.uint8 / float32 image
        # Returns:
        #   keypoints: Nx4 float32 [x, y, scale, angle]
        #   descriptors: Nx128 float32
        kpts4, des = self.sift.extract(gray)  # numpy arrays

        if max_kpts is not None and kpts4.shape[0] > max_kpts:
            idx = np.argsort(-kpts4[:, 2])[:max_kpts]  # keep by scale as a proxy for saliency
            kpts4 = kpts4[idx]
            des = des[idx]

        # Convert to torch tensors on CPU (match your MethodOutput expectation)
        kpts_xy = torch.from_numpy(kpts4[:, :2].copy()).float()
        des_t = torch.from_numpy(des.copy()).float()

        output = MethodOutput(
            kpts=kpts_xy,
            kpts_scores=None,  # pycolmap doesn’t return DoG scores in this API
            des=des_t,
        )
        return output
