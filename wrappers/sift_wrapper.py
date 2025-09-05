import torch
import numpy as np

import pycolmap as pcm
from wrappers.wrapper import MethodWrapper, MethodOutput


class SIFTPyColmapWrapper(MethodWrapper):
    def __init__(self, device: str = 'cuda', border=16, sift_opts: dict = None):
        """
        device: 'cuda' | 'cpu' | 'auto' (pycolmap Device; CUDA build required for GPU)
        sift_opts: dict of pycolmap.SiftExtractionOptions fields (optional)
        """
        super().__init__(name='sift', border=border, device=device)
        print('By default, SIFT uses CPU even if CUDA is available. \
              To use GPU, install pycolmap with CUDA support.\
              ')  # https://colmap.github.io/pycolmap/index.html

        # Build SIFT options (defaults are good; expose a dict for tweaks)
        if sift_opts is None:
            sift_opts = {}
        self.options = pcm.SiftExtractionOptions(**sift_opts)
        self.sift = pcm.Sift(self.options, device=pcm.Device.auto)

    def _to_gray_uint8(self, img_chw: torch.Tensor) -> np.ndarray:
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

    def _extract(self, x, max_kpts: int = 2048) -> MethodOutput:
        """
        x: CHW torch tensor in [0,1] or [0,255] (H,W multiples of 16 OK).
        Returns MethodOutput with:
          - kpts: Tensor [N,2] (x,y)
          - kpts_scores: None (pycolmap SIFT doesn't expose scores via extract)
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
            idx = np.argsort(-kpts4[:, 2])[:max_kpts]
            kpts4 = kpts4[idx]
            des = des[idx]

        # Convert to torch tensors on CPU (match your MethodOutput expectation)
        kpts_xy = torch.from_numpy(kpts4[:, :2].copy()).float()
        des_t = torch.from_numpy(des.copy()).float()

        return MethodOutput(kpts=kpts_xy, kpts_scores=None, des=des_t,)
