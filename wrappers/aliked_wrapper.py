import sys
import torch

# Add method-specific path before importing from it
sys.path.append("methods/aliked")

from methods.aliked.nets.aliked import ALIKED
from wrappers.wrapper import MethodWrapper, MethodOutput


class AlikedWrapper(MethodWrapper):
    def __init__(self, device: str = "cuda", max_kpts: int = 2048, border=16):
        super().__init__(name="aliked", border=border, device=device)
        self.max_kpts = max_kpts

        self.model = (
            ALIKED(
                model_name="aliked-n16rot",
                device=device,
                top_k=max_kpts,
            )
            .eval()
            .to(device)
        )

    @torch.inference_mode()
    def _extract(self, x, max_kpts: int = 2048, custom_kpts=None) -> MethodOutput:
        """Extract keypoints and descriptors from image
        Args:
            x (Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W)
            max_kpts (int): Maximum number of keypoints to extract
            custom_kpts (Tensor, optional): Custom keypoints to use instead of detecting new ones.
                Shape should be (N, 2) in (x, y) format. Likely extracted from previous run.
        Returns:
            MethodOutput: Contains keypoints, scores, and descriptors
        """
        # Aliked requires max kpts to be set during initialization,
        # this allows changing it at any moment
        if self.max_kpts != max_kpts:
            custom_descriptor = self.custom_descriptor
            self.__init__(device=self.device, max_kpts=max_kpts, border=self.border)
            self.custom_descriptor = custom_descriptor

        x = x if x.dim() == 4 else x[None]

        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.custom_descriptor is None:  # no custom descriptor
                out = self.model(x)
                kpts, scores, des = (
                    out["keypoints"][0],
                    out["scores"][0],
                    out["descriptors"][0],
                )
                kpts = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1])

            else:  # use custom descriptor
                if custom_kpts is None:  # no custom kpts given, extract them
                    _, score_map = self.model.extract_dense_map(x)
                    kpts, scores, _ = self.model.dkd(score_map)
                    kpts, scores = kpts[0], scores[0]
                    kpts = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1])
                else:  # use the given custom kpts
                    kpts = custom_kpts.to(x.device)
                    scores = torch.ones(kpts.shape[0], device=x.device)

                des_vol = self.custom_descriptor(x)
                des = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[0][0].T

        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)
