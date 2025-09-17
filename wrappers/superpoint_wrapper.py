from __future__ import annotations

import sys
import torch
from torchvision import transforms

sys.path.append("methods/superpoint")

from wrappers.wrapper import MethodWrapper, MethodOutput
from methods.superpoint.models.superpoint import SuperPoint


class SuperPointWrapper(MethodWrapper):
    def __init__(self, device: str, border=16) -> None:
        super().__init__(name="SuperPoint", border=border, device=device)
        config = {
            "keypoint_threshold": -1,  # min score, -1 to disable
            "max_keypoints": 2048,
        }
        self.superpoint = SuperPoint(config).to(device)
        self.superpoint.requires_grad_(False)

        self.toGray = transforms.Grayscale()

    @torch.inference_mode()
    def _extract(
        self, img: torch.Tensor, max_kpts: float | int, custom_kpts=None
    ) -> MethodOutput:
        if max_kpts != self.superpoint.config["max_keypoints"]:
            self.superpoint.config["max_keypoints"] = max_kpts
            print(f"Updated max_keypoints to {max_kpts}.")

        assert img.ndim == 3, "image must be not batched"

        if custom_kpts is not None:
            raise NotImplementedError(
                "Custom keypoints not implemented for SuperPoint."
            )

        with torch.amp.autocast(self.device, enabled=False):
            output = self.superpoint({"image": self.toGray(img)[None]})

            kpts = output["keypoints"][0]
            kpts_scores = output["scores"][0]

            idxs = kpts_scores.argsort(descending=True)
            idxs = idxs[: min(idxs.shape[0], max_kpts)]

            kpts = kpts[idxs] + 0.5
            kpts_scores = kpts_scores[idxs]
            kpts_sizes = (2 * self.border) * torch.ones_like(idxs)

            if self.custom_descriptor is not None:
                des_vol = self.custom_descriptor(img[None])
                descriptors = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[
                    0
                ][0].T
            else:
                descriptors = output["descriptors"][0].permute(1, 0)  # N,256
                descriptors = descriptors[idxs]

        output = MethodOutput(
            kpts=kpts, kpts_scores=kpts_scores, kpts_sizes=kpts_sizes, des=descriptors
        )

        return output
