import sys
import torch
import kornia.feature as KF

from wrappers.wrapper import MethodWrapper, MethodOutput

sys.path.append("methods/disk")
from methods.disk.disk import DISK


class DiskWrapper(MethodWrapper):
    def __init__(self, device: str = "cuda:0", border=16) -> None:
        super().__init__(name="disk", border=border, device=device)
        weights_path = "methods/disk/depth-save.pth"

        disk = DISK(window=8, desc_dim=128)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        disk.load_state_dict(state_dict["extractor"])

        self.disk = disk.to(device)

    @torch.inference_mode()
    def _extract(
        self, img: torch.Tensor, max_kpts: int, custom_kpts=None
    ) -> MethodOutput:
        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if custom_kpts is not None:
                raise NotImplementedError("Custom keypoints not implemented for DISK.")
            # desc_vol is None if use_disk_descriptors is False
            features = self.disk.features(
                img[None], kind="nms", window_size=5, cutoff=0, n=max_kpts
            )
            kpts, kpts_scores = features[0].kp, features[0].kp_logp
            des = features[0].desc

            # ? order keypoints and descriptors by scores
            order = kpts_scores.argsort(descending=True)
            kpts_scores = kpts_scores[order]
            kpts = kpts[order]
            des = des[order]

            # ? only keep the first max_kpts keypoints
            kpts_scores = kpts_scores[:max_kpts]
            kpts = kpts[:max_kpts]
            des = des[:max_kpts]
            des = F.normalize(des, dim=1)

            if self.custom_descriptor is not None:
                des_vol = self.custom_descriptor(img[None])
                des = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[0][0].T

        return MethodOutput(kpts=kpts, kpts_scores=kpts_scores, des=des)


import kornia as K
import torch.nn.functional as F


class DiskWrapperKornia(MethodWrapper):
    def __init__(self, device: str = "cuda:0", border=16) -> None:
        super().__init__(name="disk", border=border, device=device)

        self.disk = KF.DISK.from_pretrained("depth").to(device)

    @torch.inference_mode()
    def _extract(
        self, img: torch.Tensor, max_kpts: int, custom_kpts=None
    ) -> MethodOutput:
        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            out = self.disk(img[None], max_kpts)[0]
            kpts, des = out.keypoints, out.descriptors

            if self.custom_descriptor is not None:
                des_vol = self.custom_descriptor(img[None])
                des = self.grid_sample_nan(kpts[None], des_vol, mode="nearest")[0][0].T

        return MethodOutput(kpts=kpts, des=des)
