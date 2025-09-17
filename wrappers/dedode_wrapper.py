import sys
import gc
import warnings
import torch
import torch.nn.functional as F
from torchvision import transforms

from pathlib import Path

method_path = Path(__file__).resolve().parents[1] / "methods/dedode"
sys.path.append(str(method_path))

warnings.filterwarnings("ignore", category=UserWarning)

from methods.dedode.DeDoDe import (
    dedode_detector_L,
    dedode_descriptor_B,
    dedode_descriptor_G,
)
from wrappers.wrapper import MethodWrapper, MethodOutput


class DeDoDeWrapper(MethodWrapper):
    def __init__(self, descriptor_G: bool = False, device: str = "cuda:0", border=16):
        name = "dedode-G" if descriptor_G else "dedode-B"
        super().__init__(name=name, border=border, device=device)

        # Load weights
        detector_path = method_path / "weights/dedode_detector_L.pth"
        descriptor_G_path = method_path / "weights/dedode_descriptor_G.pth"
        descriptor_B_path = method_path / "weights/dedode_descriptor_B.pth"

        self.detector = dedode_detector_L(
            weights=torch.load(detector_path, map_location=device), device=device
        )
        self.descriptor_G = descriptor_G
        if descriptor_G:
            self.descriptor = dedode_descriptor_G(
                weights=torch.load(descriptor_G_path, map_location=device),
                device=device,
            )
        else:
            self.descriptor = dedode_descriptor_B(
                weights=torch.load(descriptor_B_path, map_location=device),
                device=device,
            )

        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def add_custom_descriptor(self, model):
        self.custom_descriptor = model
        # clean up
        self.descriptor = None
        self.descriptor_G = False
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _extract(self, x, max_kpts: int = 2048, custom_kpts=None) -> MethodOutput:
        x = x if x.dim() == 4 else x[None]

        # eventually cropping to multiples of 14
        if self.descriptor_G:
            x = self.crop_multiple_of(x, multiple_of=14)

        batch = {"image": self.normalizer(x)}

        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            # detector
            if custom_kpts is None:
                out = self.detector.detect(batch, num_keypoints=max_kpts)
                kpts, scores = out["keypoints"], out["confidence"]
                kpts_pix = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1])
            else:  # use the given custom kpts
                kpts_pix = custom_kpts.to(x.device)
                scores = torch.ones(kpts_pix.shape[0], device=x.device)

            # descriptors
            if self.custom_descriptor is None:
                out = self.descriptor.describe_keypoints(batch, kpts)
                des = out["descriptions"][0]
                # L2 normalization, needed since not done in dedode descriptor
                des = F.normalize(des, p=2, dim=-1)

            else:  # custom descriptor network
                des_vol = self.custom_descriptor(x)
                des = self.grid_sample_nan(kpts_pix[None], des_vol, mode="nearest")[0][
                    0
                ].permute(1, 2, 0)[0]

        return MethodOutput(kpts=kpts_pix[0], kpts_scores=scores[0], des=des)
