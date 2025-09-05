import sys
import torch
from pathlib import Path

sys.path.append('methods/ripe')
from methods.ripe.ripe import vgg_hyper
from wrappers.wrapper import MethodWrapper, MethodOutput


class RIPEWrapper(MethodWrapper):
    def __init__(self, device: str = 'cuda', border=16):
        super().__init__(name='ripe', border=border, device=device)

        model_path = "methods/ripe/ckpt/ripe_weights.pth"
        self.model = vgg_hyper(Path(model_path)).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _extract(self, x, max_kpts: int = 2048) -> MethodOutput:
        x = x if x.dim() == 4 else x[None]

        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype,
                                enabled=self.use_amp):
            kpts, des, scores = self.model.detectAndCompute(x, threshold=0.5,
                                                            top_k=max_kpts)

            if self.custom_descriptor is not None:
                des_vol = self.custom_descriptor(x)
                des = self.grid_sample_nan(kpts[None], des_vol, mode='nearest')[0][0].T

        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)
