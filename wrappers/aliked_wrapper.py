from pathlib import Path
import sys
import torch
import numpy as np


sys.path.append('methods/aliked')
from methods.aliked.nets.aliked import ALIKED

from wrappers.wrapper import MethodWrapper, MethodOutput

class AlikedWrapper(MethodWrapper):
    def __init__(self, device: str = 'cuda', max_kpts: int = 2048, border=16):
        super().__init__(name='aliked', border=border, device=device)
        self.max_kpts = max_kpts

        self.aliked = ALIKED(
            model_name='aliked-n16rot',
            device=device,
            top_k=max_kpts,
        ).eval().to(device)

    @torch.inference_mode()
    def _extract(self, x, max_kpts: int=2048) -> MethodOutput:
        # Aliked asked for max kpts to be set during initialization, 
        # in this way it's possible to change it in any moment
        if self.max_kpts != max_kpts:
            description_network = self.descriptor_network
            self.__init__(name='aliked', device=self.device, max_kpts=max_kpts, border=self.border)
            self.descriptor_network = description_network
        
        x = x if x.dim() == 4 else x[None]

        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            if self.custom_descriptor is None:
                out = self.aliked(x)
                kpts, scores, des = out['keypoints'][0], out['scores'][0], out['descriptors'][0]
                kpts = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1])

            else:
                _, score_map = self.aliked.extract_dense_map(x)
                kpts, scores, _ = self.aliked.dkd(score_map)
                kpts, scores = kpts[0], scores[0]

                des_vol = self.descriptor_network(x)
                kpts = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1])
                des = self.grid_sample_nan(kpts[None], des_vol, mode='nearest')[0][0].T

        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)

