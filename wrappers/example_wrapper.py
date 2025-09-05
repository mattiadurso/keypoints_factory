# NOT working class

import sys
import torch
sys.path.append('methods/method')

from methods.method import method
from wrappers.wrapper import MethodWrapper, MethodOutput


class ExampleWrapper(MethodWrapper):
    def __init__(self, device: str = 'cuda', border=16):
        super().__init__(name='aliked', border=border, device=device)

        self.model = None  # .eval().to(device)

    @torch.inference_mode()
    def _extract(self, x, max_kpts: int = 2048) -> MethodOutput:
        x = x if x.dim() == 4 else x[None]

        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, 
                                enabled=self.use_amp):
            out = self.model(x)

        kpts = out['kpts'][:max_kpts]
        scores = out['scores'][:max_kpts]
        des = out['des'][:max_kpts]

        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)