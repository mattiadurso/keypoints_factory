from __future__ import annotations
import numpy as np
import torch as th
from torch import Tensor
from torchvision import transforms

from libutils.utils_local_feature_wrapper import LocalFeatureWrapper, LocalFeatureOutput
from superpoint.superpoint.superpoint import SuperPoint
from libutils.utils_2D import grid_sample_nan

class SuperPointWrapper(LocalFeatureWrapper):
    def __init__(self, device: str) -> None:
        super().__init__(name='SuperPoint', border=0, device=device)
        self.superpoint = SuperPoint({}).to(device)
        self.superpoint.requires_grad_(False)
        
        self.toGray = transforms.Grayscale()

        self.descriptor_network = None
        self.device = device
        self.dtype = next(self.superpoint.parameters()).dtype

    def img_from_numpy(self, img: np.ndarray):
        #return self.transform(img).to(self.device)
        img = img[:img.shape[0] // 16 * 16, :img.shape[1] // 16 * 16] # ensure divisible by 16
        # img_out = self.toTensor(img.copy()) 
        img_out = th.from_numpy(img).permute(2, 0, 1) / 255.
        return img_out.to(self.dtype).to(self.device)
    
    def add_custom_descriptors(self, model):
        self.descriptor_network = model.to(self.dtype).to(self.device)
        print(f'Adding custom descriptors to {self.name} with dtype {self.dtype} and device {self.device}')


    @ th.inference_mode()
    def _extract(self, img: Tensor | np.ndarray, max_kpts: float | int) -> LocalFeatureOutput:
        assert img.ndim == 3, 'image must be not batched'

        with th.amp.autocast(self.device, enabled=False):
            output = self.superpoint({'image': self.toGray(img)[None]})

            kpts = output['keypoints'][0]
            kpts_scores = output['scores'][0]

            idxs = kpts_scores.argsort(descending=True)
            idxs = idxs[:min(idxs.shape[0], max_kpts)]

            kpts = kpts[idxs] + 0.5
            kpts_scores = kpts_scores[idxs]
            kpts_sizes = (2 * self.border) * th.ones_like(idxs)

            if self.descriptor_network is not None:
                des_vol = self.descriptor_network(img[None])
                descriptors = grid_sample_nan(kpts[None], des_vol, mode='nearest')[0][0].T
            else:
                des_vol = None  # des_dim,H/8,W/8
                descriptors = output['descriptors'][0].permute(1, 0) # N,256
                descriptors = descriptors[idxs]
        
        output = LocalFeatureOutput(
            kpts=kpts,
            kpts_scores=kpts_scores,
            kpts_sizes=kpts_sizes,
            des=descriptors,
            des_vol=des_vol,
            # det_map=output['det_map'][0],
        )

        return output


    def get_keypoints_from_detmap(self, det_map: Tensor, max_kpts: float | int,
                                  border: int) -> LocalFeatureOutput:
        raise NotImplementedError

    def compute_detmap(self, img: Tensor) -> Tensor:
        raise NotImplementedError