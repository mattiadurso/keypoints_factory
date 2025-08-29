from pathlib import Path
import sys
sys.path.append('wrappers/aliked/aliked')

import os
import torch
from torch import Tensor
import numpy as np
from typing import Union, List
from libutils.utils_2D import grid_sample_nan
from utils.method_wrapper import MethodOutput

# no need for dedode code, just importing from kornia
from aliked.nets.aliked import ALIKED

class AlikedWrapper():
    def __init__(self, device: str = 'cuda:0', max_kpts: int = 2048, mode: str = 'nearest'):
        self.name = 'aliked'
        self.device = device
        self.max_kpts = max_kpts

        self.aliked = ALIKED(
            model_name='aliked-n16rot',
            device=device,
            top_k=max_kpts,
        ).eval().to(device)
        
        self.device = device
        self.dtype = next(self.aliked.parameters()).dtype
        self.descriptor_network = None
        self.mode = mode  # 'nearest' or 'bilinear'
        # print(f'Descriptor sampling mode: {mode}')

    def add_custom_descriptors(self, model):
        self.descriptor_network = model.to(self.dtype).to(self.device)

    def img_from_numpy(self, img: np.ndarray) -> Union[Tensor, np.ndarray]:
        # Crop image such that dimensions are multiples of 16
        img = img[:img.shape[0] // 16 * 16, :img.shape[1] // 16 * 16]
        if img.ndim == 2:
            # Grayscale image, add channel dimension
            img = np.expand_dims(img, axis=-1)
        if img.shape[2] == 1:
            # Convert single channel to 3 channels
            img = np.repeat(img, 3, axis=2)
        img_out = torch.from_numpy(img.copy()).permute(2, 0, 1).to(self.device) / 255.
        return img_out.to(self.dtype)

    @torch.inference_mode()
    def extract(self, x, max_kpts: int=2048, kpts=None) -> MethodOutput:
        if self.max_kpts != max_kpts:
            description_network = self.descriptor_network
            self.__init__(self.device, max_kpts)
            self.descriptor_network = description_network
        
        x = x if x.dim() == 4 else x[None]

        with torch.amp.autocast(self.device, enabled=False):
            # if self.descriptor_network do not compute aliked descriptors
            if self.descriptor_network is None and kpts is None:
                out = self.aliked(x)
                kpts, scores, des = out['keypoints'][0], out['scores'][0], out['descriptors'][0]
                kpts = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1])
                des_vol = None

            elif self.descriptor_network is None and kpts is not None:
                feature_map, score_map = self.aliked.extract_dense_map(x)
                des, _ = self.aliked.desc_head(feature_map, self.from_pixel_coords(kpts[None],x.shape[-2], x.shape[-1]))
                des_vol = torch.zeros((1, des[0].shape[1], x.shape[-2], x.shape[-1]), device=x.device)
                des_vol[:, :, kpts[:, 1].long(), kpts[:, 0].long()] = des[0].T
                scores = None
                # use kpts to compute descriptors
            elif self.descriptor_network is not None:
                _, score_map = self.aliked.extract_dense_map(x)
                kpts, scores, _ = self.aliked.dkd(score_map)
                
                kpts, scores = kpts[0], scores[0]
                des_vol = self.descriptor_network(x,kpts[None])
                kpts = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1])
                des = grid_sample_nan(kpts[None], des_vol, mode=self.mode)[0][0].T 
                
        output = MethodOutput(
            kpts=kpts,
            kpts_scores=scores,
            des=des,
            des_vol=des_vol,
        )

        return output
    
    def to_pixel_coords(self, flow, h1, w1):
        flow = (
            torch.stack(
                (
                    w1 * (flow[..., 0] + 1) / 2,
                    h1 * (flow[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
        )
        return flow

    def from_pixel_coords(self, flow_pixel, h1, w1):
        x_norm = 2 * flow_pixel[..., 0] / w1 - 1
        y_norm = 2 * flow_pixel[..., 1] / h1 - 1
        return torch.stack((x_norm, y_norm), dim=-1)

    
    def match(self, des0: List[Tensor], des1: List[Tensor]):
        return self.matcher.match(des0, des1)
    
    
if __name__=='__main__':
    # test
    img = np.random.rand(480,640,3).astype(np.float32)
    wrapper = AlikedWrapper()
    img_t = wrapper.img_from_numpy(img)
    out = wrapper.extract(img_t)
    print(out.kpts.shape, out.kpts_scores.shape, out.des.shape)
    print(out.kpts[:5], out.kpts_scores[:5], out.des[:5])
