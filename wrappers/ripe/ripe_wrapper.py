from pathlib import Path
import sys
file_path = Path(__file__).resolve().parent 
sys.path.append(str(file_path))
sys.path.append(str(file_path / "ripe"))
sys.path.append(str(file_path.parent))  # Add parent directory to sys.path
# print(sys.path)

import gc
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Union, List
from libutils.utils_2D import grid_sample_nan
from utils.method_wrapper import MethodOutput
from torchvision import transforms

import cv2
import kornia.feature as KF
import kornia.geometry as KG
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import decode_image

from ripe import vgg_hyper
from ripe.utils.utils import cv2_matches_from_kornia, resize_image, to_cv_kpts

from torchinfo import summary
from libutils.utils_2D import grid_sample_nan
from utils.method_wrapper import MethodOutput


class RIPEWrapper():
    def __init__(self, device: str = 'cuda'):
        self.name = 'ripe'
        self.device = torch.device(device)

        model_path = file_path / "ripe/ckpt/ripe_weights.pth"
        self.model = vgg_hyper(model_path).to(self.device)
        self.model.eval()

        self.descriptor_network = None
        self.amp_enabled = False  # Automatic Mixed Precision
        self.use_ripe_desc = True
    
    def add_custom_descriptors(self, model):
        self.descriptor_network = model.to(self.device).eval()
        self.use_ripe_desc = False # disable RIPE descriptors

        print(f'Adding custom descriptors to {self.name}.')


    def img_from_numpy(self, img: np.ndarray) -> Union[Tensor, np.ndarray]:
        # ? crop image such that dimensions are multiple of 16
        factor = 16
        img = img[:img.shape[0] // factor * factor, :img.shape[1] // factor * factor]
        img_out = torch.from_numpy(img.copy()).permute(2, 0, 1).to(self.device) / 255.
        return img_out


    @torch.inference_mode()
    def extract(self, x, max_kpts: int = 2048, kpts=None) -> MethodOutput:
        x = x if x.dim() == 4 else x[None]

        # if torch image too big > 4M activate amp
        if (x.shape[-1]*x.shape[-2]) > 4e6:
            print(f"Using AMP for {self.name} due to large image size: {x.shape[-1]}x{x.shape[-2]}")
            self.amp_enabled = True
        else:
            self.amp_enabled = False

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp_enabled):
            kpts, des, score = self.model.detectAndCompute(x, threshold=0.5, top_k=max_kpts)
            des_vol = None
    

        if self.descriptor_network is not None:
            des = None
            if self.amp_enabled:
                gc.collect()  # clear GPU memory before computing descriptors
                torch.cuda.empty_cache()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp_enabled):
                des_vol = self.descriptor_network(x)
            des = grid_sample_nan(kpts[None], des_vol, mode='nearest')[0][0].T
            des_vol = None  # clear memory
            
            if self.amp_enabled:
                gc.collect()  # clear GPU memory before computing descriptors
                torch.cuda.empty_cache()

        output = MethodOutput(
            kpts=kpts,
            kpts_scores=score,
            des=des,
            # des_vol=des_vol
        )

        return output
    

    def match(self, des0: List[Tensor], des1: List[Tensor]):
        return self.matcher.match(des0, des1)


if __name__ == "__main__":
    # Example usage
    wrapper = RIPEWrapper(device='cpu')
    # img = torch.rand(3, 512, 512).cuda()  # Random image tensor

    # # output = wrapper.extract(img, max_kpts=30_000)
    # print(f"Extracted {output.kpts.shape}")
    
    # if output.des is not None:
    #     print(f"Extracted descriptors shape: {output.des.shape}")
    
    from torchinfo import summary
    print(summary(wrapper.model))