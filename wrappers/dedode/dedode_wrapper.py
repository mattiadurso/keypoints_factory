from pathlib import Path
import sys
file_path = Path(__file__).resolve().parent
sys.path.append('wrappers/dedode')

import gc
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Union, List
from libutils.utils_2D import grid_sample_nan
from utils.method_wrapper import MethodOutput
from torchvision import transforms

from DeDoDe import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G # installed from Repos/DeDoDe

class DeDoDeWrapper():
    def __init__(self, descriptor_G:bool= False, device: str = 'cuda:0'):
        self.name = 'dedode'
        self.device = device
        detector_path =     file_path / "dedode_detector_L.pth" 
        descriptor_G_path = file_path / "dedode_descriptor_G.pth" 
        descriptor_B_path = file_path / "dedode_descriptor_B.pth"
        self.detector = dedode_detector_L(weights = torch.load(detector_path, map_location = device), device = device)
        self.descriptor_G = descriptor_G
        if descriptor_G:
            self.descriptor = dedode_descriptor_G(weights = torch.load(descriptor_G_path, map_location = device), device = device)
        else:
            self.descriptor = dedode_descriptor_B(weights = torch.load(descriptor_B_path, map_location = device), device = device)
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.dtype = next(iter(self.detector.parameters())).dtype
        self.wrapper = None
        self.descriptor_network = None

    def img_from_numpy(self, img: np.ndarray) -> Union[Tensor, np.ndarray]:
        # ? crop image such that dimensions are multiple of 16 or 14
        factor = 16 #14 if (self.descriptor_G and self.descriptor_network is None) else 16
        img = img[:img.shape[0] // factor * factor, :img.shape[1] // factor * factor]
        img_out = torch.from_numpy(img.copy()).permute(2, 0, 1) / 255.
        return img_out.to(self.dtype).to(self.device)

    @torch.inference_mode()
    def extract(self, x, max_kpts: int = 2048) -> MethodOutput:
        x = x if x.dim() == 4 else x[None]
        batch = {"image": self.normalizer(x)}
        kpts = torch.zeros((1, 0, 2), device=x.device).to(self.dtype)

        with torch.amp.autocast(self.device, enabled=False): 
            # detector
            if self.detector is not None:
                out = self.detector.detect(batch, num_keypoints=max_kpts)
                kpts, scores = out['keypoints'], out['confidence']
                kpts_pix = self.to_pixel_coords(kpts, x.shape[-2], x.shape[-1]).to(self.dtype)
            else:
                kpts_pix = torch.zeros((1, 0, 2), device=x.device).to(self.dtype)
                scores = torch.zeros((1, 0), device=x.device).to(self.dtype)

            ## descriptors
            # no descriptor network
            if self.descriptor is None and self.descriptor_network is None:
                des, des_vol = None, None
            # dedode descriptor
            elif self.descriptor is not None and self.descriptor_network is None:
                if self.descriptor_G:
                    # need to pad to make it multiple of 14
                    h1, w1 = x.shape[-2], x.shape[-1]
                    h_pad = (14 - h1 % 14) % 14
                    w_pad = (14 - w1 % 14) % 14
                    x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
                    batch = {"image": self.normalizer(x)}

                out = self.descriptor.describe_keypoints(batch, kpts)
                des = out['descriptions'][0] 
                des_vol = out['description_grid']

                if self.descriptor_G:
                    # need to crop back to original size
                    des_vol = des_vol[:, :, :h1, :w1]
                    
            # custom descriptor network
            elif self.descriptor_network is not None:
                des_vol = self.descriptor_network(x)
                des = grid_sample_nan(kpts_pix[None], des_vol, mode='bilinear')[0][0].permute(1,2,0)[0]
            else:
                raise ValueError("Error in dedode descriptors.")


        output = MethodOutput(
            kpts=kpts_pix[0],
            kpts_scores=scores[0],
            des=des,
            des_vol=des_vol
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


    def match(self, des0: List[Tensor], des1: List[Tensor]):
        return self.matcher.match(des0, des1)
    

    def add_custom_descriptors(self, model):
        self.descriptor_network = model.to(self.dtype).to(self.device)
        self.descriptor = None
        gc.collect()
        torch.cuda.empty_cache()

        print(f'Adding custom descriptors to {self.name} with dtype {self.dtype} and device {self.device}')

        

# '''
# "detector": {
#     "L-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
#     "L-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_C4.pth",
#     "L-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_SO2.pth",
#     "L-C4-v2": "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth",
# },
# "descriptor": {
#     "B-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
#     "B-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_descriptor_setting_C.pth",
#     "B-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_C.pth",
#     "G-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
#     "G-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_C4_Perm_descriptor_setting_C.pth",
#     "G-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_SO2_Spread_descriptor_setting_C.pth",
# }
# '''
# from kornia.feature import DeDoDe
# class DeDoDeWrapper_kornia():
#     def __init__(self, v2=True, use_dino=False, device: str = 'cuda:0'):
#         self.name = 'dedodev2' if v2 else 'dedode'
#         self.device = device
#         self.dedode = DeDoDe.from_pretrained(
#                 detector_weights="L-C4-v2" if v2 else "L-upright", 
#                 descriptor_weights="G-upright" if use_dino else "B-upright",
#             ).eval().to(device)
        
#         self.descriptor_network = None

#     def img_from_numpy(self, img: np.ndarray) -> Union[Tensor, np.ndarray]:
#         # ? crop image such that dimensions are multiple of 16
#         # img = img[:img.shape[0] // 16 * 16, :img.shape[1] // 16 * 16]
#         img_out = torch.from_numpy(img.copy()).permute(2, 0, 1).to(self.device) / 255.
#         return img_out[None]


#     def extract(self, x, max_kpts: int = 2048) -> MethodOutput:
#         kpts, scores, des = self.dedode(x, n=max_kpts, apply_imagenet_normalization=True)

#         if self.descriptor_network is not None:
#             des_vol = self.descriptor_network(x)
#             des = grid_sample_nan(kpts[None], des_vol, mode='nearest')[0][0].permute(1,2,0)
                
#         output = MethodOutput(
#             kpts=kpts[0],
#             kpts_scores=scores[0],
#             des=des[0],
#         )

#         return output