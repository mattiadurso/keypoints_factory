import sys
from pathlib import Path

# Dynamically get the project root directory
project_root = Path(__file__).resolve().parents[2]  # Adjust based on your folder structure
print(f'Project root: {project_root}')
sys.path.append(str(project_root))

from abc import ABC
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch as th
from torch import Tensor


from libutils.utils_2D import grid_sample_nan
from wrappers.DISK.network_descriptor_original import Unet
from wrappers.DISK.disk import DISK
from utils.method_wrapper import MethodWrapper, MethodOutput


class DiskWrapper(MethodWrapper, ABC):
    def __init__(self, device: str = 'cuda:0', use_disk_descriptors:bool = True, inference=True, weights_path=None) -> None:
        super().__init__(name='DISK')
        self.use_disk_descriptors = use_disk_descriptors
        disk = DISK(window=8, desc_dim=128, use_disk_descriptors=use_disk_descriptors)
        # load weights
        weights_path = 'wrappers/disk/disk/depth-save.pth'
        
        state_dict = th.load(Path(weights_path), map_location='cpu', weights_only=False)
        disk.load_state_dict(state_dict['extractor'])

        self.disk = disk.to(device)
        self.device = device
        
        if not use_disk_descriptors:
            print('Pruning the descriptors...')
            self.prune_descriptors()
        
        if inference:
            # turn grad off
            self.disk.eval()
            for param in self.disk.parameters():
                param.requires_grad = False
            
            # print('Warming up the detector...')
            # self.disk = th.compile(self.disk)
            # for _ in range(10):
            #     x = th.rand(1, 3, 448, 448).to(device)
            #     out = self.disk.features(x)
            #     del x, out
        
        self.descriptor_network = None
        self.dtype = next(iter(self.disk.parameters())).dtype
        print(f'Using DISK with device {self.device} and dtype {self.dtype}')
    

    def prune_descriptors(self):
        self.disk.unet.path_up[-1].conv[-1].weight.data = self.disk.unet.path_up[-1].conv[-1].weight.data[-1:,...]
        self.disk.unet.path_up[-1].conv[-1].bias.data   = self.disk.unet.path_up[-1].conv[-1].bias.data[-1:,...]
        self.use_disk_descriptors = False
        self.disk.use_disk_descriptors = False


    def img_from_numpy(self, img: np.ndarray) -> Union[Tensor, np.ndarray]:
        # ? crop image such that dimensions are multiple of 16
        img = img[:img.shape[0] // 16 * 16, :img.shape[1] // 16 * 16]
        img_out = th.from_numpy(img.copy()).permute(2, 0, 1).to(self.device) / 255.
        return img_out.to(self.dtype)

    @th.inference_mode()
    def _extract(self, img: Tensor, max_kpts: int) -> MethodOutput:
        with th.amp.autocast(self.device, enabled=False):
            # desc_vol is None if use_disk_descriptors is False
            features, des_vol = self.disk.features(img[None], kind='nms', window_size=5, cutoff=0., n=max_kpts)
            kpts = features[0].kp
            kpts_scores = features[0].kp_logp
            if self.use_disk_descriptors:
                des = features[0].desc
            
            #l2 normalize des_vol
            des_vol = th.nn.functional.normalize(des_vol, dim=1) if des_vol is not None else None

            # ? order keypoints and descriptors by scores
            order = kpts_scores.argsort(descending=True)
            kpts_scores = kpts_scores[order]
            kpts = kpts[order]
            if self.use_disk_descriptors:
                des = des[order]

            # ? only keep the first max_kpts keypoints
            kpts_scores = kpts_scores[:max_kpts]
            kpts = kpts[:max_kpts]

            # if using DISK descriptors
            if self.descriptor_network is None: 
                des = des[:max_kpts] if self.use_disk_descriptors else None
            # if using my custom descriptor 
            else: 
                des_vol = self.descriptor_network(img[None])
                des = grid_sample_nan(kpts[None], des_vol, mode='nearest')[0].squeeze(0).T

        if not self.use_disk_descriptors and self.descriptor_network is None:
            des = None
            des_vol = None

        output = MethodOutput(
            kpts=kpts,
            kpts_scores=kpts_scores,
            des=des,
            des_vol=des_vol
        )

        return output


    def _extract_with_additional_descriptors_from_keypoints(self, img: Union[Tensor, np.ndarray],
                                                            max_kpts: Union[float, int], kpts_additional: Tensor
                                                            ) -> Tuple[MethodOutput, Tensor]:
        output = self._extract(img, max_kpts)

        des, _ = grid_sample_nan(kpts_additional[None], output.des_vol[None], mode='nearest')  # B,des_dim,max_n_keypoints
        des = des[0, :, :].T  # n_kpts,des_dim
        des = th.nn.functional.normalize(des, dim=-1)  # n_kpts,des_dim

        return output, des


    def add_custom_descriptors(self, model):
        self.descriptor_network = model.to(self.dtype).to(self.device)
        print(f'Adding custom descriptors to {self.name} with dtype {self.dtype} and device {self.device}')
        self.prune_descriptors()  # ensure that the descriptors are pruned if custom descriptors are used

# main+

if __name__ == '__main__':
    from torchinfo import summary

    disk = DiskWrapper(device='cpu', use_disk_descriptors=False)

    from network_descriptor_self import Unet_with_SATT
    network = Unet_with_SATT()
    disk.descriptor_network = network

    summary(disk.disk)

    out = disk._extract(th.rand(3, 448, 448), 100)
    print(out)