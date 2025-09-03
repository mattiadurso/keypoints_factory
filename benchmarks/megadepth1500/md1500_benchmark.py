import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import sys
sys.path.append('/home/mattia/Desktop/Repos/wrapper_factory/')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gc
import torch
from benchmarks.benchmark_utils import *
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm
from matchers.dual_softmax_matcher import DualSoftMaxMatcher
from matchers.mnn import MNN

DATASET_PATH  = Path('benchmarks/megadepth1500/data/') 
if not DATASET_PATH.exists():
    exit('Dataset not found')

class MegaDepthPoseMNNBenchmark:
    def __init__(self, DATASET_PATH=DATASET_PATH, th = 0.5,
                min_score: float = 0.0, ratio_test: float = 1.0, 
                max_kpts = 2048, matcher = 'mnn'
                ) -> None:
        '''
        Args:
            DATASET_PATH: path to the dataset
            scene_names: list of scene names to be used for the benchmark
            th: threshold for the pose estimation (max reprojection error)
            min_score: minimum score for the matches
            ratio_test: ratio test for the matches
            max_kpts: maximum number of keypoints to be extracted
            matcher: matcher to be used for the computation mnn or dual_softmax
            device: device to be used for the computation
        '''

        # loading data
        with open(DATASET_PATH / 'pairs_calibrated.txt', 'r') as f:
            self.pairs_calibrated = f.read().splitlines()

        self.images_path = DATASET_PATH / 'images'
        self.max_kpts = max_kpts
        self.th = th

        if matcher == 'mnn':
            self.matcher = MNN(min_score=min_score, ratio_test=ratio_test)
            self.matcher.name = 'mnn'
        elif matcher == 'dual_softmax':
            self.matcher = DualSoftMaxMatcher()
            self.matcher.name = 'dual_softmax'
        else:
            raise ValueError("Matcher not supported")
        
    @torch.no_grad()
    def benchmark(self, wrapper, calibrated=True, stats=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fix_rng()

        with torch.no_grad():
            tot_e_t, tot_e_R, tot_e_pose, inlier = [], [], [], []
            thresholds = [5, 10, 20]
            if stats:
                stats_df = {}
            
            for pair in tqdm(self.pairs_calibrated):
                img1, img2, K1, K2, R, t = parse_pair(pair)

                scene_name = img1.split('/')[0]

                im_A_path = self.images_path / img1
                im_B_path = self.images_path / img2

                img_A = Image.open(im_A_path)
                img_B = Image.open(im_B_path)
                img_A = wrapper.img_from_numpy(np.array(img_A))
                img_B = wrapper.img_from_numpy(np.array(img_B))

                # Local features extraction
                out_A = wrapper.extract(img_A, self.max_kpts)
                out_B = wrapper.extract(img_B, self.max_kpts)

                keypoints_A, description_A = out_A.kpts, out_A.des # keypoints_A: [N, 2], description_A: [N, C]
                keypoints_B, description_B = out_B.kpts, out_B.des

                # matcher
                if self.matcher.name == 'mnn':
                    matches = self.matcher.match([description_A], [description_B])[0].matches
                    kpts1 = keypoints_A[matches[:, 0]]
                    kpts2 = keypoints_B[matches[:, 1]]

                elif self.matcher.name == 'dual_softmax':
                    kpts1, kpts2, _ = self.matcher.match(
                        keypoints_A[None], description_A[None],
                        keypoints_B[None], description_B[None],
                        normalize = True, inv_temp=20, threshold = 0.01)

                shuffling = np.random.permutation(np.arange(len(kpts1)))
                kpts1 = kpts1[shuffling]
                kpts2 = kpts2[shuffling]
                try:
                    threshold = self.th
                    if calibrated:
                        norm_threshold = threshold / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                        R_est, t_est, mask = estimate_pose(
                            kpts1.cpu().numpy(),
                            kpts2.cpu().numpy(),
                            K1,
                            K2,
                            norm_threshold,
                            conf=0.99999,
                        ) # None if no pose is found

                    T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  
                    e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                    e_pose = max(e_t, e_R)
                    e_pose = 180 if e_pose > 10 else e_pose # added. this follows IMC21 protocol and sfm can't really recover with >10 error
                    # not really changing results < 10 but making failures explicit
                    num_inliers = np.sum(mask)

                except Exception as e:
                    # print('In pose estimation:', repr(e)) # use to debug
                    e_t, e_R, e_pose = 180, 180, 180 # added. this should not really change the results but now fail is worse than completely off estimation (<180)
                    num_inliers = 0

                tot_e_t.append(e_t)
                tot_e_R.append(e_R)
                tot_e_pose.append(e_pose)
                inlier.append(num_inliers)

                if stats:
                    stats_df[f'{scene_name}_{img1.replace("/", "-")}_{img2.replace("/", "-")}'] = {'idx1': img1, 'idx2': img2, 'scene_name': scene_name, 'e_t': e_t, 'e_R': e_R, 'e_pose': e_pose, 'num_inliers': num_inliers}

                gc.collect()
                torch.cuda.empty_cache()
            
            # stats to csv and save
            if stats:
                stats_df = pd.DataFrame.from_dict(stats_df, orient='index')
                path = Path(f'benchmarks/megadepth1500/stats')
                os.makedirs(path, exist_ok=True)
                stats_df.to_csv(path/f'{wrapper.name}_stats_{timestamp}.csv', index=False)

            # Compute the metrics
            tot_e_pose = np.array(tot_e_pose)
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            
            return {
                "inlier": np.mean(inlier),
                "auc_5":  auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5":  map_5,
                "map_10": map_10,
                "map_20": map_20,
            }, timestamp


if __name__ == '__main__':
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" # for reproducibility
    import torch
    torch.set_float32_matmul_precision('high')

    import json
    import argparse
    from wrappers_manager import wrappers_manager

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wrapper-name', type=str, default='disk')
    parser.add_argument('--run-tag', type=str, default=None)
    parser.add_argument('--ratio-test', type=float, default=1.0)
    parser.add_argument('--min-score', type=float, default=-1.0)
    parser.add_argument('--max-kpts', type=int, default=2048)
    parser.add_argument('--matcher', type=str, default='mnn')
    parser.add_argument('--th', type=float, default=0.5) # 1 if max kpts = 2048 else 0.25
    parser.add_argument('--custom-desc', type=str, default=None)
    parser.add_argument('--stats', type=bool, default=True)
    args = parser.parse_args()

    device = args.device
    wrapper_name = args.wrapper_name
    ratio_test = args.ratio_test
    min_score = args.min_score
    max_kpts = args.max_kpts
    th = args.th
    matcher = args.matcher
    custom_desc = args.custom_desc

    # Define the wrapper
    wrapper = wrappers_manager(name=wrapper_name, device=args.device)

    if custom_desc is not None:
        #  Eventually add my descriptors
        weights = torch.load(custom_desc, weights_only=False)
        config = weights['config']
        model_config = {'ch_in': config['model_config']['unet_ch_in'],
                        'kernel_size': config['model_config']['unet_kernel_size'],
                        'activ': config['model_config']['unet_activ'],
                        'norm': config['model_config']['unet_norm'],
                        'skip_connection': config['model_config']['unet_with_skip_connections'],
                        'spatial_attention': config['model_config']['unet_spatial_attention'],
                        'third_block': config['model_config']['third_block'],
                        }
        # print('model config for model.pth:', model_config)

        sys.path.append('/home/mattia/Desktop/Repos/strek_reference')
        from model.network_descriptor import SANDesc
        network = SANDesc(**model_config).eval().to(device) 

        weights = torch.load(custom_desc, weights_only=False)
        network.load_state_dict(weights['state_dict'])
        wrapper.add_custom_descriptors(network)
        wrapper.name = f'{wrapper.name}+SANDesc'
        print(f'Using custom descriptors from {custom_desc}.\n')

       
    if matcher=='mnn':
        key = f'{wrapper.name} ratio_test_{ratio_test}_min_score_{min_score}_th_{th}_{matcher} {max_kpts}'
    elif matcher=='dual_softmax':
        key = f'{wrapper.name} th_{th}_{matcher} {max_kpts}'
    
    if args.run_tag is not None:
        key = f'{key} {args.run_tag}'

    print(f'Running benchmark for {key}...')

    # create if not exists
    results_path = Path('benchmarks/megadepth1500/results')
    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(results_path/'results.json'):
        with open(results_path/'results.json', 'w') as f:
            json.dump({}, f)
        f.close()

    with open(results_path/'results.json', 'r') as f:
        data = json.load(f)

    if key in data:
        results = data[key]
        warnings.warn("A similar run already exists.", UserWarning)

    f.close()
    
    # # Define the benchmark
    benchmark = MegaDepthPoseMNNBenchmark(
        ratio_test=ratio_test, 
        min_score=min_score, 
        max_kpts=max_kpts, 
        th=th,
        matcher=matcher,
        )
    
    # Run the benchmark
    results, timestamp = benchmark.benchmark(wrapper, stats=args.stats)
    print_metrics(wrapper, results)
    print('------------------------------------------------------------------------------')

    # Save the results
    data[f'{key} {timestamp}'] = results
    with open(results_path/'results.json', 'w') as f:
        json.dump(data, f)
    f.close()


