# Keypoint Factory

A lightweight factory of local wrappers for feature detection and description. It downloads third-party implementations and exposes a unified interface so it iis possible to test and compare methods quickly. Exact results might slightly change according to different libraries version, hardware or unkown factor apparently.

Code to dowload and run Megadepth-1500 and the Graz High Reoslution Benchmark is provided in `bash` and `benchmarks` folders, respectively.

SANDesc is supported but not released yet, thus those parts are commented.

## Quick start

### 1) Create the environment

```bash
# from the repo root
conda env create -f environment.yaml && \
conda activate keypoint_factory
```

### 2) Download the wrappers

Edit `download_wrappers.py` to choose which method to download. Empty list means all methods listed in `download_wrappers.yaml`. Then run it.

```bash
python download_wrappers.py
```

### 3) Test in the notebook
In `demo.ipynb` it is possible to test the wrappers on images from the Graz High Reolution Benchmark visualizing keypoints/matches, and sanity-check that everything works.


### Feature Extraction Methods
Currently the following methods are supported with a wrapper.

#### **SIFT**
- **[Paper](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)**: David Lowe — *Distinctive Image Features from Scale-Invariant Keypoints*
- **[Implementation](https://github.com/colmap/pycolmap)**: PyCOLMAP GitHub (provides bindings for extracting/matching SIFT features via Python; supports CPU by default, optional CUDA).

#### **SuperPoint**
- **[Paper](https://arxiv.org/abs/1712.07629)**: Daniel DeTone, Tomasz Malisiewicz & Andrew Rabinovich — *SuperPoint: Self-Supervised Interest Point Detection and Description* (CVPR 2018 workshop; arXiv 2017)
- **[Implementation](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/)**: From SuperGlue GitHub repository.

#### **DISK**
- **[Paper](https://arxiv.org/abs/2006.13566)**: Michał J. Tyszkiewicz, Pascal Fua & Eduard Trulls — *DISK: Learning Local Features with Policy Gradient* (NeurIPS 2020)
- **[Implementation](https://github.com/cvlab-epfl/disk)**: Official EPFL CVLAB GitHub repository containing training and inference code.

#### **RIPE**
- **[Paper](https://arxiv.org/abs/2507.04839)**: Fraunhofer HHI team — *RIPE: Reinforcement Learning on Unlabeled Image Pairs* (ICCV 2025)
- **[Implementation](https://github.com/fraunhoferhhi/RIPE)**: Fraunhofer HHI GitHub repository.

#### **DeDoDe**
- **[Paper](https://arxiv.org/abs/2308.08479)**: Johan Edstedt, Georg Bökman, Mårten Wadenbäck & Michael Felsberg — *DeDoDe: Detect, Don’t Describe — Describe, Don’t Detect for Local Feature Matching* (arXiv 2023)
- **[Implementation](https://github.com/Parskatt/DeDoDe)**: Parskatt’s GitHub repository with code, training scripts, and pretrained weights :contentReference[oaicite:6]{index=6}

#### **ALIKED**
- **[Paper](https://arxiv.org/abs/2304.03608)**: Xiaoming Zhao et al. — *ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation* (2023)
- **[Implementation](https://github.com/Shiaoming/ALIKED)**: Shiaoming’s GitHub repo for the Python version.

## Results

Here below we report the results when running the benchmarks. We include also results with SANDesc.

### Megadepth-1500

| Method        | AUC@5 (2048) | AUC@10 (2048) | AUC@5 (30k) | AUC@10 (30k) |
|---------------|--------------|---------------|-------------|--------------|
| SuperPoint    | 28.9         | 44.9          | 14.4        | 28.5         |
| ↳ w/ SANDesc  | **42.2**     | **58.6**      | **34.7**    | **52.0**     |
| DISK          | 34.7         | 51.4          | 42.6        | 57.7         |
| ↳ w/ SANDesc  | **36.6**     | **54.2**      | **45.1**    | **60.9**     |
| RIPE          | 42.3         | 58.0          | 37.1        | 53.2         |
| ↳ w/ SANDesc  | **42.9**     | **58.7**      | **42.4**    | **58.3**     |
| ALIKED        | 40.7         | 56.6          | 38.2        | 54.2         |
| ↳ w/ SANDesc  | **43.5**     | **60.1**      | **44.0**    | **59.6**     |
| DeDoDe-B      | 42.9         | 59.7          | 50.8        | 65.9         |
| DeDoDe-G      | **46.4**     | **63.2**      | **55.3**    | **70.9**     |
| ↳ w/ SANDesc  | 45.2         | 61.8          | 52.8        | 67.2         |

### HRB Results (2048 keypoints)

| Method        | AUC@5 (FHD) | AUC@5 (QHD) | AUC@5 (4K) | AUC@10 (FHD) | AUC@10 (QHD) | AUC@10 (4K) |
|---------------|-------------|-------------|------------|--------------|--------------|-------------|
| SuperPoint    | 40.3        | 38.9        | 32.3       | 54.9         | 52.5         | 44.3        |
| ↳ w/ SANDesc  | **62.5**    | **61.7**    | **57.3**   | **74.7**     | **73.8**     | **69.5**    |
| DISK          | 40.2        | 37.9        | 32.8       | 52.8         | 49.7         | 44.3        |
| ↳ w/ SANDesc  | **55.3**    | **54.5**    | **48.8**   | **68.9**     | **67.1**     | **61.2**    |
| RIPE          | 54.8        | 47.5        | 33.3       | 67.2         | 59.9         | 44.7        |
| ↳ w/ SANDesc  | **63.3**    | **63.2**    | **56.9**   | **75.0**     | **74.7**     | **68.5**    |
| ALIKED        | 54.7        | 51.9        | 43.6       | 67.5         | 65.0         | 57.2        |
| ↳ w/ SANDesc  | **64.4**    | **64.4**    | **61.4**   | **75.8**     | **75.6**     | **73.0**    |
| DeDoDe-B      | 57.1        | OOM         | OOM        | 70.3         | OOM          | OOM         |
| DeDoDe-G      | 57.3        | OOM         | OOM        | 70.8         | OOM          | OOM         |
| ↳ w/ SANDesc  | **57.4**    | **56.0**    | **52.0**   | **70.7**     | **69.1**     | **65.3**    |

### Speed Comparison
The table below compares per image processing time in milliseconds for keypoint detection and description under a budget of 2048 keypoints, and reports the model size, in millions of parameters, in the corresponding column. All images were processed in FHD on an NVIDIA RTX 4090 with 24GB. 

Overall, methods such as DISK, SuperPoint, ALIKED, and RIPE run slower when paired with SANDesc, since all-in-one pipelines reuse intermediate features to compute descriptors, whereas SANDesc operates directly on raw images. 
By contrast, with the decoupled DeDoDe methods, SANDesc remains competitive: its accuracy matches DeDoDe-G and exceeds DeDoDe-B, while its runtime is close to DeDoDe-B and faster than DeDoDe-G.
SANDesc alone requires approximately 87 ms on our hardware.

| Method      | Size (M) | Speed Orig (ms) | Speed Ours (ms) | VRAM Orig (GB) | VRAM Ours (GB) |
|-------------|----------|-----------------|-----------------|----------------|----------------|
| DISK        | 0.26     | 62.1 ± 0.1      | 135.2 ± 0.2     | 6.96           | 7.01           |
| SuperPoint  | 0.30     | 18.3 ± 0.2      | 114.8 ± 0.2     | 3.50           | 7.18           |
| ALIKED      | 0.32     | 19.3 ± 0.3      | 114.2 ± 0.2     | 3.89           | 5.76           |
| RIPE        | 0.24     | 175.9 ± 0.2     | 275.0 ± 0.2     | 6.55           | 8.42           |
| DeDoDe-B    | 15.1     | 181.5 ± 0.1     | 189.0 ± 0.2     | 8.11           | 5.78           |
| DeDoDe-G    | 323.2    | 316.8 ± 0.4     | 189.0 ± 0.2     | 9.31           | 5.78           |


--- 
### TODO after release
* [ ] Add more methods. 
    - maybe also superglue and lightglue
    - maybe RDD
    - Maybe add dense methods like loftr, roma, aspanformer, eff loftr etc.



## License and Attribution

This repo provides wrappers around third-party research code and models. Each downloaded project remains under its original license. Please review and comply with the licenses of the respective upstream authors.
