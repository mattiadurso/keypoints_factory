# Keypoint Factory

A lightweight factory of local wrappers for feature detection and description. It downloads third-party implementations and exposes a unified interface so it is possible to test and compare methods quickly. Exact results might slightly change according to different libraries version, hardware or unkown factor apparently.

Code to download the benchmarks' data is provided in `bash` folder.

SANDesc is supported but not released yet, thus those parts are commented.

## Quick start

### 1) Create the environment

```bash
# from the repo root
conda env create -f environment.yaml && \
conda activate keypoint_factory
```

### 2) Download the wrappers and benchmarks

Edit `download_wrappers.py` to choose which method to download. Empty list means all methods listed in `download_wrappers.yaml`. Then to download benchmarks data and/or code run the following.

```bash
python download_wrappers.py && \
bash bash/download_all.sh
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
- **[Implementation](https://github.com/Parskatt/DeDoDe)**: Parskatt’s GitHub repository with code, training scripts, and pretrained weights.
- **Note:** Both -B and -G descriptor models proposed in the paper are available.

#### **ALIKED**
- **[Paper](https://arxiv.org/abs/2304.03608)**: Xiaoming Zhao et al. — *ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation* (2023)
- **[Implementation](https://github.com/Shiaoming/ALIKED)**: Shiaoming’s GitHub repo for the Python version.

## Supported Benchmarks

After downloading the target methods, verifying that the wrapper exists and runs correctly, and downloading the benchmark data, the following benchmarks are supported:

#### Graz High Resolution Benchmark
The **Graz High-Resolution Benchmark (HRB)** is a dataset for evaluating feature extractors and reconstruction models under high-resolution conditions, where compute and memory limits are most stressed. It contains six urban scenes recorded in 4K at 30 fps (sampled at 1 fps) using pre-calibrated cameras. Sparse reconstructions built with COLMAP achieved a mean reprojection error of \~0.97 px across 1.3M 3D points. After pruning view graphs and filtering pairs, the final benchmark includes 1,866 images and 4,413 image pairs. HRB supports evaluation at three resolutions, namely 4K (3840×2160), QHD (2560×1440), and FHD (1920×1080). Results are computed following the MegaDepth-1500 protocol.

Use the following command to run it.
```bash
python benchmarks/graz_high_res/run_ghr.py
```

Here below we report the results benchmarks. We include also results with SANDesc.

####  Results with a budget of 2048 keypoints

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
------

### MegaDepth-1500
[MegaDepth-1500](https://arxiv.org/abs/2104.00680) (MD1500)  is a curated subset of the MegaDepth dataset, designed to maintain a uniform covisibility ratio across image pairs, unlike IMC where the distribution is Gaussian-shaped. We follow the standard MD1500 evaluation protocol, assigning a score of 180 degrees when fundamental matrix recovery fails or the error is greater than 10 degrees. To ensure fairness, results are computed for all evaluated methods at keypoint budgets of 2K and 30K.

Use the following command to run it.
```bash
python benchmarks/megadepth1500/run_md1500.py
```

------

### HPatches
[HPatches](https://arxiv.org/abs/1704.05939) is a benchmark of image sequences with viewpoint or illumination changes. We evaluate on 108 scenes, each with one reference and five target images paired by ground-truth homographies, using a fixed budget of 2048 keypoints and MNN for matching. 


Use the following command to run it.
```bash
python benchmarks/hpatches/run_hpatches.py
```
--- 


### Image Matching Challenge (Phototourism)
[Image Matching Challenge 2021](https://github.com/ubc-vision/image-matching-benchmark) (IMC) evaluates local feature matching in complex real-world settings. We use the Phototourism test set, which contains nine scenes of 100 tourist photos each, captured under diverse cameras, viewpoints, and lighting. Images within a scene are exhaustively compared, and evaluation follows the official protocol: pose accuracy is measured using the AUC of relative pose error at a 5° threshold, with failures assigned when error exceeds 10°.

Use the following command to run it.
```bash
python benchmarks/imc/run_imc.py
```
--- 
---
### TODO 
* [ ] add pretrained kpts in all wrappers
    - sift
    - disk
    - aliked
    - ripe
* [ ] double check memory usage in speed table in the paper
    - when in place, it shoule be able to extract N features, and eventually change matching params from the second run
* [ ] remove thos two images errors in ghr
* [ ] add speed and memory testing scripts
* [ ] add code to run these wrappers to populate colmap database
* [ ] add eth3D?
* [ ] add aachen day and night?
* [ ] re run all tests before make them public?
* [ ] docuemnt and explain repo logic/functioning
* [ ] update env file at the end
* [ ] unify displaying functions names?

## License and Attribution

This repo provides wrappers around third-party research code and models. Each downloaded project remains under its original license. Please review and comply with the licenses of the respective upstream authors.

Part of the repo is based on [Emanuele Santellani](https://scholar.google.com/citations?user=1JwKYK8AAAAJ&hl=en) work.

