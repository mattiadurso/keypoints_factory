# Keypoint Factory

Keypoint Factory is lightweight suite of local wrappers for feature detection and description. It downloads third-party implementations and implements a unified interface allowing to test and compare methods quickly. Exact results might slightly change according to different libraries version, hardware or other unkown factors apparently.

SANDesc is supported but not released yet, thus those parts are commented.

## Quick start

### 1) Create the environment

```bash
# from the repo root
conda env create -f environment.yaml && \  # there might be more dependencies than needed here
conda activate keypoint_factory  # or . ./activate_env.sh if you are lazy
```
Otherwise, the following packages can be manually installed:
```bash
# Install PyTorch with CUDA 12.4 support
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

pip install \
  opencv-python==4.11.0.86 \
  h5py==3.13.0 \
  Pillow==11.1.0 \
  numpy==1.26.4 \
  matplotlib==3.10.1 \
  kornia==0.8.0 \
  pyyaml==6.0.2 \
  joblib==1.4.2 \
  tqdm==4.67.1 \
  pandas==2.2.3 \
  pydegensac==0.1.2

# and optionally 
# nvidia-ml-py: to measure VARM usage
# xformers: to increse speed when using transformer-based models models
pip install \
  nvidia-ml-py==13.580.82 \
  xformers==0.0.29.post2

# to run imc, those are also needed
pip install \
  schema \
  scipy \
  shortuuid \
  jsmin
```
Other dependencies might be related to different methods (e.g., ALIKED). Downloading them with scripts in this repo should install their dependencies automatically.

### 2) Download the wrappers and benchmarks

Edit `download_wrappers.py` to choose which method to download. Empty list means all methods listed in `download_wrappers.yaml`. Then, to download __all__ benchmarks data and/or code run the following. first run.

```bash
python download_wrappers.py && \
bash bash/download_all.sh
```
To download only one benchamrk use the corespondnt bash file in ```bash/```.

### 3) Test in the notebook
In `demo.ipynb` it is possible to test the wrappers on images from the Graz High Reolution Benchmark visualizing keypoints/matches, and sanity-check that everything works.


## Feature Extraction Methods

Currently the following methods are supported with a wrapper.

#### **SIFT**
- **[Paper](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)**: David Lowe — *Distinctive Image Features from Scale-Invariant Keypoints*
- **[Implementation](https://github.com/colmap/pycolmap)**: PyCOLMAP (provides bindings for extracting/matching SIFT features via Python; supports CPU by default (quite slow), optional CUDA).

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
- **Note:** Both -B and -G descriptor models proposed in the paper are available. Repeatability results might slighly change since -G expects images to have edges multple of 14.

#### **ALIKED**
- **[Paper](https://arxiv.org/abs/2304.03608)**: Xiaoming Zhao et al. — *ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation* (2023)
- **[Implementation](https://github.com/Shiaoming/ALIKED)**: Shiaoming’s GitHub repo for the Python version.



## Supported Benchmarks

After downloading the target methods, verifying that the wrapper exists and runs correctly, the following benchmarks are supported:

### Graz High Resolution Benchmark
The **Graz High-Resolution Benchmark (HRB)** is a dataset for evaluating feature extractors and reconstruction models under high-resolution conditions, where compute and memory limits are most stressed. It contains six urban scenes recorded in 4K at 30 fps (sampled at 1 fps) using pre-calibrated cameras. Sparse reconstructions built with COLMAP achieved a mean reprojection error of \~0.97 px across 1.3M 3D points. After pruning view graphs and filtering pairs, the final benchmark includes 1,866 images and 4,413 image pairs. HRB is run at three resolutions, namely 4K (3840×2160), QHD (2560×1440), and FHD (1920×1080). Results are computed following the MegaDepth-1500 protocol.

Depth is estimated using [APD-MVS](https://github.com/whoiszzj/APD-MVS). A sky mask is then generated by combining predictions from [DeepLabv3](https://arxiv.org/abs/1706.05587) and [SAMv2](https://arxiv.org/abs/2408.00714). Specifically, the SAMv2 mask with the highest intersection-over-union (IoU) score relative to the DeepLabv3 sky mask is selected. This refined sky mask is subsequently applied to the raw APD-MVS depth before saving.


The benchmark computes the following metrics.
- **Repeatabilty**  
  Repeatability is defined as the ratio of the number of corresponding keypoints detected in two images to the minimum number of keypoints detected in either image of the pair. Usually reported at 3 pixels.

- **Number of Inliers**  
  The raw count of inlier correspondences found during pose estimation.

- **Area Under the Curve of the Relative Pose Estimation (AUC)**  
  Area under the curve of the percentage of correctly estimated camera poses as a function of angular error thresholds (5°, 10°, 20°).  
  A pose is correct if both translation and rotation errors are below the threshold.

Use the following command to run it.
```bash
python benchmarks/benchmark_parallel.py --benchmark-name ghr # for single test
python benchmarks/graz_high_res/run.py                       # for battery tests
```

Here below we report the results benchmarks. We include also results with SANDesc.

####  Results with a budget of 2048 keypoints

| Method        | AUC@5 (FHD) | AUC@5 (QHD) | AUC@5 (4K) | AUC@10 (FHD) | AUC@10 (QHD) | AUC@10 (4K) |
|---------------|-------------|-------------|------------|--------------|--------------|-------------|
| SuperPoint    | 40.3        | 38.9        | 32.3       | 54.9         | 52.5         | 44.3        |
| ↳w/SANDesc  | **62.5**    | **61.7**    | **57.3**   | **74.7**     | **73.8**     | **69.5**    |
| DISK          | 40.2        | 37.9        | 32.8       | 52.8         | 49.7         | 44.3        |
| ↳w/SANDesc  | **55.3**    | **54.5**    | **48.8**   | **68.9**     | **67.1**     | **61.2**    |
| RIPE          | 54.8        | 47.5        | 33.3       | 67.2         | 59.9         | 44.7        |
| ↳w/SANDesc  | **63.3**    | **63.2**    | **56.9**   | **75.0**     | **74.7**     | **68.5**    |
| ALIKED        | 54.7        | 51.9        | 43.6       | 67.5         | 65.0         | 57.2        |
| ↳w/SANDesc  | **64.4**    | **64.4**    | **61.4**   | **75.8**     | **75.6**     | **73.0**    |
| DeDoDe-B      | 57.1        | OOM         | OOM        | 70.3         | OOM          | OOM         |
| DeDoDe-G      | 57.3        | OOM         | OOM        | 70.8         | OOM          | OOM         |
| ↳w/SANDesc  | **57.4**    | **56.0**    | **52.0**   | **70.7**     | **69.1**     | **65.3**    |
------

### MegaDepth-1500
[MegaDepth-1500](https://arxiv.org/abs/2104.00680) (MD1500)  is a curated subset of the MegaDepth dataset, designed to maintain a uniform covisibility ratio across image pairs, unlike IMC where the distribution is Gaussian-shaped. We assigning a score of 180 degrees (the worst score) when essential matrix recovery fails or the error is greater than 10 degrees. To ensure fairness, we suggest to evaluated methods at keypoint budgets of 2K and 30K. On a RTX 4090 and using 16 cores, SuperPoint completes the benckmark in less than a minute.

The benchmark computes the same metrics as the Graz High-Resolution Benchmark.

Use the following command to run it.
```bash
python benchmarks/benchmark_parallel.py --benchmark-name md  # for single test
python benchmarks/megadepth1500/run.py                       # for battery tests
```

------
### Scannet-1500
[ScanNet-1500](https://arxiv.org/abs/1911.11763) (SC1500) is a curated benchmark derived from the ScanNet dataset, designed to evaluate wide-baseline indoor image matching. Unlike earlier works that select pairs based on temporal proximity or SfM covisibility, SC1500 uses an overlap score computed directly from ground-truth poses and depth, producing significantly more challenging and diverse image pairs. The benchmark consists of 1500 test pairs spanning a range of scene geometries and viewpoints. 

The benchmark computes the same metrics as the Graz High-Resolution Benchmark, except Repeatabilty since absolute poses are missing.

Use the following command to run it.
```bash
python benchmarks/benchmark_parallel.py --benchmark-name sc  # for single test
python benchmarks/scannet1500/run.py                         # for battery tests

```

------

### HPatches
[HPatches](https://arxiv.org/abs/1704.05939) is a benchmark of image sequences with viewpoint or illumination changes. We evaluate on 108 scenes, each with one reference and five target images paired by ground-truth homographies, using a fixed budget of 2048 keypoints and MNN for matching. Wer follow [S-TREK](https://arxiv.org/abs/2308.14598) and [D2-Net](https://arxiv.org/abs/1905.03561) protocol.

The benchmark computes the following metrics.
- **Repeatabilty**  
  Ratio of repeated keypoints between an image pair after applying the known homography, relative to the smaller number of keypoints detected in the two images.

- **Mean Matching Accuracy**  
  Percentage of matches whose reprojection error is within ε pixels.  
  Reported at 1, 2, and 3 pixel thresholds.

- **Matching Score**  
  Ratio of correct matches (within pixel threshold) to the average number of keypoints in the overlapping image area.

- **Homography Accuracy (AUC)**  
  Area under the curve of the percentage of estimated homographies whose corner error is below ε.  
  Corner error = average distance between the four reference corners and the warped target corners.  
  The best score across multiple RANSAC thresholds is reported.


Use the following command to run it.
```bash
python benchmarks/hpatches/hpatches_benchmark.py  # for single test
python benchmarks/hpatches/run_hpatches.py        # for battery tests
```
--- 


### Image Matching Challenge (Phototourism)
[Image Matching Challenge 2021](https://github.com/ubc-vision/image-matching-benchmark) (IMC) evaluates local feature matching in complex real-world settings. We use the Phototourism test set, which contains nine scenes of 100 tourist photos each, captured under diverse cameras, viewpoints, and lighting. Images within a scene are exhaustively compared, and evaluation follows the official protocol: pose accuracy is measured using the AUC of relative pose error at a 5° threshold, with failures assigned when error exceeds 10°. Currently only the stereo matching is supported, both at 2048 and 8000 keypoints budgets.

Despite this benchmark is heavily parallelized, it takes ~1h per method. Nevertheless, it is (arguably) the most complete and exhaustive.

Between the metrics compute by the benchmark, usually oen find in lietarture the following.
- **Repeatabilty**  
  Repeatability is defined as the ratio of the number of corresponding keypoints detected in two images to the minimum number of keypoints detected in either image of the pair. Usually reported at 3 pixels. 

- **Number of Inliers**  
  The raw count of inlier correspondences found during pose estimation.
  
- **Area Under the Curve of the Relative Pose Estimation (AUC)**  
  Area under the curve of the percentage of correctly estimated camera poses as a function of angular error thresholds (5°, 10°). A pose is correct if both translation and rotation errors are below the threshold.

Use the following command to run it.
```bash
python benchmarks/hpatches/imc_benchmark.py  # for single test
python benchmarks/imc/run_imc.py             # for battery tests
```
--- 

### Speed and Memory
The **Speed and Memory** Benchmark evaluates the computational efficiency of feature extraction methods in terms of runtime and memory usage. This benchmark is designed to measure the practical feasibility of using different methods in real-world applications, where resource constraints such as processing time and memory availability are critical.

Metrics Computed:
- **Runtime**
  The average time taken to process a single image or image pair, measured in milliseconds (ms).

- **Memory Usage** 
  The peak memory consumption during feature extraction, measured in megabytes (MB).

Benchmark Details:
The benchmark runs each method on a predefined set of images or image pairs.
It uses a consistent environment to ensure fair comparisons across methods.

Use the following command to run it.
```bash
python benchmarks/speed_and_memory/speed.py
```




### Why this repo?

I couldn’t find a single, unified, and reproducible way to **benchmark feature extractors** quickly. Setting up fair benchmarks shouldn’t steal time from research—so this repo aims to make it fast and consistent.

### Core idea

Wrap each method with a **thin adapter** that standardizes I/O between the model and the benchmark:

* The **wrapper** handles everything the model needs for input:
  normalization, resizing/cropping/padding, color space conversion, etc.
* It produces a **standard output format**, so benchmarks can treat all methods uniformly.
* This makes swapping methods trivial—change a single argument instead of rewriting code.

#### How to add your method

1. **Place your implementation** in `methods/`
2. **Write its wrapper** in `wrappers/`

   * Do all preprocessing here
   * Return outputs in the repo’s standard format (see other wrappers)
3. **Register it** in `wrappers_manager.py`

That’s it, you’re ready to benchmark.

### Benefits

* **Reproducible**: consistent I/O and evaluation across methods
* **Simple to use**: swap methods via a flag
* **Extensible**: add new models with small, focused wrappers


## TODO 

#### MISC
* [ ] final test with all


#### Benchmarks
* [ ] IMC
    - add multiview support, now only stereo
* [ ] add support for matchers (LoFTR, RoMA, etc)
* [ ] maybe unify printing functions names



## License and Attribution

This repo provides wrappers around third-party research code and models. Each downloaded project remains under its original license. Please review and comply with the licenses of the respective upstream authors.

Part of the repo is based on [Emanuele Santellani](https://scholar.google.com/citations?user=1JwKYK8AAAAJ&hl=en)'s work.
