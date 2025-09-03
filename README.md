# Keypoint Factory

A lightweight factory of local wrappers for feature detection and description. It downloads third-party implementations and exposes a unified interface so it iis possible to test and compare methods quickly. Exact results might slightly change according to different libraries version, hardware or unkown factor apparently.

Code to dowload and run Megadepth-1500 and the Graz High Reoslution Benchmark is provided in `bash` and `benchmarks` folders, respectively.

## Quick start

### 1) Create the environment

```bash
# from the repo root
conda env create -f env.yaml
# activate whatever name is set inside env.yaml, e.g.:
conda activate <env-name>
```

### 2) Download the wrappers

Edit `download_wrappers.py` to choose which metrhod to download, by default all of the listed in `download_wrappers.yaml`. Then run it.

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


## Development roadmap (TODO)

* [ ] Add env from `anydesc_old` (merge into `env.yaml`)
* [ ] Standardize code style with **flake8/ruff** and a **pre-commit** config
* [x] add md1500 
    * [x] bench and battery code
    * [x] data download
    * [x] support to sandesc / custom descrptor
    * [x] run and save results
    * [ ] rerun dedode with normalization in wrapper since they dont do that
* [ ] add GHRB 
    * [x] bench and battery code
    * [x] data download
    * [x] support to sandesc / custom descrptor
    * [ ] run and save results
    * [ ] rerun dedode with normalization in wrapper since they dont do that
* [ ] clean and comment code
* [ ] find a good couple of images to show in demo from clock tower
---
* [ ] Add more methods. 
    - maybe also superglue and lightglue
    - maybe RDD
    - Maybe add dense methods like loftr, roma, aspanformer, eff loftr etc.



## License and Attribution

This repo provides wrappers around third-party research code and models. Each downloaded project remains under its original license. Please review and comply with the licenses of the respective upstream authors.
