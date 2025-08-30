# Wrapper Factory

A lightweight factory of local wrappers for feature detection, description, and matching. It downloads third-party implementations and exposes a unified interface so you can test and compare methods quickly.

## Quick start

### 1) Create the environment

```bash
# from the repo root
conda env create -f env.yaml
# activate whatever name is set inside env.yaml, e.g.:
conda activate <env-name>
```

### 2) Configure which wrappers to download

Edit `download_wrappers.yaml` (example below). This file declares where to fetch each method (git, direct URLs) and any post-steps to run (e.g., `pip install -e`, downloading weights).

### 3) Download the wrappers

```bash
python download_wrappers.py
```

### 4) Test in the notebook

```bash
jupyter notebook demo.ipynb
```

Use the notebook to run each wrapper, visualize keypoints/matches, and sanity-check that everything works.


## Supported methods

* **SIFT (pycolmap)** — CPU by default; CUDA optional with a source build
* **SuperPoint** — model + weights (Hugging Face mirror)
* **DISK** — EPFL implementation with editable submodules
* **RIPE** — Fraunhofer HHI implementation + weights
* **DeDoDe** — Parskatt implementation + pretrained checkpoints
* **ALIKED** — Shiaoming implementation (custom ops fix script)

---

## Development roadmap (TODO)

* [ ] Add **SIFT CUDA** path (keep CPU as default)
* [ ] Remove dependencies on `libutils/`, `libutils_md/`, and `utils/`
* [ ] Add **commit hash pinning** when cloning from GitHub (e.g., add a `checkout` step in `submodules`)
* [ ] Find & test **SuperPoint** reference implementation: [https://github.com/rpautrat/SuperPoint](https://github.com/rpautrat/SuperPoint)
* [ ] Remove all hard-coded paths in wrappers
* [ ] Fix **RIPE model** downloading (currently copied from an old save)
* [ ] Re-check **DeDoDe** install; ensure it works without previous environment
* [ ] Add `add_custom_descriptor` to all wrappers
* [ ] Standardize wrapper APIs; introduce a base class (ABC)
* [ ] Add env from `anydesc_old` (merge into `env.yaml`)
* [ ] Standardize code style with **flake8/ruff** and a **pre-commit** config
* [ ] Add project structure in readme
* [ ] add md1500 and GHRB code to run, add benchmarks
* [ ] add GHRB replicable results
* [ ] add md1500 replicable results
* [ ] create final readme


## License and attribution

This repo provides wrappers around third-party research code and models. Each downloaded project remains under its original license. Please review and comply with the licenses of the respective upstream authors.
