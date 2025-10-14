# Code adapted from https://github.com/cvg/LightGlue/blob/main/benchmark.py

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

abs_root_path = Path(__file__).parents[2]
sys.path.append(str(abs_root_path))

import glob
import json
import time
import torch
import logging
import numpy as np


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    logger.info(
        "tqdm not found, you'll get no progress bars. Install it with `pip install tqdm`."
    )
    from benchmarks_2D.utils_benchmark import fake_tqdm as tqdm

try:
    import pynvml
except ImportError:
    logger.info(
        "pynvml is not available, VRAM usage will not be logged. You can install it with pip install nvidia-ml-py"
    )
    pynvml = None


def get_vram_usage(gpu_index=0):
    if pynvml is None:
        # logger.warning(
        #     "pynvml is not available, cannot get VRAM usage. You can install it with pip install nvidia-ml-py"
        # )
        return 0
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / 1024**2  # MB


def measure(
    wrapper, image_paths, max_kpts=2048, device="cuda", r=100, scaling_factor=1.0
):
    device = torch.device(device)
    timings = np.zeros((r, 1))
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    data = []
    # read images, using GHR images
    for img_path in tqdm(image_paths, desc="Reading images"):
        # img = Image.open(img_path).convert("RGB")
        # H, W = img.size
        # newH, newW = int(H * scaling_factor), int(W * scaling_factor)
        # img = img.resize((newH, newW), Image.LANCZOS)
        # img = wrapper.img_from_numpy(np.array(img)).cpu()
        img = wrapper.load_image(img_path, scaling=scaling_factor).cpu()
        H, W = img.shape[-2:]

        data.append(img)

    # warmup
    for _ in tqdm(range(5), desc="Warming up"):
        for img in data:
            img_ = img.to(device)
            _ = wrapper.extract(img_, max_kpts)

    # measurements
    vram = []
    with torch.inference_mode():
        for rep in tqdm(range(r), desc="Measuring"):
            if device.type == "cuda":
                for img in data:
                    img_ = img.to(device)
                    starter.record()
                    _ = wrapper.extract(img_, max_kpts)
                    ender.record()
                    vram.append(get_vram_usage())
                # sync gpu
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
            else:
                start = time.perf_counter()
                for img in data:
                    img_ = img.to(device)
                    _ = wrapper.extract(img_, max_kpts)
                curr_time = (time.perf_counter() - start) * 1e3
                vram.append(0)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / r  # in milliseconds
    std_syn = np.std(timings)
    vram = np.max(vram)
    return {"mean": mean_syn, "std": std_syn, "image_size": (H, W), "vram": vram}


if __name__ == "__main__":
    import argparse
    from wrappers_manager import wrappers_manager

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wrapper-name", type=str, default="disk-kornia")
    parser.add_argument(
        "--scaling-factor", type=float, default=1.0, help="Scaling factor for images"
    )
    parser.add_argument(
        "--max-kpts", type=int, default=2048, help="Max keypoints to detect"
    )
    parser.add_argument(
        "--r", type=int, default=100, help="Number of repetitions for timing"
    )
    parser.add_argument(
        "--nimages", type=int, default=25, help="Number of images to process"
    )

    args = parser.parse_args()

    device = args.device
    wrapper_name = args.wrapper_name
    scaling_factor = args.scaling_factor
    max_kpts = args.max_kpts
    r = args.r
    nimages = args.nimages

    wrapper = wrappers_manager(name=wrapper_name, device=args.device)

    ghr_data_path = (
        abs_root_path / "benchmarks/graz_high_res/data/graz_clocktower/frames/26"
    )
    if not ghr_data_path.exists():
        raise ValueError(
            f"Path {ghr_data_path} does not exist. Update the path with 4K images or download GHR data."
        )
    else:
        logger.info(f"Using data from {ghr_data_path}")
    images_path = sorted(glob.glob(str(ghr_data_path / "*.jpg")))[:nimages]

    res = measure(
        wrapper,
        image_paths=images_path,
        device=args.device,
        r=r,
        scaling_factor=scaling_factor,
        max_kpts=max_kpts,
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    key = f"{wrapper.name}_{timestamp}"
    res_folder = abs_root_path / "benchmarks/speed_and_memory/results"
    res_folder.mkdir(parents=True, exist_ok=True)
    res_path = res_folder / "results.json"
    if not res_path.exists():
        with open(res_path, "w") as f:
            json.dump({}, f)

    with open(res_path, "r") as f:
        results = json.load(f)

    results[key] = {
        "mean_time_ms": round(res["mean"], 2),
        "std_time_ms": round(res["std"], 2),
        "image_size": res["image_size"],
        "max_kpts": max_kpts,
        "num_images": nimages,
        "scaling_factor": scaling_factor,
        "repetitions": r,
        "vram_GB": round(res["vram"] / 1_000, 2),
        "device": device,
    }
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {res_path}.")
    logger.info(f"Results: {results[key]}")
