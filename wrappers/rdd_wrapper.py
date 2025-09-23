import sys
import yaml
import warnings
import torch
import torch.nn.functional as F

from pathlib import Path

method_path = Path(__file__).resolve().parents[1] / "methods/rdd"
sys.path.append(str(method_path))

warnings.filterwarnings("ignore", category=UserWarning)

from RDD.RDD import build
from wrappers.wrapper import MethodWrapper, MethodOutput


class RDDWrapper(MethodWrapper):
    def __init__(self, device: str = "cuda:0", border=16, config=None):
        super().__init__(name="rdd", border=border, device=device)

        try:
            # Load weights
            config_path = method_path / "configs/default.yaml"
            weights_path = method_path / "weights/RDD-v1.pth"

            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            if config is None:
                with open(config_path, "r") as file:
                    config = yaml.safe_load(file)
                # print("RDD config:", config)

            self.config = config

            RDD_model = build(
                config=config,
                weights=str(weights_path),
            )
            RDD_model.eval()

            # disable gradients
            for p in RDD_model.parameters():
                p.requires_grad = False

            self.RDD = RDD_model.to(device)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize RDD model: {e}")

    @torch.inference_mode()
    def _extract(self, x, max_kpts: int = 2048, custom_kpts=None) -> MethodOutput:
        if self.config["top_k"] != max_kpts:
            self.config["top_k"] = max_kpts
            self.__init__(device=self.device, border=self.border, config=self.config)

        x = x if x.dim() == 4 else x[None]

        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            out = self.RDD.extract(x)[0]
            kpts, scores, des = (
                out["keypoints"],
                out["scores"],
                out["descriptors"],
            )

        return MethodOutput(kpts=kpts, kpts_scores=scores, des=des)
