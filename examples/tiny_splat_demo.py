from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch

from src.splatting import GaussianSplatRenderer, create_toy_gaussian_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a tiny differentiable Gaussian splat scene.")
    parser.add_argument("--output", default="results/splat_demo/toy_splat.png")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = create_toy_gaussian_scene(device=device)
    renderer = GaussianSplatRenderer(image_size=(args.image_size, args.image_size))
    image = renderer(scene.means3d, scene.colors, scene.log_scales, scene.opacity_logits)["image"]
    plt.imsave(out, image.detach().cpu().permute(1, 2, 0).numpy())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
