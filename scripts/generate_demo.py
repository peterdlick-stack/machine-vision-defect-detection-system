"""Deterministic synthetic images and synchronized-looking NIR pairs (not measured data)."""

from pathlib import Path
import json
import cv2
import numpy as np
from services.storage import write_image
from utils.config import ROOT


def generate(folder: Path | None = None):
    folder = folder or ROOT / "data/input/demo"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    base = np.full((360, 540), 170, dtype=np.uint8)
    cv2.line(base, (130, 120), (210, 154), 45, 5)
    cv2.circle(base, (370, 235), 11, 70, -1)
    for index, drift in enumerate([0, 1, 3, 8, 16, 28]):
        noise = rng.normal(0, 0.35, base.shape)
        visible = np.clip(base.astype(float) + drift + noise, 0, 255).astype(np.uint8)
        nir = np.clip(base.astype(float) + drift + noise * 0.5, 0, 255).astype(np.uint8)
        name = folder / f"frame_{index:03d}.png"
        write_image(name, visible)
        write_image(folder / f"frame_{index:03d}_nir.png", nir)
        name.with_suffix(".json").write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "interference": {
                        "light_lux": 2000 + index * 900,
                        "temperature_c": 28 + index * 2,
                        "reflectance": 0.26 + index * 0.03,
                        "source": "synthetic-demo",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    write_image(folder.parent / "blank.png", np.full_like(base, 170))
    video = cv2.VideoWriter(str(folder.parent / "demo.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (540, 360))
    if not video.isOpened():
        raise RuntimeError("无法生成MJPG演示视频")
    try:
        for _ in range(5):
            for index in range(6):
                image = cv2.imread(str(folder / f"frame_{index:03d}.png"))
                video.write(image)
    finally:
        video.release()
    return folder


if __name__ == "__main__":
    print(generate())
