"""100-frame undisturbed grayscale reference calibration."""

from pathlib import Path
import numpy as np
from vision.preprocessing import preprocess


def build_baseline(pairs, output: Path):
    mean = None
    count = 0
    for visible, nir in pairs:
        frame = preprocess(visible, nir).astype(float)
        if mean is not None and mean.shape != frame.shape:
            raise ValueError("基准帧必须同一目标、同一视角、相同尺寸")
        count += 1
        mean = frame if mean is None else mean + (frame - mean) / count
    if count < 100:
        raise ValueError("文档基准灰度标定要求至少100帧无干扰图像")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, mean_gray=mean, frame_count=count)
    return count


def load_baseline(path: Path):
    with np.load(path, allow_pickle=False) as bundle:
        image, count = bundle["mean_gray"], int(bundle["frame_count"])
    if image.ndim != 2 or min(image.shape) < 8 or count < 100 or not np.isfinite(image).all():
        raise ValueError("基准灰度文件无效")
    if np.any((image < 0) | (image > 255)):
        raise ValueError("基准灰度超出0–255")
    return image, count
