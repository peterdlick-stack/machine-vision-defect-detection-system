import cv2
import numpy as np
import pywt


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("图像必须是uint8数组")
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 8:
        raise ValueError("图像尺寸至少8×8")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("彩色图像必须有3通道(BGR)")


def gray(image: np.ndarray) -> np.ndarray:
    validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


def preprocess(image: np.ndarray, nir: np.ndarray | None = None, wavelet: str = "haar") -> np.ndarray:
    visible = gray(image)
    if nir is None:
        return cv2.GaussianBlur(visible, (3, 3), 0.6)
    infrared = gray(nir)
    if infrared.shape != visible.shape:
        raise ValueError("NIR与可见光尺寸不同；请先完成相机几何标定/配准")
    a = pywt.wavedec2(visible.astype(float), wavelet, level=2)
    b = pywt.wavedec2(infrared.astype(float), wavelet, level=2)
    fused = [(a[0] + b[0]) / 2]
    for high_a, high_b in zip(a[1:], b[1:]):
        fused.append(tuple(np.where(abs(x) >= abs(y), x, y) for x, y in zip(high_a, high_b)))
    restored = pywt.waverec2(fused, wavelet)
    return np.clip(restored[: visible.shape[0], : visible.shape[1]], 0, 255).astype(np.uint8)
