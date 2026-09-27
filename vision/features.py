"""原文的四种特征；候选提议和128维编码规则属于工程补充。"""

import cv2
import numpy as np
from vision.types import Gene


def embedding(values) -> list[float]:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        values = np.zeros(1)
    vector = np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, len(values)), values)
    norm = np.linalg.norm(vector)
    return (vector / norm if norm > 1e-12 else vector).tolist()


def candidates(image, threshold: float, min_area: int, max_ratio: float):
    background = cv2.GaussianBlur(image, (0, 0), 9)
    residual = cv2.absdiff(image, background)
    mask = (residual > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= h * w * max_ratio:
            x, y, bw, bh = cv2.boundingRect(contour)
            boxes.append((max(0, x - 2), max(0, y - 2), min(w, x + bw + 2), min(h, y + bh + 2)))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def extract_gene(image: np.ndarray, bbox, binding_id: str) -> Gene:
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    h, w = image.shape
    anchors = [x1 / w, y1 / h, x2 / w, y1 / h, x2 / w, y2 / h, x1 / w, y2 / h]
    edges = cv2.Canny(roi, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    chain = []
    directions = {
        (1, 0): 0,
        (1, 1): 1,
        (0, 1): 2,
        (-1, 1): 3,
        (-1, 0): 4,
        (-1, -1): 5,
        (0, -1): 6,
        (1, -1): 7,
    }
    if contours:
        points = max(contours, key=len).reshape(-1, 2)
        for delta in np.roll(points, -1, axis=0) - points:
            direction = tuple(np.sign(delta).tolist())
            if direction in directions:
                chain.append(directions[direction])
    topology = np.bincount(chain, minlength=8).astype(float)
    dx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(dx, dy)
    gradient = np.histogram(magnitude, bins=128, range=(0, 1443))[0]
    # Quantized gray-level co-occurrence at d=1, theta=0, as specified.
    levels = roi.astype(np.int32) // 16
    glcm = np.bincount((levels[:, :-1] * 16 + levels[:, 1:]).ravel(), minlength=256).astype(float)
    glcm /= max(glcm.sum(), 1)
    positive = glcm[glcm > 0]
    entropy = float(-np.sum(positive * np.log2(positive)))
    # Encode entropy as a distribution: scalar cosine would always be one.
    axis = np.linspace(0, 8, 128)
    texture = np.exp(-0.5 * ((axis - entropy) / 0.5) ** 2)
    return Gene(
        tuple(bbox),
        [embedding(anchors), embedding(topology), embedding(gradient), embedding(texture)],
        float(roi.mean()),
        float(magnitude.mean()),
        entropy,
        binding_id,
        chain,
    )
