"""Dynamic thresholds, weighted gene similarity and region-level fusion."""

import cv2
import numpy as np

LOW = np.array([0.2, 0.2, 0.5, 0.1])
HIGH = np.array([0.3, 0.4, 0.1, 0.2])


def dynamic_weights(c: float, correction=None) -> np.ndarray:
    ratio = float(np.clip((c - 0.3) / 0.4, 0, 1))
    weights = (1 - ratio) * LOW + ratio * HIGH
    if correction is not None:
        weights = np.maximum(0.01, weights + np.asarray(correction))
    return weights / weights.sum()


def similarities(current, reference) -> list[float]:
    scores = []
    for a, b in zip(current.vectors, reference.vectors):
        a, b = np.asarray(a), np.asarray(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        score = float(np.dot(a, b) / denom) if denom > 1e-12 else float(np.array_equal(a, b))
        scores.append(float(np.clip(score, 0, 1)))
    return scores


def thresholds(mean: float, c: float, k1: float, k: float):
    drift = k1 * c
    return [max(0, mean - drift), min(255, mean + drift)], k * c


def fuse_region(current, reference, bbox, reference_bbox, confidence, interval):
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = reference_bbox
    a = current[y1:y2, x1:x2].astype(float)
    b = reference[ry1:ry2, rx1:rx2]
    b = cv2.resize(b, (x2 - x1, y2 - y1)).astype(float)
    drift = abs(float(a.mean() - b.mean())) / max(float(b.mean()), 1)
    if drift < 0.1:
        fused, mode = (a + b) / 2, "mean"
    else:
        target = np.clip(b.mean(), interval[0], interval[1])
        corrected = np.clip(a + target - a.mean(), 0, 255)
        fused, mode = (confidence * corrected + b) / (confidence + 1), "aligned_corrected"
    return np.clip(fused, 0, 255).astype(np.uint8), mode


def clarity(image) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())
