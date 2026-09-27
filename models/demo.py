from dataclasses import asdict
from time import perf_counter
import cv2
import numpy as np
from utils.config import Config
from vision.types import DetectionResult, Defect, FramePacket
from vision.preprocessing import preprocess
from vision.calibration import load_baseline
from vision.features import candidates, extract_gene
from vision.adaptive import dynamic_weights, similarities, thresholds, fuse_region, clarity


class OpenCVDemoDetector:
    """Document-driven prototype; candidate extraction is an untrained demo heuristic."""

    def __init__(self, config: Config, state: dict | None = None):
        self.config = config
        self.alpha = np.asarray((state or {}).get("alpha", config.alpha), dtype=float)
        self.correction = (state or {}).get("weight_correction", [0, 0, 0, 0])
        self.match_threshold = (state or {}).get("match_threshold", config.match_threshold)
        self.gray_baseline = None
        self.baseline_count = 0
        if config.baseline_path:
            self.gray_baseline, self.baseline_count = load_baseline(config.path(config.baseline_path))
        self.reset()

    def load_model(self, path: str | None = None) -> None:
        if path:
            raise ValueError("文档算法Demo无需权重；真实模型请选择对应适配器")

    def reset(self) -> None:
        self.reference = None
        self.reference_genes = []
        self.reference_c = 1.0
        self.reference_clarity = 0.0
        self.reference_id = ""

    def predict(self, packet: FramePacket) -> DetectionResult:
        start = perf_counter()
        cfg = self.config
        values = packet.interference.normalized()
        c = float(np.dot(self.alpha, values))
        processed = preprocess(packet.image, packet.nir, cfg.wavelet)
        if self.gray_baseline is not None and self.gray_baseline.shape != processed.shape:
            raise ValueError("标定基准与当前帧尺寸不匹配")
        if self.reference is not None and self.reference.shape != processed.shape:
            self.reset()
        genes = [
            extract_gene(processed, b, packet.binding_id)
            for b in candidates(processed, cfg.candidate_threshold, cfg.min_area, cfg.max_area_ratio)
        ]
        weights = dynamic_weights(c, self.correction)
        fused = processed.copy()
        defects = []
        used = set()
        h, w = processed.shape
        reference_used = self.reference_id
        for gene in genes:
            defect = Defect(
                "候选异常",
                0.0,
                gene.bbox,
                "unconfirmed",
                gene,
                "没有基准或空间对应区域；单帧候选分数不代表概率",
            )
            best = None
            for i, base in enumerate(self.reference_genes):
                if i in used:
                    continue
                a, b = gene.bbox, base.bbox
                distance = np.hypot(
                    (a[0] + a[2] - b[0] - b[2]) / (2 * w), (a[1] + a[3] - b[1] - b[3]) / (2 * h)
                )
                if distance > cfg.spatial_gate:
                    continue
                sims = similarities(gene, base)
                score = float(np.dot(weights, sims))
                if best is None or score > best[0]:
                    best = score, i, sims
            if best is not None:
                score, index, sims = best
                used.add(index)
                base = self.reference_genes[index]
                base_mean = base.mean_gray
                if self.gray_baseline is not None:
                    bx1, by1, bx2, by2 = base.bbox
                    base_mean = float(self.gray_baseline[by1:by2, bx1:bx2].mean())
                interval, allowance = thresholds(base_mean, c, cfg.k1, cfg.k)
                defect.confidence = score
                defect.similarities = sims
                defect.reference_bbox = base.bbox
                defect.gray_interval = interval
                defect.gradient_allowance = allowance
                valid_gray = interval[0] <= gene.mean_gray <= interval[1]
                valid_gradient = abs(gene.gradient_mean - base.gradient_mean) <= allowance
                if score < cfg.pseudo_threshold:
                    defect.status, defect.reason = "filtered", "基因相似度低于伪缺陷阈值"
                elif score >= self.match_threshold and valid_gray and valid_gradient:
                    defect.status, defect.label = "matched", "跨帧匹配异常"
                    patch, mode = fuse_region(
                        processed, self.reference, gene.bbox, base.bbox, score, interval
                    )
                    x1, y1, x2, y2 = gene.bbox
                    fused[y1:y2, x1:x2] = patch
                    defect.fusion, defect.reason = mode, "余弦匹配及灰度/梯度动态阈值通过"
                else:
                    defect.reason = "相似度不足或动态灰度/梯度阈值未通过；保留复核"
            defects.append(defect)
        current_clarity = clarity(processed)
        # Reference choice follows the formal document, not arbitrary periodic replacement.
        if genes and (
            self.reference is None
            or (c < self.reference_c and current_clarity >= self.reference_clarity * 0.8)
        ):
            self.reference = processed.copy()
            self.reference_genes = genes
            self.reference_c, self.reference_clarity = c, current_clarity
            self.reference_id = packet.binding_id
        metrics = {
            "baseline_frame_count": self.baseline_count,
            "baseline_mode": "calibrated" if self.gray_baseline is not None else "single-reference-demo",
            "c_total": c,
            "normalized_interference": values.tolist(),
            "weights": weights.tolist(),
            "alpha": self.alpha.tolist(),
            "match_threshold": self.match_threshold,
            "reference_used": reference_used,
            "reference_next": self.reference_id,
            "nir_mode": "paired-wavelet" if packet.nir is not None else "missing-single-spectrum",
            "clarity_before": current_clarity,
            "clarity_after": clarity(fused),
            "filtered_ratio": sum(d.status == "filtered" for d in defects) / max(len(defects), 1),
            "notice": "工程演示：未经真实标定；匹配分数不是缺陷分类概率",
        }
        return DetectionResult(
            packet.source,
            packet.timestamp,
            packet.binding_id,
            "adaptive_demo",
            asdict(packet.interference),
            defects,
            (perf_counter() - start) * 1000,
            cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR),
            packet.image.copy(),
            metrics,
        )
