"""Optional real-model adapter. Never downloads weights implicitly."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from vision.preprocessing import gray
from vision.features import extract_gene
from vision.types import DetectionResult, Defect


class YOLODetector:
    def __init__(self, path: str):
        self.load_model(path)

    def load_model(self, path: str | None = None) -> None:
        if not path or not Path(path).is_file():
            raise FileNotFoundError("未找到本地YOLO权重，请配置真实模型路径或选择Demo")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements-yolo.txt") from exc
        self.model = YOLO(str(Path(path).resolve()))

    def reset(self) -> None:
        """Stateless single-frame adapter has no reference frame."""

    def predict(self, packet):
        start = perf_counter()
        prediction = self.model.predict(packet.image, verbose=False)[0]
        defects = []
        image_gray = gray(packet.image)
        h, w = image_gray.shape
        for box in prediction.boxes:
            coords = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = [int(v) for v in coords]
            bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                continue
            gene = extract_gene(image_gray, bbox, packet.binding_id)
            defects.append(
                Defect(
                    str(prediction.names[int(box.cls.item())]),
                    float(box.conf.item()),
                    bbox,
                    "model",
                    gene,
                    "真实模型输出；不执行文档多帧融合",
                )
            )
        return DetectionResult(
            packet.source,
            packet.timestamp,
            packet.binding_id,
            "yolo",
            asdict(packet.interference),
            defects,
            (perf_counter() - start) * 1000,
            packet.image.copy(),
            packet.image.copy(),
            {"notice": "独立YOLO模式"},
        )
