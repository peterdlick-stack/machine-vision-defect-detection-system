import json
import logging
from pathlib import Path
from uuid import uuid4
from models.demo import OpenCVDemoDetector
from models.yolo import YOLODetector
from services.learning import load_state
from services.storage import Storage, read_image
from vision.types import FramePacket, Interference

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
logger = logging.getLogger(__name__)


class DetectionService:
    def __init__(self, config):
        self.config = config.validate()
        self.storage = Storage(config.path(config.output_dir))
        self.detector = (
            YOLODetector(str(config.path(config.model_path)))
            if config.detector == "yolo"
            else OpenCVDemoDetector(config, load_state(config.path(config.learning_state), config))
        )
        self.session_id = uuid4().hex
        self.frame_id = 0

    def packet(self, image, source: str, nir=None, metadata=None):
        cfg = self.config
        interference = Interference(cfg.light_lux, cfg.temperature_c, cfg.reflectance)
        if metadata is not None:
            interference = Interference(**metadata)
        interference.normalized()
        packet = FramePacket(image, interference, source, nir, self.frame_id, self.session_id)
        self.frame_id += 1
        return packet

    def detect(self, packet, sequence=False):
        if not sequence:
            self.detector.reset()
        result = self.storage.save(self.detector.predict(packet))
        logger.info(
            "检测保存完成 source=%s frame=%s count=%s inference_ms=%.3f",
            packet.source,
            packet.binding_id,
            len(result.defects),
            result.inference_ms,
        )
        return result

    def detect_file(self, path: Path, sequence=False):
        path = Path(path)
        image = read_image(path)
        nir_path = path.with_name(path.stem + "_nir.png")
        sidecar = path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        nir = read_image(nir_path) if nir_path.is_file() else None
        packet = self.packet(image, str(path.resolve()), nir, metadata.get("interference"))
        return self.detect(packet, sequence)


def image_files(folder: Path):
    return sorted(
        p
        for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.stem.endswith("_nir")
    )
