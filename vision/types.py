from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Interference:
    light_lux: float
    temperature_c: float
    reflectance: float
    source: str = "simulated-config"

    def normalized(self) -> np.ndarray:
        values = np.array(
            [self.light_lux / 10000, (self.temperature_c - 20) / 40, (self.reflectance - 0.1) / 0.8],
            dtype=float,
        )
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("干扰数据超出标定范围或包含非有限值")
        return values


@dataclass
class FramePacket:
    image: np.ndarray
    interference: Interference
    source: str
    nir: np.ndarray | None = None
    frame_id: int = 0
    session_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=timestamp)

    @property
    def binding_id(self) -> str:
        return f"{self.session_id}:{self.frame_id & 0xFFFF:016b}"


@dataclass
class Gene:
    bbox: tuple[int, int, int, int]
    vectors: list[list[float]]
    mean_gray: float
    gradient_mean: float
    entropy: float
    binding_id: str
    chain_code: list[int]


@dataclass
class Defect:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    status: str
    gene: Gene
    reason: str = ""
    similarities: list[float] = field(default_factory=list)
    reference_bbox: tuple | None = None
    gray_interval: list[float] = field(default_factory=list)
    gradient_allowance: float = 0.0
    fusion: str = "none"


@dataclass
class DetectionResult:
    source: str
    timestamp: str
    binding_id: str
    detector: str
    interference: dict
    defects: list[Defect]
    inference_ms: float
    image: np.ndarray
    original: np.ndarray
    metrics: dict = field(default_factory=dict)
    result_id: str = field(default_factory=lambda: uuid4().hex)
    paths: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "source": self.source,
            "timestamp": self.timestamp,
            "binding_id": self.binding_id,
            "detector": self.detector,
            "interference": self.interference,
            "defects": [asdict(d) for d in self.defects],
            "inference_ms": self.inference_ms,
            "metrics": self.metrics,
            "paths": self.paths,
            "count": sum(d.status != "filtered" for d in self.defects),
        }
