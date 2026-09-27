"""Validated configuration. Relative paths always resolve against the project root."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    detector: str = "adaptive_demo"
    baseline_path: str = ""
    model_path: str = "weights/best.pt"
    output_dir: str = "data/output"
    camera_index: int = 0
    candidate_threshold: float = 18.0
    min_area: int = 16
    max_area_ratio: float = 0.2
    alpha: tuple = (0.3, 0.4, 0.3)
    k1: float = 10.0
    k: float = 3.0
    match_threshold: float = 0.75
    pseudo_threshold: float = 0.5
    spatial_gate: float = 0.15
    wavelet: str = "haar"
    light_lux: float = 3000.0
    temperature_c: float = 30.0
    reflectance: float = 0.3
    learning_rate: float = 0.001
    learning_batch: int = 1000
    learning_state: str = "data/output/learning_state.json"

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"参数 {key} 必须为有限数值")
        for key in ("min_area", "camera_index", "learning_batch"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"参数 {key} 必须为整数")
        if self.camera_index < 0:
            raise ValueError("摄像头索引须非负")
        if self.detector not in {"adaptive_demo", "yolo"}:
            raise ValueError("detector 必须为 adaptive_demo 或 yolo")
        if not 0 <= self.pseudo_threshold < self.match_threshold <= 1:
            raise ValueError("必须满足 0 ≤ 伪缺陷阈值 < 匹配阈值 ≤ 1")
        if not 0 < self.candidate_threshold <= 255 or self.min_area < 1:
            raise ValueError("候选阈值须在 (0,255] 且最小面积 ≥ 1")
        if not 0 < self.max_area_ratio <= 1 or not 0 < self.spatial_gate <= 1:
            raise ValueError("面积比例和空间门限须在 (0,1]")
        bounds = [(0.2, 0.4), (0.3, 0.5), (0.2, 0.4)]
        if len(self.alpha) != 3 or any(not lo <= a <= hi for a, (lo, hi) in zip(self.alpha, bounds)):
            raise ValueError("alpha 超出文档范围")
        if abs(sum(self.alpha) - 1) > 1e-6:
            raise ValueError("alpha 总和须为1")
        if not 5 <= self.k1 <= 15 or self.k < 0:
            raise ValueError("k1须在[5,15]，k须非负")
        if not 0 <= self.light_lux <= 10000 or not 20 <= self.temperature_c <= 60:
            raise ValueError("光强须为0–10000lux，温度须为20–60℃")
        if not 0.1 <= self.reflectance <= 0.9:
            raise ValueError("反光率须为0.1–0.9")
        if not 0 < self.learning_rate <= 0.1 or self.learning_batch < 1:
            raise ValueError("学习率或批大小无效")
        if self.wavelet not in {"haar", "db2"}:
            raise ValueError("支持的小波基：haar/db2")
        return self

    def path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    def save(self, path: Path):
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config/config.json"
    try:
        return Config(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"配置格式错误: {exc}") from exc
