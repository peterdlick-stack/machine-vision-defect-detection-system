from contextlib import contextmanager
import csv
import json
import sqlite3
from pathlib import Path
import cv2
import numpy as np
from vision.visualization import annotate


def read_image(path: Path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error) as exc:
        raise ValueError(f"读取图像失败: {path}") from exc
    if image is None:
        raise ValueError(f"无效或损坏的图像: {path}")
    return image


def write_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"无法编码图像: {path}")
    encoded.tofile(path)


class Storage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "history.sqlite3"
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, source TEXT NOT NULL,
                detector TEXT NOT NULL, count INTEGER NOT NULL, inference_ms REAL NOT NULL,
                payload TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON results(timestamp)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, result):
        prefix = self.root / result.result_id
        paths = {
            "original": str(prefix) + "_original.png",
            "result": str(prefix) + "_result.png",
            "fused": str(prefix) + "_fused.png",
            "json": str(prefix) + ".json",
        }
        if any(Path(p).exists() for p in paths.values()):
            raise FileExistsError("结果ID已存在，拒绝覆盖历史文件")
        result.paths = paths
        try:
            write_image(Path(paths["original"]), result.original)
            write_image(Path(paths["result"]), annotate(result))
            write_image(Path(paths["fused"]), result.image)
            payload = result.to_dict()
            serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            Path(paths["json"]).write_text(serialized, encoding="utf-8")
            with self.connect() as db:
                db.execute(
                    "INSERT INTO results VALUES(?,?,?,?,?,?,?)",
                    (
                        result.result_id,
                        result.timestamp,
                        result.source,
                        result.detector,
                        payload["count"],
                        result.inference_ms,
                        serialized,
                    ),
                )
        except Exception:
            for path in paths.values():
                Path(path).unlink(missing_ok=True)
            result.paths = {}
            raise
        return result

    def history(self, limit=200):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM results ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def export_csv(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db, path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "record_id",
                    "source",
                    "timestamp",
                    "detector",
                    "count",
                    "inference_ms",
                    "label",
                    "score",
                    "status",
                    "bbox",
                    "result_path",
                ]
            )
            for (text,) in db.execute("SELECT payload FROM results ORDER BY timestamp"):
                row = json.loads(text)
                base = [
                    row[k] for k in ["result_id", "source", "timestamp", "detector", "count", "inference_ms"]
                ]
                for defect in row["defects"] or [{}]:
                    values = base + [defect.get(k, "") for k in ["label", "confidence", "status", "bbox"]]
                    values.append(row["paths"]["result"])
                    # Spreadsheet formula injection defense for user-controlled filenames.
                    writer.writerow(
                        [
                            ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
                            for v in values
                        ]
                    )
