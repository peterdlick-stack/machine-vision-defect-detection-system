from dataclasses import replace
from pathlib import Path
import csv
import json
import numpy as np
import pytest
from utils.config import Config, load_config
from vision.types import Interference, FramePacket
from vision.preprocessing import preprocess
from vision.features import extract_gene
from vision.adaptive import dynamic_weights, thresholds, fuse_region
from models.demo import OpenCVDemoDetector
from models.yolo import YOLODetector
from services.storage import Storage, write_image, read_image
from services.detection import DetectionService, image_files
from services.learning import calibrate, incremental_update, project_alpha
from scripts.generate_demo import generate


@pytest.fixture
def image():
    image = np.full((128, 160, 3), 170, np.uint8)
    image[50:56, 40:95] = 40
    return image


def packet(image, frame_id=0):
    return FramePacket(image, Interference(6000, 32, 0.5), "test", frame_id=frame_id, session_id="test")


def test_config(tmp_path):
    cfg = load_config()
    cfg.save(tmp_path / "config.json")
    assert load_config(tmp_path / "config.json").alpha == [0.3, 0.4, 0.3]
    assert cfg.path("data/input").is_absolute()
    with pytest.raises(ValueError):
        replace(cfg, match_threshold=0.2).validate()
    with pytest.raises(ValueError):
        replace(cfg, alpha=[0.1, 0.8, 0.1]).validate()
    with pytest.raises(ValueError):
        replace(cfg, candidate_threshold=float("nan")).validate()


def test_formula():
    values = Interference(6000, 32, 0.5).normalized()
    np.testing.assert_allclose(values, [0.6, 0.3, 0.5])
    assert np.dot([0.3, 0.4, 0.3], values) == pytest.approx(0.45)
    np.testing.assert_allclose(dynamic_weights(0.1), [0.2, 0.2, 0.5, 0.1])
    np.testing.assert_allclose(dynamic_weights(0.7), [0.3, 0.4, 0.1, 0.2])
    assert sum(dynamic_weights(0.5)) == pytest.approx(1)
    assert thresholds(100, 0.45, 10, 3) == ([95.5, 104.5], pytest.approx(1.35))
    with pytest.raises(ValueError):
        Interference(20000, 32, 0.5).normalized()


def test_preprocess(image):
    result = preprocess(image)
    assert result.shape == image.shape[:2] and result.dtype == np.uint8
    pair = preprocess(image, image)
    np.testing.assert_allclose(pair, image[:, :, 0], atol=1)
    with pytest.raises(ValueError):
        preprocess(image, image[:64])
    with pytest.raises(ValueError):
        preprocess(np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError):
        preprocess(image.astype(float))


def test_genes(image):
    gene = extract_gene(preprocess(image), (35, 45, 100, 62), "session:0000")
    assert len(gene.vectors) == 4
    assert all(len(vector) == 128 for vector in gene.vectors)
    assert all(np.isfinite(vector).all() for vector in gene.vectors)
    assert gene.chain_code and 0 <= gene.entropy <= 8
    assert gene.binding_id == "session:0000"


def test_demo_without_model_and_repeat_matching(image):
    detector = OpenCVDemoDetector(Config())
    detector.load_model()
    first = detector.predict(packet(image))
    second = detector.predict(packet(image, 1))
    assert first.defects
    assert all(d.status == "unconfirmed" for d in first.defects)
    assert any(d.status == "matched" for d in second.defects)
    assert second.metrics["reference_used"] == first.binding_id
    assert second.to_dict()["count"] > 0
    assert "image" not in second.to_dict()
    json.dumps(second.to_dict(), allow_nan=False)
    detector.reset()
    assert all(d.status == "unconfirmed" for d in detector.predict(packet(image)).defects)


def test_blank_and_size_change(image):
    detector = OpenCVDemoDetector(Config())
    assert detector.predict(packet(np.full_like(image, 170))).defects == []
    detector.predict(packet(image))
    result = detector.predict(packet(np.full((90, 90, 3), 170, np.uint8)))
    assert result.metrics["reference_used"] == ""


def test_fusion_branches():
    reference = np.full((30, 30), 40, np.uint8)
    current = np.full((30, 30), 41, np.uint8)
    patch, mode = fuse_region(current, reference, (0, 0, 20, 20), (0, 0, 20, 20), 0.9, [30, 50])
    assert mode == "mean" and patch.mean() == 40
    current[:] = 48
    patch, mode = fuse_region(current, reference, (0, 0, 20, 20), (0, 0, 15, 15), 0.9, [30, 50])
    assert mode == "aligned_corrected" and patch.mean() == 40


def test_database_and_export(image, tmp_path):
    result = OpenCVDemoDetector(Config()).predict(packet(image))
    storage = Storage(tmp_path)
    storage.save(result)
    row = storage.history()[0]
    assert row["binding_id"] == result.binding_id
    assert row["defects"][0]["gene"]["binding_id"] == result.binding_id
    assert all(Path(p).exists() for p in row["paths"].values())
    storage.export_csv(tmp_path / "out.csv")
    with (tmp_path / "out.csv").open(encoding="utf-8-sig") as stream:
        records = list(csv.reader(stream))
    assert len(records) > 1 and "bbox" in records[0]
    assert np.array_equal(read_image(Path(result.paths["original"])), image)


def test_unicode_and_invalid_images(tmp_path, image):
    path = tmp_path / "缺陷 图.png"
    write_image(path, image)
    assert np.array_equal(read_image(path), image)
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        read_image(path)


def test_end_to_end(tmp_path):
    folder = generate(tmp_path / "input")
    cfg = replace(
        Config(), output_dir=str(tmp_path / "output"), learning_state=str(tmp_path / "no-state.json")
    )
    service = DetectionService(cfg)
    files = image_files(folder)
    assert len(files) == 6
    results = [service.detect_file(path, sequence=True) for path in files]
    assert len(service.storage.history()) == 6
    assert all(r.metrics["nir_mode"] == "paired-wavelet" for r in results)
    assert all(r.interference["source"] == "synthetic-demo" for r in results)
    assert results[0].defects


def test_missing_yolo():
    with pytest.raises(FileNotFoundError):
        YOLODetector("/no-such-model.pt")


def samples(n=100):
    rng = np.random.default_rng(7)
    values = rng.uniform(0, 1, (n, 3))
    return [
        {
            "sample_id": str(i),
            "interference": {
                "light_lux": float(x[0] * 10000),
                "temperature_c": float(20 + x[1] * 40),
                "reflectance": float(0.1 + x[2] * 0.8),
                "source": "measured",
            },
            "measured_c": float(x @ [0.3, 0.4, 0.3]),
            "similarities": [0.8, 0.9, 0.7, 0.8],
            "match_label": 1,
        }
        for i, x in enumerate(values)
    ]


def test_supervised_learning_and_deduplication():
    with pytest.raises(ValueError):
        calibrate(samples(99))
    rows = samples(1000)
    state = calibrate(rows[:100])
    assert state["calibration_target_met"]
    np.testing.assert_allclose(state["alpha"], [0.3, 0.4, 0.3], atol=1e-6)
    updated, count = incremental_update(state, rows, Config())
    assert count == 1000
    assert updated["weight_correction"] != [0, 0, 0, 0]
    assert sum(updated["alpha"]) == pytest.approx(1)
    _, count = incremental_update(updated, rows, Config())
    assert count == 0
    _, count = incremental_update(state, rows[:999], Config())
    assert count == 0
    for values in [[-1, 4, 3], [0.3, 0.4, 0.3]]:
        alpha = project_alpha(values)
        assert sum(alpha) == pytest.approx(1)
        assert 0.2 <= alpha[0] <= 0.4
        assert 0.3 <= alpha[1] <= 0.5


def test_binding_wrap(image):
    assert packet(image, 65536).binding_id.endswith("0" * 16)
    one, two = packet(image), packet(image)
    two.session_id = "different"
    assert one.binding_id != two.binding_id


def test_100_frame_baseline(image, tmp_path):
    from vision.calibration import build_baseline, load_baseline

    target = tmp_path / "baseline.npz"
    with pytest.raises(ValueError):
        build_baseline(((image, None) for _ in range(99)), target)
    assert not target.exists()
    assert build_baseline(((image, None) for _ in range(100)), target) == 100
    mean, count = load_baseline(target)
    assert mean.shape == image.shape[:2] and count == 100
    detector = OpenCVDemoDetector(replace(Config(), baseline_path=str(target)))
    result = detector.predict(packet(image))
    assert result.metrics["baseline_mode"] == "calibrated"


def test_history_collision_preserves_saved_files(image, tmp_path):
    storage = Storage(tmp_path)
    result = OpenCVDemoDetector(Config()).predict(packet(image))
    storage.save(result)
    original = Path(result.paths["original"]).read_bytes()
    with pytest.raises(FileExistsError):
        storage.save(result)
    assert Path(result.paths["original"]).read_bytes() == original
    assert len(storage.history()) == 1


def test_invalid_sidecar_fails_explicitly(tmp_path, image):
    cfg = replace(Config(), output_dir=str(tmp_path / "out"), learning_state=str(tmp_path / "none"))
    path = tmp_path / "input.png"
    write_image(path, image)
    path.with_suffix(".json").write_text(
        '{"interference":{"light_lux":20000,"temperature_c":30,"reflectance":0.5}}'
    )
    with pytest.raises(ValueError):
        DetectionService(cfg).detect_file(path)
