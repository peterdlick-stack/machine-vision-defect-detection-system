import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
from time import monotonic
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
from app.worker import DetectionWorker
from scripts.generate_demo import generate
from services.detection import image_files
from utils.config import Config


def test_gui_worker_handles_bad_image_and_saves(tmp_path):
    app = QApplication.instance() or QApplication([])
    folder = generate(tmp_path / "input")
    bad = folder / "bad.png"
    bad.write_bytes(b"broken")
    config = replace(Config(), output_dir=str(tmp_path / "out"), learning_state=str(tmp_path / "no.json"))
    window = MainWindow(config)
    window.show()
    window.select_paths([image_files(folder)[1], bad, image_files(folder)[2]], sequence=True)
    finished = []
    window.run_finished.connect(lambda: finished.append(True))
    window.start()
    deadline = monotonic() + 15
    while not finished and monotonic() < deadline:
        app.processEvents()
    if not finished:
        window.stop()
        window.worker.wait(5000)
    assert finished
    assert window.processed_count == 2
    assert len(window.errors) == 1
    assert window.last_result.paths
    assert window.original.pixmap() is not None
    assert window.result_panel.pixmap() is not None
    assert window.start_button.isEnabled()
    window.close()


def test_video_stop_releases_thread(tmp_path):
    app = QApplication.instance() or QApplication([])
    generate(tmp_path / "input")
    config = replace(Config(), output_dir=str(tmp_path / "out"), learning_state=str(tmp_path / "no.json"))
    worker = DetectionWorker(config, capture=str(tmp_path / "demo.avi"), sequence=True)
    results, errors = [], []

    def receive(result):
        results.append(result)
        worker.stop()
        worker.displayed.set()

    worker.result_ready.connect(receive)
    worker.error.connect(errors.append)
    worker.start()
    deadline = monotonic() + 15
    while worker.isRunning() and monotonic() < deadline:
        app.processEvents()
    worker.stop()
    assert worker.wait(5000)
    assert results and not errors


def test_image_dialog_and_csv_export(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from services.detection import DetectionService

    QApplication.instance() or QApplication([])
    folder = generate(tmp_path / "input")
    chosen = folder / "frame_000.png"
    cfg = replace(Config(), output_dir=str(tmp_path / "out"), learning_state=str(tmp_path / "none.json"))
    window = MainWindow(cfg)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(chosen), ""))
    window.open_image()
    assert window.paths == [chosen]
    assert not window.sequence.isChecked()
    result = DetectionService(cfg).detect_file(chosen)
    window.display_result(result)
    csv_path = tmp_path / "历史.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(csv_path), ""))
    window.export_csv()
    assert csv_path.exists() and "record_id" in csv_path.read_text(encoding="utf-8-sig")
    window.close()
