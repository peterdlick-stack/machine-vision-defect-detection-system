import logging
from threading import Event
from PySide6.QtCore import QThread, Signal
from services.detection import DetectionService
from vision.camera import VideoSource

logger = logging.getLogger(__name__)


class DetectionWorker(QThread):
    result_ready = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, config, paths=None, capture=None, sequence=False, parent=None):
        super().__init__(parent)
        self.config, self.paths, self.capture = config, paths or [], capture
        self.sequence = sequence
        self.stop_event = Event()
        self.displayed = Event()

    def stop(self):
        self.stop_event.set()
        self.displayed.set()

    def deliver(self, result):
        self.displayed.clear()
        self.result_ready.emit(result)
        # Backpressure prevents an unbounded Qt event queue during live capture.
        while not self.displayed.wait(0.1):
            if self.stop_event.is_set():
                break

    def run(self):
        source = None
        try:
            service = DetectionService(self.config)
            if self.capture is not None:
                source = VideoSource(self.capture)
                while not self.stop_event.is_set():
                    image = source.read()
                    if image is None:
                        break
                    packet = service.packet(image, f"capture:{self.capture}")
                    self.deliver(service.detect(packet, sequence=True))
            else:
                for index, path in enumerate(self.paths):
                    if self.stop_event.is_set():
                        break
                    self.progress.emit(f"检测 {index + 1}/{len(self.paths)} · {path.name}")
                    try:
                        self.deliver(service.detect_file(path, self.sequence))
                    except Exception as exc:
                        logger.exception("图像检测失败: %s", path)
                        self.error.emit(f"{path.name}: {exc}")
        except Exception as exc:
            logger.exception("后台任务失败")
            self.error.emit(str(exc))
        finally:
            if source:
                source.close()
