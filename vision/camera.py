from typing import Protocol
import cv2
from vision.types import FramePacket


class IndustrialCamera(Protocol):
    """Hardware SDK adapters must supply synchronized VIS/NIR and measured interference."""

    def read_packet(self) -> FramePacket | None: ...
    def close(self) -> None: ...


class VideoSource:
    def __init__(self, source: int | str):
        self.source = source
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"无法打开摄像头/视频: {source}")

    def read(self):
        success, image = self.capture.read()
        if success:
            return image
        if isinstance(self.source, int):
            raise RuntimeError("摄像头读取失败或已断开")
        return None

    def close(self):
        self.capture.release()
