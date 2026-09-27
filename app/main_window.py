from dataclasses import replace
from pathlib import Path
import json
import logging
import cv2
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialog,
    QFormLayout,
    QDoubleSpinBox,
    QDialogButtonBox,
    QLineEdit,
    QSplitter,
    QTextEdit,
)
from app.worker import DetectionWorker
from services.storage import Storage, read_image
from services.detection import image_files
from vision.visualization import annotate
from utils.config import ROOT

logger = logging.getLogger(__name__)
STATUS = {"unconfirmed": "待复核", "matched": "跨帧匹配", "filtered": "已过滤", "model": "模型输出"}


class ImagePanel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(280, 240)
        self.setStyleSheet("background:#111c2e;color:#8093b0;border-radius:10px;")
        self.source_pixmap = None

    def set_image(self, image):
        rgb = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim == 3
            else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        )
        self.source_pixmap = QPixmap.fromImage(
            QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        )
        self.refresh()

    def refresh(self):
        if self.source_pixmap:
            self.setPixmap(
                self.source_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


class MainWindow(QMainWindow):
    run_finished = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.worker = None
        self.paths = []
        self.capture = None
        self.last_result = None
        self.errors = []
        self.processed_count = 0
        self.closing = False
        self.setWindowTitle("基于机器视觉的缺陷检测系统 · 多帧融合实验台")
        self.resize(1320, 890)
        self.setMinimumSize(1000, 720)
        self.setStyleSheet("""
            QMainWindow,QDialog {background:#f1f5fa;color:#17283f;}
            QLabel {color:#263b55;}
            QPushButton {padding:9px 13px;background:white;border:1px solid #ccd7e5;border-radius:6px;}
            QPushButton:hover {background:#e6edf7;}
            QPushButton:disabled {color:#a7b3c3;background:#edf1f6;}
            QComboBox,QLineEdit,QDoubleSpinBox {padding:7px;background:white;border:1px solid #ccd7e5;border-radius:5px;}
            QTableWidget,QTextEdit {background:white;border:1px solid #d6e0eb;selection-background-color:#d3e6fb;}
            QHeaderView::section {background:#e7eef7;border:0;padding:7px;color:#37516f;}
        """)
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(22, 18, 22, 16)
        heading = QLabel("VISION LAB   /   自适应缺陷检测")
        heading.setStyleSheet("font-size:23px;font-weight:700;color:#142941;")
        layout.addWidget(heading)
        self.notice = QLabel("文档算法原型 · 演示候选提取 · 无训练权重也可运行 · 匹配分数不是分类概率")
        self.notice.setStyleSheet("color:#8d6221;background:#fff1d9;padding:9px;border-radius:5px;")
        layout.addWidget(self.notice)
        toolbar = QHBoxLayout()
        self.controls = []
        for text, handler in [
            ("打开图片", self.open_image),
            ("打开文件夹", self.open_folder),
            ("视频文件", self.open_video),
            ("USB 摄像头", self.open_camera),
            ("演示序列", self.open_demo),
            ("参数设置", self.settings),
        ]:
            button = QPushButton(text)
            button.clicked.connect(handler)
            toolbar.addWidget(button)
            self.controls.append(button)
        self.model = QComboBox()
        self.model.addItem("文档算法 Demo", "adaptive_demo")
        self.model.addItem("本地 YOLO 权重", "yolo")
        self.model.setCurrentIndex(0 if config.detector == "adaptive_demo" else 1)
        toolbar.addWidget(self.model)
        layout.addLayout(toolbar)
        actions = QHBoxLayout()
        self.sequence = QCheckBox("同一目标连续帧（共享基准）")
        actions.addWidget(self.sequence)
        self.input_label = QLabel("请选择图片、文件夹或演示序列")
        actions.addWidget(self.input_label, 1)
        self.start_button = QPushButton("开始检测")
        self.start_button.setStyleSheet("background:#1768d0;color:white;border:0;padding:10px 25px;")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        layout.addLayout(actions)
        panels = QSplitter(Qt.Horizontal)
        self.original = ImagePanel("原始图像")
        self.result_panel = ImagePanel("融合图像与检测框")
        for title, panel in [
            ("01 / 原始可见光", self.original),
            ("02 / 融合结果与异常定位", self.result_panel),
        ]:
            container = QWidget()
            inner = QVBoxLayout(container)
            inner.setContentsMargins(0, 4, 0, 0)
            inner.addWidget(QLabel(title))
            inner.addWidget(panel)
            panels.addWidget(container)
        layout.addWidget(panels, 4)
        self.stats = QLabel("候选 0    匹配 0    已过滤 0    推理 — ms    干扰系数 —")
        self.stats.setStyleSheet(
            "font-size:16px;font-weight:600;padding:12px;background:#e0ebfa;border-radius:6px;"
        )
        layout.addWidget(self.stats)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["类别", "匹配分数", "坐标 x1,y1,x2,y2", "状态", "融合", "判定依据"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, 2)
        self.diagnostics = QLabel("干扰数据：配置模拟值；NIR：未加载；基准：尚未建立")
        self.diagnostics.setWordWrap(True)
        layout.addWidget(self.diagnostics)
        bottom = QHBoxLayout()
        self.save_label = QLabel("每次检测自动保存：原图 / 融合图 / 标注图 / JSON / SQLite")
        bottom.addWidget(self.save_label, 1)
        for text, callback in [
            ("保存当前结果副本", self.save_copy),
            ("历史记录", self.show_history),
            ("导出历史 CSV", self.export_csv),
        ]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            bottom.addWidget(button)
        layout.addLayout(bottom)
        self.statusBar().showMessage("就绪 · 建议点击「演示序列」体验多帧检测")

    def select_paths(self, paths, sequence=False):
        self.paths, self.capture = list(paths), None
        self.sequence.setChecked(sequence)
        self.input_label.setText(f"已选 {len(self.paths)} 张图像")
        self.input_label.setToolTip("\n".join(map(str, self.paths)))
        if self.paths:
            try:
                self.original.set_image(read_image(self.paths[0]))
            except Exception as exc:
                self.show_error(str(exc))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(ROOT / "data/input"), "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.select_paths([Path(path)])

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", str(ROOT / "data/input"))
        if path:
            self.select_paths(image_files(Path(path)))

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", str(ROOT / "data/input"), "视频 (*.avi *.mp4 *.mov *.mkv)"
        )
        if path:
            self.capture, self.paths = path, []
            self.input_label.setText(f"视频：{Path(path).name}")
            self.sequence.setChecked(True)

    def open_camera(self):
        self.capture, self.paths = self.config.camera_index, []
        self.input_label.setText(f"USB 摄像头 {self.capture} · 开始检测后打开")
        self.sequence.setChecked(True)

    def open_demo(self):
        try:
            from scripts.generate_demo import generate

            folder = ROOT / "data/input/demo"
            if not (folder / "frame_000.png").exists():
                generate(folder)
            self.select_paths(image_files(folder), sequence=True)
        except Exception as exc:
            self.show_error(str(exc))

    def settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("算法与采集参数")
        form = QFormLayout(dialog)
        controls = {}
        specs = [
            ("candidate_threshold", "候选残差阈值", 1, 255),
            ("min_area", "最小面积 px²", 1, 100000),
            ("match_threshold", "匹配阈值", 0.01, 1),
            ("pseudo_threshold", "过滤阈值", 0, 0.99),
            ("k1", "灰度漂移 k1", 5, 15),
            ("k", "梯度偏差 k", 0, 100),
            ("light_lux", "模拟光强 lux", 0, 10000),
            ("temperature_c", "模拟温度 ℃", 20, 60),
            ("reflectance", "模拟反光率", 0.1, 0.9),
            ("camera_index", "USB 索引", 0, 20),
        ]
        for key, title, low, high in specs:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(0 if key in {"camera_index", "min_area"} else 3)
            spin.setValue(getattr(self.config, key))
            form.addRow(title, spin)
            controls[key] = spin
        model_path = QLineEdit(self.config.model_path)
        form.addRow("本地模型路径", model_path)
        hint = QLabel("无旁车传感数据时使用模拟值；完整配置见 config/config.json。")
        form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        form.addRow(buttons)
        buttons.rejected.connect(dialog.reject)

        def accept():
            try:
                values = {key: control.value() for key, control in controls.items()}
                for key in ("camera_index", "min_area"):
                    values[key] = int(values[key])
                updated = replace(self.config, **values, model_path=model_path.text())
                updated.validate()
                self.config = updated
                dialog.accept()
            except Exception as exc:
                QMessageBox.warning(dialog, "参数错误", str(exc))

        buttons.accepted.connect(accept)
        dialog.exec()

    def start(self):
        if self.worker and self.worker.isRunning():
            return
        if not self.paths and self.capture is None:
            self.show_error("请先选择图片、演示序列、视频或摄像头")
            return
        self.errors.clear()
        self.processed_count = 0
        cfg = replace(self.config, detector=self.model.currentData())
        self.worker = DetectionWorker(cfg, self.paths, self.capture, self.sequence.isChecked(), self)
        self.worker.result_ready.connect(self.display_result)
        self.worker.error.connect(self.show_error)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.finished)
        for control in self.controls + [self.start_button, self.model, self.sequence]:
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("正在启动检测…")
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.statusBar().showMessage("停止请求已发送，等待当前帧安全保存…")

    def display_result(self, result):
        try:
            self.last_result = result
            self.processed_count += 1
            self.original.set_image(result.original)
            self.result_panel.set_image(annotate(result))
            self.table.setRowCount(len(result.defects))
            for row, defect in enumerate(result.defects):
                values = [
                    defect.label,
                    f"{defect.confidence:.3f}",
                    str(defect.bbox),
                    STATUS[defect.status],
                    defect.fusion,
                    defect.reason,
                ]
                for col, value in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(value))
            matched = sum(d.status == "matched" for d in result.defects)
            filtered = sum(d.status == "filtered" for d in result.defects)
            c = result.metrics.get("c_total", 0)
            self.stats.setText(
                f"候选 {len(result.defects)}    匹配 {matched}    已过滤 {filtered}    "
                f"推理 {result.inference_ms:.1f} ms    干扰系数 {c:.3f}    已处理 {self.processed_count} 帧"
            )
            self.diagnostics.setText(
                f"干扰来源：{result.interference['source']}  |  NIR："
                f"{result.metrics.get('nir_mode', '未参与')}  |  权重："
                f"{[round(v, 3) for v in result.metrics.get('weights', [])]}  |  "
                f"匹配阈值：{result.metrics.get('match_threshold', '—')}\n"
                f"参考帧：{result.metrics.get('reference_used') or '无（本帧用于建立基准）'}"
            )
            self.save_label.setText(f"已保存 · {result.result_id[:12]} · PNG / JSON / SQLite")
            self.save_label.setToolTip(result.paths.get("json", ""))
        except Exception as exc:
            logger.exception("显示结果失败")
            self.show_error(str(exc))
        finally:
            if self.worker:
                self.worker.displayed.set()

    def finished(self):
        for control in self.controls + [self.start_button, self.model, self.sequence]:
            control.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage(f"任务结束 · 已处理 {self.processed_count} 帧 · 错误 {len(self.errors)}")
        self.run_finished.emit()
        if self.closing:
            QTimer.singleShot(0, self.close)

    def show_error(self, message):
        self.errors.append(message)
        self.statusBar().showMessage(f"错误：{message}")
        self.diagnostics.setText(f"错误：{message}\n详细信息见 data/output/app.log；坏图会跳过继续处理。")
        logger.error(message)

    def save_copy(self):
        if self.last_result is None:
            self.show_error("暂无检测结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存当前标注图", "检测结果.png", "PNG (*.png)")
        if path:
            try:
                from services.storage import write_image

                write_image(Path(path), annotate(self.last_result))
                self.statusBar().showMessage(f"已保存：{path}")
            except Exception as exc:
                self.show_error(str(exc))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出全部历史", "检测历史.csv", "CSV (*.csv)")
        if path:
            try:
                Storage(self.config.path(self.config.output_dir)).export_csv(Path(path))
                self.statusBar().showMessage(f"已导出：{path}")
            except Exception as exc:
                self.show_error(str(exc))

    def show_history(self):
        try:
            rows = Storage(self.config.path(self.config.output_dir)).history()
        except Exception as exc:
            self.show_error(str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("历史记录 · 最近200帧（CSV导出全部）")
        dialog.resize(1000, 650)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(rows), 5)
        table.setHorizontalHeaderLabels(["时间 UTC", "输入", "算法", "数量", "推理 ms"])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for i, row in enumerate(rows):
            for j, key in enumerate(["timestamp", "source", "detector", "count", "inference_ms"]):
                table.setItem(i, j, QTableWidgetItem(str(row[key])))
        layout.addWidget(table)
        panel = ImagePanel("选择历史行查看标注结果")
        layout.addWidget(panel)
        details = QTextEdit()
        details.setReadOnly(True)
        details.setMaximumHeight(100)
        layout.addWidget(details)

        def selected(row, _column):
            item = rows[row]
            try:
                panel.set_image(read_image(Path(item["paths"]["result"])))
                details.setPlainText(
                    json.dumps(
                        {
                            "interference": item["interference"],
                            "metrics": item["metrics"],
                            "paths": item["paths"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            except Exception as exc:
                details.setPlainText(str(exc))

        table.cellClicked.connect(selected)
        dialog.exec()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.closing = True
            self.stop()
            event.ignore()
        else:
            event.accept()
