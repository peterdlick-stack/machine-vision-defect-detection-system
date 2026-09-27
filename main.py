"""Desktop entrypoint, headless demo, and real Qt worker smoke verification."""

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="基于机器视觉的缺陷检测系统")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--demo", action="store_true", help="无界面执行演示序列并保存")
    parser.add_argument("--gui-smoke", action="store_true", help="启动真实GUI及Worker，完成演示后截图退出")
    parser.add_argument("--output", type=Path, help="覆盖结果目录")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output:
        cfg = replace(cfg, output_dir=str(args.output.resolve()))
    output = cfg.path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(output / "app.log", encoding="utf-8"), logging.StreamHandler()],
    )
    if args.demo or args.gui_smoke:
        from scripts.generate_demo import generate

        folder = generate()
    if args.demo:
        from services.detection import DetectionService, image_files

        service = DetectionService(cfg)
        results = [service.detect_file(path, sequence=True).to_dict() for path in image_files(folder)]
        service.storage.export_csv(output / "demo_history.csv")
        summary = [
            {
                "source": Path(r["source"]).name,
                "count": r["count"],
                "ms": r["inference_ms"],
                "matched": sum(d["status"] == "matched" for d in r["defects"]),
            }
            for r in results
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from app.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("自适应缺陷检测")
    window = MainWindow(cfg)
    window.show()
    if args.gui_smoke:

        def complete():
            window.grab().save(str(output / "gui_smoke.png"))
            passed = window.processed_count == 6 and not window.errors and window.last_result is not None
            (output / "gui_smoke.json").write_text(
                json.dumps({"passed": passed, "processed": window.processed_count, "errors": window.errors}),
                encoding="utf-8",
            )
            QTimer.singleShot(50, lambda: app.exit(0 if passed else 1))

        window.run_finished.connect(complete)
        window.open_demo()
        QTimer.singleShot(200, window.start)
        QTimer.singleShot(60000, window.stop)
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("程序启动/执行失败")
        raise SystemExit(1)
