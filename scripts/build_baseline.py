import argparse
from pathlib import Path
from services.detection import image_files
from services.storage import read_image
from vision.calibration import build_baseline


def main():
    parser = argparse.ArgumentParser(description="使用至少100帧同目标无干扰图像标定基准灰度")
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    def pairs():
        for path in image_files(args.folder):
            nir = path.with_name(path.stem + "_nir.png")
            yield read_image(path), read_image(nir) if nir.exists() else None

    count = build_baseline(pairs(), args.output)
    print(f"已用 {count} 帧建立 {args.output}；请配置 baseline_path，并保持相同视角与坐标")


if __name__ == "__main__":
    main()
