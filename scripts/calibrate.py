"""Run using python -m scripts.calibrate SAMPLES.json [--incremental]."""

import argparse
import json
from services.learning import calibrate, incremental_update, load_state, save_state
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="使用独立实测数据进行标定/受监督SGD")
    parser.add_argument("samples")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    with open(args.samples, encoding="utf-8") as stream:
        rows = json.load(stream)
    path = config.path(config.learning_state)
    if args.incremental:
        state = load_state(path, config)
        if state is None:
            raise ValueError("请先完成至少100组初始标定")
        state, count = incremental_update(state, rows, config)
        print(f"新增样本更新: {count}；未满批样本请保留在输入文件下次提交")
    else:
        state = calibrate(rows)
    save_state(path, state)
    print(json.dumps({k: v for k, v in state.items() if k != "processed_ids"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
