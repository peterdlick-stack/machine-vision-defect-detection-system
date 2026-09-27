"""Supervised calibration and incremental SGD, never trained on its own predictions."""

import json
from pathlib import Path
import numpy as np
from vision.adaptive import dynamic_weights
from vision.types import Interference


def project_alpha(values):
    """Euclidean projection onto the document's bounded simplex."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.array([0.2, 0.3, 0.2]), np.array([0.4, 0.5, 0.4])
    left, right = -10.0, 10.0
    for _ in range(80):
        mid = (left + right) / 2
        if np.clip(values - mid, lo, hi).sum() > 1:
            left = mid
        else:
            right = mid
    return np.clip(values - (left + right) / 2, lo, hi)


def samples_arrays(rows):
    x, y = [], []
    for row in rows:
        x.append(Interference(**row["interference"]).normalized())
        target = float(row["measured_c"])
        if not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("measured_c 必须是[0,1]内的独立实测值")
        y.append(target)
    return np.asarray(x), np.asarray(y)


def calibrate(rows):
    if len(rows) < 100:
        raise ValueError("初始标定需要至少100组独立实测样本")
    x, y = samples_arrays(rows)
    if np.linalg.matrix_rank(x) < 3:
        raise ValueError("标定样本缺乏干扰变化，无法估计三个系数")
    alpha = project_alpha(np.linalg.lstsq(x, y, rcond=None)[0])
    # Refine constrained least squares rather than merely clipping unconstrained values.
    for _ in range(3000):
        updated = project_alpha(alpha - 0.2 * (x.T @ (x @ alpha - y)) / len(y))
        if np.linalg.norm(updated - alpha) < 1e-10:
            break
        alpha = updated
    error = float(np.max(abs(x @ alpha - y)))
    return {
        "alpha": alpha.tolist(),
        "weight_correction": [0.0, 0.0, 0.0, 0.0],
        "match_threshold": 0.75,
        "processed_ids": [],
        "calibration_count": len(rows),
        "calibration_max_error": error,
        "calibration_target_met": error <= 0.02,
    }


def load_state(path: Path, config):
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    alpha = np.asarray(state["alpha"], dtype=float)
    correction = np.asarray(state.get("weight_correction", [0, 0, 0, 0]), dtype=float)
    if alpha.shape != (3,) or not np.isfinite(alpha).all() or not np.allclose(alpha, project_alpha(alpha)):
        raise ValueError("学习状态 alpha 无效")
    if correction.shape != (4,) or not np.isfinite(correction).all():
        raise ValueError("学习状态权重无效")
    if not config.pseudo_threshold < state["match_threshold"] <= 1:
        raise ValueError("学习状态阈值无效")
    return state


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def incremental_update(state, rows, config):
    state = dict(state)
    processed = set(state.get("processed_ids", []))
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id重复，不能重复学习")
    new = [row for row in rows if str(row["sample_id"]) not in processed]
    count = len(new) // config.learning_batch * config.learning_batch
    if count == 0:
        return state, 0
    alpha = np.asarray(state["alpha"], dtype=float)
    correction = np.asarray(state.get("weight_correction", [0.0, 0.0, 0.0, 0.0]))
    threshold = float(state.get("match_threshold", config.match_threshold))
    errors = []
    # One pass over each newly completed batch, deterministic random SGD order.
    for offset in range(0, count, config.learning_batch):
        batch = new[offset : offset + config.learning_batch]
        x, y = samples_arrays(batch)
        for index in np.random.default_rng(offset).permutation(len(batch)):
            error = float(alpha @ x[index] - y[index])
            alpha = project_alpha(alpha - config.learning_rate * 2 * error * x[index])
            errors.append(abs(error))
            row = batch[index]
            if "match_label" in row:
                label = float(row["match_label"])
                sims = np.asarray(row["similarities"], dtype=float)
                if (
                    label not in (0, 1)
                    or sims.shape != (4,)
                    or not np.isfinite(sims).all()
                    or np.any((sims < 0) | (sims > 1))
                ):
                    raise ValueError("人工标签须为0/1，similarities须含4个[0,1]值")
                c = float(alpha @ x[index])
                weights = dynamic_weights(c, correction)
                score = float(weights @ sims)
                correction -= config.learning_rate * 2 * (score - label) * sims
                correction -= correction.mean()
                correction = np.clip(correction, -0.15, 0.15)
                false_negative = label == 1 and score < threshold
                false_positive = label == 0 and score >= threshold
                threshold += config.learning_rate * (int(false_positive) - int(false_negative))
                threshold = float(np.clip(threshold, config.pseudo_threshold + 0.01, 0.99))
            processed.add(str(row["sample_id"]))
    state.update(
        alpha=alpha.tolist(),
        weight_correction=correction.tolist(),
        match_threshold=threshold,
        processed_ids=sorted(processed),
        last_mean_error=float(np.mean(errors)),
        convergence_target_met=bool(max(errors) <= 0.01),
    )
    return state, count
