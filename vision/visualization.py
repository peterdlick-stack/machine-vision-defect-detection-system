import cv2

COLORS = {
    "matched": (83, 190, 36),
    "unconfirmed": (0, 177, 240),
    "filtered": (140, 140, 140),
    "model": (83, 190, 36),
}


def annotate(result):
    image = result.image.copy()
    for index, defect in enumerate(result.defects):
        if defect.status == "filtered":
            continue
        x1, y1, x2, y2 = defect.bbox
        color = COLORS[defect.status]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"#{index + 1} {defect.status} {defect.confidence:.2f}"
        cv2.putText(image, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    count = sum(d.status != "filtered" for d in result.defects)
    cv2.putText(
        image,
        f"{result.detector} | count {count} | {result.inference_ms:.1f} ms",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 210, 120),
        1,
    )
    return image
