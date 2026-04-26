from __future__ import annotations

import math


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _to_yolo_segmentation(class_id: int, points: list[tuple[float, float]]) -> str:
    serialized_points = " ".join(
        f"{_clip01(x):.6f} {_clip01(y):.6f}" for x, y in points
    )
    return f"{class_id} {serialized_points}"


def _build_wheel_polygon(frame_index: int) -> list[tuple[float, float]]:
    cx = 0.34 + 0.18 * math.sin(frame_index * 0.2)
    cy = 0.76
    rx = 0.07
    ry = 0.09
    vertices = 10

    points: list[tuple[float, float]] = []
    for i in range(vertices):
        angle = 2 * math.pi * i / vertices
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        points.append((_clip01(x), _clip01(y)))
    return points


def build_stub_masks(frame_index: int) -> list[dict[str, object]]:
    track_limits_polygon = [
        (0.06, 0.18),
        (0.91, 0.14),
        (0.98, 0.87),
        (0.02, 0.90),
    ]
    wheel_polygon = _build_wheel_polygon(frame_index)

    masks = [
        {
            "model_name": "track_markup_stub",
            "class_id": 0,
            "class_name": "track_limits",
            "points": [[x, y] for x, y in track_limits_polygon],
            "yolo_segmentation": _to_yolo_segmentation(0, track_limits_polygon),
        },
        {
            "model_name": "wheel_segmentation_stub",
            "class_id": 1,
            "class_name": "wheel",
            "points": [[x, y] for x, y in wheel_polygon],
            "yolo_segmentation": _to_yolo_segmentation(1, wheel_polygon),
        },
    ]
    return masks
