from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime with a clearer error message
    YOLO = None


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "yolov8n.pt"
DEFAULT_INTRUSION_CLASSES = {"person", "dog", "cat", "cow", "horse", "sheep"}
DEFAULT_POLYGON_POINTS = np.array(
    [(200, 200), (400, 200), (450, 350), (350, 450), (200, 400)],
    dtype=np.int32,
)


def _resolve_model_path(model_path: str | Path | None) -> Path:
    resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not resolved_path.is_absolute():
        resolved_path = (Path(__file__).resolve().parent / resolved_path).resolve()
    return resolved_path


def _class_name(model: Any, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _to_polygon_array(points: Sequence[Sequence[int]] | np.ndarray | None) -> np.ndarray:
    if points is None:
        return DEFAULT_POLYGON_POINTS.copy()

    polygon = np.asarray(points, dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon_points must be a sequence of (x, y) pairs")
    return polygon


@lru_cache(maxsize=1)
def load_intrusion_model(model_path: str | Path | None = None) -> Any:
    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. Install it with 'pip install ultralytics' before running intrusion detection."
        )

    resolved_path = _resolve_model_path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Intrusion model not found: {resolved_path}")

    return YOLO(str(resolved_path))


def _point_inside_polygon(polygon: np.ndarray, centroid: tuple[int, int]) -> bool:
    return cv2.pointPolygonTest(polygon, centroid, False) >= 0


def detect_intrusion(
    frame,
    polygon_points: Sequence[Sequence[int]] | np.ndarray | None = None,
    model: Any | None = None,
    model_path: str | Path | None = None,
    confidence_threshold: float = 0.35,
    target_classes: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    if frame is None:
        return None

    intrusion_model = model or load_intrusion_model(model_path)
    polygon = _to_polygon_array(polygon_points)
    allowed_classes = set(target_classes or DEFAULT_INTRUSION_CLASSES)

    results = intrusion_model.predict(frame, verbose=False)
    best_detection: dict[str, Any] | None = None

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < confidence_threshold:
                continue

            class_id = int(box.cls[0])
            class_name = _class_name(result, class_id)
            if class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            if not _point_inside_polygon(polygon, centroid):
                continue

            candidate = {
                "intrusion": True,
                "object": class_name,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
                "centroid": centroid,
            }

            if best_detection is None or candidate["confidence"] > best_detection["confidence"]:
                best_detection = candidate

    return best_detection