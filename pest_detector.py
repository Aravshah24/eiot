from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime with a clearer error message
    YOLO = None


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "best.pt"


def _resolve_model_path(model_path: str | Path | None) -> Path:
    resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not resolved_path.is_absolute():
        resolved_path = (Path(__file__).resolve().parent / resolved_path).resolve()

    if not resolved_path.exists() and model_path is None:
        legacy_path = (Path(__file__).resolve().parent / "pest-detection" / "best.pt").resolve()
        if legacy_path.exists():
            return legacy_path

    return resolved_path


def _class_name(model: Any, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _normalize_label(label: str) -> str:
    return label.replace("_", " ").strip()


@lru_cache(maxsize=1)
def load_pest_model(model_path: str | Path | None = None) -> Any:
    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. Install it with 'pip install ultralytics' before running pest detection."
        )

    resolved_path = _resolve_model_path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Pest model not found: {resolved_path}")

    return YOLO(str(resolved_path))


def detect_pest(
    frame,
    model: Any | None = None,
    model_path: str | Path | None = None,
    confidence_threshold: float = 0.6,
) -> dict[str, Any] | None:
    if frame is None:
        return None

    pest_model = model or load_pest_model(model_path)
    results = pest_model.predict(frame, verbose=False)

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
            pest_name = _normalize_label(_class_name(result, class_id))
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]

            candidate = {
                "pest_detected": True,
                "pest_type": pest_name,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
            }

            if best_detection is None or candidate["confidence"] > best_detection["confidence"]:
                best_detection = candidate

    return best_detection