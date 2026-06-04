"""Quick self-test: load both YOLO models and run inference on a synthetic frame."""
from __future__ import annotations

import numpy as np

from intrusion_detector import load_intrusion_model, detect_intrusion, DEFAULT_POLYGON_POINTS
from pest_detector import load_pest_model, detect_pest


def main() -> None:
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    print("=== Intrusion model (yolov8n.pt) ===")
    im = load_intrusion_model()
    print("classes:", len(im.names), "->", list(im.names.values())[:10], "...")
    r = detect_intrusion(frame, polygon_points=DEFAULT_POLYGON_POINTS)
    print("detection on random frame:", r)

    print("\n=== Pest model (best.pt) ===")
    pm = load_pest_model()
    print("classes:", pm.names)
    r = detect_pest(frame)
    print("detection on random frame:", r)

    print("\nBoth models loaded and ran inference successfully.")


if __name__ == "__main__":
    main()
