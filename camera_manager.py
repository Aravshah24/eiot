from __future__ import annotations

import os
import threading
import time
from typing import Any, Sequence

import cv2
import numpy as np

from intrusion_detector import DEFAULT_POLYGON_POINTS, detect_intrusion
from pest_detector import detect_pest


def _resolve_capture_source(source: int | str) -> int | str:
    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return source


def _open_capture(source: int | str) -> cv2.VideoCapture:
    capture_source = _resolve_capture_source(source)
    backend = cv2.CAP_DSHOW if os.name == "nt" and isinstance(capture_source, int) else cv2.CAP_ANY
    capture = cv2.VideoCapture(capture_source, backend)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class CameraManager:
    def __init__(
        self,
        source: int | str = 0,
        polygon_points: Sequence[Sequence[int]] | np.ndarray | None = None,
        intrusion_interval: float = 0.15,
        pest_interval: float = 60.0,
        show_backend_window: bool = True,
        backend_window_name: str = "Smart Farm Backend View",
    ) -> None:
        self.source = source
        self.polygon_points = np.asarray(
            polygon_points if polygon_points is not None else DEFAULT_POLYGON_POINTS,
            dtype=np.int32,
        )
        self.intrusion_interval = intrusion_interval
        self.pest_interval = pest_interval
        self.show_backend_window = show_backend_window
        self.backend_window_name = backend_window_name

        self.system_state: dict[str, Any] = {
            "intrusion": False,
            "intrusion_object": None,
            "pest_detected": False,
            "pest_type": None,
            "last_pest_check": None,
        }

        self.latest_frame: np.ndarray | None = None
        self.latest_annotated_frame: np.ndarray | None = None
        self.latest_intrusion_result: dict[str, Any] | None = None
        self.latest_pest_result: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.backend_window_error: str | None = None

        self._capture: cv2.VideoCapture | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_intrusion_check = 0.0
        self._last_pest_check = 0.0

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        self._capture = _open_capture(self.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.source}")

        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, name="CameraManager", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        if self.show_backend_window:
            try:
                cv2.destroyWindow(self.backend_window_name)
            except cv2.error:
                pass

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.system_state)

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._capture is None:
                self.last_error = "Camera capture is not initialized"
                time.sleep(0.1)
                continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                self.last_error = "Failed to read frame from camera"
                time.sleep(0.05)
                continue

            now = time.monotonic()
            intrusion_result = self.latest_intrusion_result
            pest_result = self.latest_pest_result

            if now - self._last_intrusion_check >= self.intrusion_interval:
                intrusion_result = detect_intrusion(frame, polygon_points=self.polygon_points)
                self._last_intrusion_check = now

                with self._lock:
                    self.latest_intrusion_result = intrusion_result
                    self.system_state["intrusion"] = bool(intrusion_result)
                    self.system_state["intrusion_object"] = intrusion_result["object"] if intrusion_result else None

            if now - self._last_pest_check >= self.pest_interval:
                pest_result = detect_pest(frame)
                self._last_pest_check = now

                with self._lock:
                    self.latest_pest_result = pest_result
                    self.system_state["pest_detected"] = bool(pest_result)
                    self.system_state["pest_type"] = pest_result["pest_type"] if pest_result else None
                    self.system_state["last_pest_check"] = time.strftime("%Y-%m-%d %H:%M:%S")

            annotated_frame = self._annotate_frame(frame, intrusion_result, pest_result)


            if self.show_backend_window:
                self._display_backend_window(annotated_frame)

            with self._lock:
                self.latest_frame = frame.copy()
                self.latest_annotated_frame = annotated_frame

            time.sleep(0.001)

    def _display_backend_window(self, frame: np.ndarray) -> None:
        try:
            cv2.imshow(self.backend_window_name, frame)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.backend_window_error = str(exc)
            self.show_backend_window = False

    def _annotate_frame(
        self,
        frame: np.ndarray,
        intrusion_result: dict[str, Any] | None,
        pest_result: dict[str, Any] | None,
    ) -> np.ndarray:
        annotated = frame.copy()

        cv2.polylines(annotated, [self.polygon_points], True, (0, 255, 255), 2)
        cv2.putText(
            annotated,
            "Virtual Geofence",
            (int(self.polygon_points[0][0]), max(30, int(self.polygon_points[0][1]) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if intrusion_result:
            x1, y1, x2, y2 = intrusion_result["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(annotated, tuple(intrusion_result["centroid"]), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"Intrusion: {intrusion_result['object']} ({intrusion_result['confidence']:.2f})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        if pest_result:
            x1, y1, x2, y2 = pest_result["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(
                annotated,
                f"Pest: {pest_result['pest_type']} ({pest_result['confidence']:.2f})",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 165, 0),
                2,
            )

        return annotated


def create_camera_manager(
    source: int | str = 0,
    polygon_points: Sequence[Sequence[int]] | np.ndarray | None = None,
    intrusion_interval: float = 0.15,
    pest_interval: float = 60.0,
    show_backend_window: bool = True,
    backend_window_name: str = "Smart Farm Backend View",
) -> CameraManager:
    manager = CameraManager(
        source=source,
        polygon_points=polygon_points,
        intrusion_interval=intrusion_interval,
        pest_interval=pest_interval,
        show_backend_window=show_backend_window,
        backend_window_name=backend_window_name,
    )
    manager.start()
    return manager