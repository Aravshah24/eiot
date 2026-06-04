from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 Nano
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Restricted Polygon
polygon_points = np.array([
    (200, 200),
    (400, 200),
    (450, 350),
    (350, 450),
    (200, 400)
], np.int32)

# Classes we care about
target_classes = {
    "person",
    "dog",
    "cat",
    "cow",
    "horse",
    "sheep"
}

window_name = "Smart Farm Intrusion Detection"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    intrusion_detected = False
    intrusion_label = ""

    # Draw polygon
    cv2.polylines(
        frame,
        [polygon_points],
        True,
        (0, 255, 255),
        3
    )

    results = model(frame, verbose=False)

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            if class_name not in target_classes:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(box.conf[0])

            centroid = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

            # Draw detection
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                centroid,
                5,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Check if centroid is inside polygon
            inside = cv2.pointPolygonTest(
                polygon_points,
                centroid,
                False
            )

            if inside >= 0:

                intrusion_detected = True
                intrusion_label = class_name

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

    if intrusion_detected:

        cv2.putText(
            frame,
            f"ALERT: {intrusion_label.upper()} INSIDE RESTRICTED AREA",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()