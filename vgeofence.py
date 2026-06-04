import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    cv2.rectangle(
        frame,
        (150,100),
        (500,350),
        (0,0,255),
        2
    )

    cv2.putText(
        frame,
        "Restricted Area",
        (150,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.imshow("Virtual Geofence", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()