import cv2 as cv
from pupil_apriltags import Detector

cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

detector = Detector(families="tag36h11")

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't read frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    for tag in tags:
        corners = tag.corners.astype(int)
        for i in range(4):
            cv.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
        cx, cy = map(int, tag.center)
        cv.putText(frame, f"ID {tag.tag_id}", (cx - 20, cy), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"Tag {tag.tag_id} at ({cx}, {cy})")

    cv.imshow("AprilTags", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
