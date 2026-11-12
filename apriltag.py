import cv2 as cv

cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
params = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, params)

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't read frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            cx, cy = tag_corners[0].mean(axis=0).astype(int)
            print(f"Tag {tag_id} at ({cx}, {cy})")

    cv.imshow("AprilTags", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
