import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("the camera can't be opened :(")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not readable :(")
        break

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()