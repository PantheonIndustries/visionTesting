import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
yunet = cv.FaceDetectorYN.create("face_detection_yunet_2026may.onnx", "", (0, 0))

if not cap.isOpened():
    print("the camera can't be opened :(")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not readable :(")
        break

    height, width, _ = frame.shape
    yunet.setInputSize((width, height))

    retval, faces = yunet.detect(frame)

    if faces is not None:
        for face in faces:
            bbox = face[0:4].astype(np.int32)
            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[0]+bbox[2], bbox[1]+bbox[3])
            cv.rectangle(frame, top_left, bottom_right, (0,255,0), 2)

            cv.putText(frame,str(face[14]),top_left,cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            landmarks = face[4:14].astype(np.int32).reshape((5,2))
            for idx, landmark in enumerate(landmarks):
                cv.circle(frame,landmark,2,(255,0,0),2)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()