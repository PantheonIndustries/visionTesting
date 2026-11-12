import asyncio
import cv2 as cv
import numpy as np
import websockets


async def video(websocket):
    print(f"Client connected {websocket.remote_address} :)")

    cap = cv.VideoCapture(0)

    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, params)

    if not cap.isOpened():
        print("the camera can't be opened :(")
        exit()
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Frame not readable :(")
                break

            # put processing code here :)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None:
                cv.aruco.drawDetectedMarkers(frame, corners, ids)
                for tag_corners, tag_id in zip(corners, ids.flatten()):
                    cx, cy = tag_corners[0].mean(axis=0).astype(int)
                    print(f"Tag {tag_id} at ({cx}, {cy})")

            success, encoded_image = cv.imencode(".jpg", frame)
            if not success:
                continue

            buffer = encoded_image.tobytes()

            await websocket.send(buffer)
            await asyncio.sleep(0.033)
    except websockets.exceptions.ConnectionClosed:
        print(f"Client gone {websocket.remote_address} :(")
    finally:
        cap.release()


async def main():
    async with websockets.serve(video, "0.0.0.0", 8765):
        print("WebSocket Video Server started on ws://localhost:8765")
        await asyncio.Future()  # Keeps the server running indefinitely


if __name__ == "__main__":
    asyncio.run(main())
