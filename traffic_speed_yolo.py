import cv2
import math
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# ======== CAMERA STREAM ========
CAMERA_URL = "http://192.0.0.4:8080/video"   # change IP

SPEED_LIMIT = 10
METER_PER_PIXEL = 0.126

WINDOW_FRAMES = 15
CONFIDENCE = 0.4

VEHICLE_CLASSES = [2,3]

MIN_PIXEL_MOVE = 20
MIN_SPEED_THRESHOLD = 3

os.makedirs("overspeed_vehicles", exist_ok=True)

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=20)

cap = cv2.VideoCapture(CAMERA_URL)

track_data = {}

print("Live monitoring started")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(720,400))

    results = model(frame, conf=CONFIDENCE, imgsz=416, verbose=False)[0]

    detections = []

    for box,cls,conf in zip(results.boxes.xyxy,
                            results.boxes.cls,
                            results.boxes.conf):

        if int(cls) not in VEHICLE_CLASSES:
            continue

        x1,y1,x2,y2 = map(int,box)

        w = x2-x1
        h = y2-y1

        if w < 30 or h < 30:
            continue

        detections.append(([x1,y1,w,h], float(conf)))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        x1,y1,x2,y2 = map(int,track.to_ltrb())

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        if track_id not in track_data:

            track_data[track_id] = {
                "positions": deque(maxlen=WINDOW_FRAMES),
                "speed":0
            }

        track_data[track_id]["positions"].append(cy)

        speed = track_data[track_id]["speed"]

        if len(track_data[track_id]["positions"]) == WINDOW_FRAMES:

            start = track_data[track_id]["positions"][0]
            end = track_data[track_id]["positions"][-1]

            pixel_move = abs(end-start)

            dt = WINDOW_FRAMES / 30   # assume ~30 fps

            if pixel_move > MIN_PIXEL_MOVE:

                distance_m = pixel_move * METER_PER_PIXEL
                new_speed = (distance_m / dt) * 3.6

                if new_speed < MIN_SPEED_THRESHOLD:
                    new_speed = 0

                speed = 0.7 * track_data[track_id]["speed"] + 0.3 * new_speed

        track_data[track_id]["speed"] = speed

        color = (0,0,255) if speed > SPEED_LIMIT else (0,255,0)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        label = f"{int(speed)} km/h"

        if speed > SPEED_LIMIT:
            label += "  OVERSPEED"

        cv2.putText(frame,
                    label,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    cv2.imshow("Vehicle Speed Monitoring (Live)",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()