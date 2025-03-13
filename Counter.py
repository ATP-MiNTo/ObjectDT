import cv2
import torch
import pandas as pd
from collections import defaultdict
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_your_model/yolo11n.pt')

cap = cv2.VideoCapture(0)

object_count = defaultdict(list)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.pandas().xywh[0]

    for index, row in detections.iterrows():
        class_name = row['name']
        timestamp = time.time() - start_time
        object_count[class_name].append(timestamp)

    frame = results.render()[0]
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

output_data = []
for class_name, timestamps in object_count.items():
    for timestamp in timestamps:
        output_data.append([class_name, timestamp])

df = pd.DataFrame(output_data, columns=['Class', 'Timestamp'])

df.to_csv('object_appearance.csv', index=False)
print("Data saved to 'object_appearance.csv'.")
