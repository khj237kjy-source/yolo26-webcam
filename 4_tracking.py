# 4. Object Tracking (객체추적) - YOLO26 + 웹캠
# 영상의 각 프레임에서 객체 좌표 + 고유 ID를 유지하며 추적
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

track_history = defaultdict(list)  # ID별 이동 경로

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # persist=True: 프레임 간 ID 유지 / tracker: bytetrack.yaml 또는 botsort.yaml
    result = model.track(frame, persist=True, conf=0.5,
                         tracker="bytetrack.yaml", verbose=False)[0]
    annotated = result.plot()

    if result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        ids = result.boxes.id.int().cpu().tolist()

        for (x, y, w, h), track_id in zip(boxes, ids):
            # 중심점 이동 경로 저장 (최근 30프레임)
            history = track_history[track_id]
            history.append((float(x), float(y)))
            if len(history) > 30:
                history.pop(0)

            points = np.array(history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [points], False, (0, 255, 255), 2)

    cv2.imshow("YOLO26 Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
