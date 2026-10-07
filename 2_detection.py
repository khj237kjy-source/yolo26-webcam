# 2. Object Detection (객체인식) - YOLO26 + 웹캠
# 객체의 위치를 사각형 박스 좌표(x1, y1, x2, y2)로 반환
import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # 탐지 모델 (COCO 80개 클래스)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, conf=0.5, verbose=False)[0]

    # 박스 좌표 정보 출력
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = result.names[int(box.cls)]
        conf = float(box.conf)
        print(f"{cls_name} {conf:.2f} -> ({x1}, {y1}, {x2}, {y2})")

    annotated = result.plot()  # 박스 + 라벨 그리기
    cv2.imshow("YOLO26 Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
