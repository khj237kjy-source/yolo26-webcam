# 3. Segmentation (분할) - YOLO26 + 웹캠
# 박스보다 정확한 픽셀 단위 윤곽(폴리곤 좌표, 누끼)을 반환
import cv2
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")  # 분할 모델

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, conf=0.5, verbose=False)[0]

    # 마스크 윤곽 좌표 정보 출력
    if result.masks is not None:
        for polygon, cls_id in zip(result.masks.xy, result.boxes.cls):
            cls_name = result.names[int(cls_id)]
            print(f"{cls_name}: 윤곽점 {len(polygon)}개, 첫 좌표 {polygon[0].astype(int).tolist()}")

    annotated = result.plot()  # 마스크 + 박스 + 라벨 그리기
    cv2.imshow("YOLO26 Segmentation", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
