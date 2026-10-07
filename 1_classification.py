# 1. Classification (분류) - YOLO26 + 웹캠
# 이미지 전체가 어떤 클래스인지 판별 (Teachable Machine과 같은 방식)
import cv2
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")  # 분류 모델 (ImageNet 1000개 클래스)

cap = cv2.VideoCapture(0)  # 0 = 기본 웹캠
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False)[0]
    probs = result.probs

    # 상위 5개 클래스 표시
    for i, cls_id in enumerate(probs.top5):
        name = result.names[cls_id]
        conf = probs.top5conf[i].item()
        text = f"{i + 1}. {name}: {conf:.2f}"
        cv2.putText(frame, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO26 Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # q 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
