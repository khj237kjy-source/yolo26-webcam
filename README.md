# YOLO26 웹캠 실습: 분류 · 객체인식 · 분할 · 객체추적

> Ultralytics YOLO26 모델로 웹캠 영상을 실시간 분석하는 실습 강의노트입니다.
> 컴퓨터 비전의 대표 작업 4가지를 같은 구조의 코드로 비교해 봅니다.

---

## 학습 목표

1. 컴퓨터 비전의 4가지 작업(분류, 객체인식, 분할, 객체추적)의 차이를 설명할 수 있다.
2. Python 가상환경을 만들고 `ultralytics`, `opencv-python`을 설치할 수 있다.
3. 웹캠 영상을 프레임 단위로 읽어 YOLO26 모델로 추론하고 결과를 화면에 그릴 수 있다.
4. 실행 중 자주 만나는 오류(모듈 없음, 카메라 권한)를 스스로 해결할 수 있다.

---

## 1. 개념 정리: 컴퓨터 비전의 4가지 작업

| 작업 | 한국어 | 결과물 | 비유 | YOLO26 모델 |
|---|---|---|---|---|
| Classification | 분류 | 이미지 전체의 **클래스 이름 + 확률** | "이 사진은 고양이 사진이다" | `yolo26n-cls.pt` |
| Object Detection | 객체인식 | 객체마다 **사각형 박스 좌표** | "고양이가 (10,20)~(200,300)에 있다" | `yolo26n.pt` |
| Segmentation | 분할 | 객체마다 **윤곽선 좌표(폴리곤)**, 누끼 | "고양이 털 경계까지 정확히 오려낸다" | `yolo26n-seg.pt` |
| Object Tracking | 객체추적 | 영상에서 박스 좌표 + **고유 ID** | "1번 고양이가 왼쪽에서 오른쪽으로 이동했다" | `yolo26n.pt` + 트래커 |

- **분류**는 Teachable Machine과 같은 방식입니다. 물체가 *어디에* 있는지는 알려주지 않습니다.
- **객체인식**은 물체의 위치를 박스 4개 값 `(x1, y1, x2, y2)`로 알려줍니다.
- **분할**은 박스보다 훨씬 정확합니다. 물체의 외곽선을 점 수십~수백 개로 표현합니다.
- **객체추적**은 객체인식에 "기억"을 더한 것입니다. 프레임이 바뀌어도 같은 물체에 같은 ID를 붙입니다.

### 모델 이름 읽는 법

```
yolo26 n - seg .pt
  │    │    │
  │    │    └─ 작업 종류: (없음)=탐지, -cls=분류, -seg=분할
  │    └────── 크기: n(nano) < s < m < l < x  (클수록 정확하지만 느림)
  └─────────── 모델 세대: YOLO26
```

---

## 2. 실습 환경 준비

### 2-1. 가상환경 만들기

프로젝트마다 패키지를 따로 관리하기 위해 **가상환경(venv)** 을 사용합니다.

```bash
# Python 3.13으로 가상환경 생성
python3.13 -m venv .venv

# 가상환경 활성화 (macOS / Linux)
source .venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install ultralytics opencv-python
```

> 💡 **왜 Python 3.13인가요?**
> 가장 최신 버전(3.14)은 PyTorch 같은 딥러닝 라이브러리가 아직 지원하지 않을 수 있습니다.
> 딥러닝 실습에서는 한 단계 낮은 안정 버전을 쓰는 것이 안전합니다.

### 2-2. 설치 확인

```bash
python -c "import cv2, ultralytics; print(cv2.__version__, ultralytics.__version__)"
```

이 실습에서 사용한 버전: OpenCV 5.0.0 / ultralytics 8.4.174 / PyTorch 2.14.1

> 모델 파일(`.pt`)은 처음 실행할 때 자동으로 다운로드됩니다.

---

## 3. 공통 코드 구조

4개 파일은 모두 아래와 같은 뼈대로 되어 있습니다. **이 구조만 이해하면 나머지는 응용입니다.**

```python
import cv2
from ultralytics import YOLO

model = YOLO("모델파일.pt")        # ① 모델 불러오기
cap = cv2.VideoCapture(0)          # ② 웹캠 열기 (0 = 기본 카메라)

while True:
    ret, frame = cap.read()        # ③ 한 프레임 읽기
    if not ret:
        break

    result = model(frame)[0]       # ④ 추론
    annotated = result.plot()      # ⑤ 결과 그리기

    cv2.imshow("창 제목", annotated)  # ⑥ 화면에 표시
    if cv2.waitKey(1) & 0xFF == ord("q"):  # ⑦ q 키로 종료
        break

cap.release()                      # ⑧ 정리
cv2.destroyAllWindows()
```

| 단계 | 핵심 포인트 |
|---|---|
| ② | `VideoCapture(0)`에서 숫자를 바꾸면 다른 카메라를 쓸 수 있습니다. 파일 경로를 넣으면 동영상도 됩니다. |
| ④ | `model(frame)`은 결과 **리스트**를 돌려줍니다. 이미지 1장이므로 `[0]`으로 꺼냅니다. |
| ⑤ | `result.plot()`은 박스·라벨·마스크를 그린 이미지를 돌려주는 편리한 함수입니다. |
| ⑦ | `waitKey(1)`은 1ms 동안 키 입력을 기다립니다. 이게 없으면 창이 갱신되지 않습니다. |

---

## 4. 실습 코드 살펴보기

### 4-1. 분류 — `1_classification.py`

```python
result = model(frame, verbose=False)[0]
probs = result.probs

for i, cls_id in enumerate(probs.top5):
    name = result.names[cls_id]          # 클래스 번호 → 이름
    conf = probs.top5conf[i].item()      # 확률
```

- 결과는 `result.probs`에 들어 있습니다. 박스가 없습니다.
- `top5`: 확률이 높은 클래스 번호 5개, `top5conf`: 그 확률들.
- ImageNet 1000개 클래스(사람, 개, 키보드, 커피잔 등)로 분류합니다.

### 4-2. 객체인식 — `2_detection.py`

```python
result = model(frame, conf=0.5, verbose=False)[0]

for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])   # 박스 좌표
    cls_name = result.names[int(box.cls)]    # 클래스 이름
    conf = float(box.conf)                   # 신뢰도
```

- `conf=0.5`: 신뢰도 50% 미만인 결과는 버립니다. 값을 높이면 오탐이 줄고, 낮추면 더 많이 잡습니다.
- `box.xyxy`: 왼쪽 위 `(x1, y1)`, 오른쪽 아래 `(x2, y2)` 좌표.
- COCO 데이터셋 80개 클래스(person, chair, cup, cell phone 등)를 인식합니다.

**실제 실행 결과 (1920×1080 웹캠)**

```
person 0.89 -> (412, 243, 1604, 1080)   ← 화면 가운데 사람
person 0.75 -> (1694, 800, 1919, 1079)  ← 오른쪽 아래 가장자리 사람
chair  0.65 -> (118, 981, 434, 1079)    ← 왼쪽 아래 의자
```

### 4-3. 분할 — `3_segmentation.py`

```python
if result.masks is not None:
    for polygon, cls_id in zip(result.masks.xy, result.boxes.cls):
        print(f"윤곽점 {len(polygon)}개")
```

- 결과의 `result.masks`에 마스크가 들어 있습니다. 아무것도 못 찾으면 `None`이므로 꼭 확인합니다.
- `masks.xy`: 객체마다 외곽선을 이루는 점들의 좌표 배열(폴리곤).

**실제 실행 결과**

```
person: 윤곽점 281개, 첫 좌표 [1098, 456]
person: 윤곽점 46개,  첫 좌표 [1857, 879]
```

> 🔍 **비교해 보기**: 같은 사람을 객체인식은 숫자 **4개**(박스)로, 분할은 점 **281개**(윤곽)로 표현합니다.
> 그래서 분할은 배경 제거(누끼)처럼 정밀한 작업에 쓰입니다.

### 4-4. 객체추적 — `4_tracking.py`

```python
result = model.track(frame, persist=True, conf=0.5,
                     tracker="bytetrack.yaml", verbose=False)[0]

if result.boxes.id is not None:
    ids = result.boxes.id.int().cpu().tolist()   # 객체별 고유 ID
```

- `model(...)` 대신 **`model.track(...)`** 을 호출합니다.
- `persist=True`: 이전 프레임의 추적 정보를 기억하라는 뜻입니다. **빠뜨리면 매 프레임 ID가 새로 매겨집니다.**
- `tracker`: 추적 알고리즘 선택. `bytetrack.yaml`(빠름) 또는 `botsort.yaml`(기본값, 가려짐에 강함).
- ID별 중심점을 최근 30프레임 동안 저장해 `cv2.polylines`로 **이동 경로**를 그립니다.

---

## 5. 실행 방법

```bash
source .venv/bin/activate
python 1_classification.py   # 분류
python 2_detection.py        # 객체인식
python 3_segmentation.py     # 분할
python 4_tracking.py         # 객체추적
```

창을 클릭해 선택한 뒤 **q** 키를 누르면 종료됩니다.

---

## 6. 트러블슈팅: 실습 중 실제로 만난 오류

### ❌ 오류 1: `ModuleNotFoundError: No module named 'cv2'`

```
File "1_classification.py", line 3, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
```

- **원인**: OpenCV가 설치되지 않았거나, 패키지를 설치한 환경과 다른 Python으로 실행했습니다.
- **해결**:
  1. 가상환경을 만들고 `pip install ultralytics opencv-python`으로 설치합니다.
  2. 실행할 때 **가상환경의 Python**을 써야 합니다. `source .venv/bin/activate` 후 실행하거나 `.venv/bin/python 파일.py`로 실행합니다.
  3. VS Code라면 오른쪽 아래 인터프리터를 `.venv`로 바꿉니다.

> 📌 패키지 이름(`opencv-python`)과 import 이름(`cv2`)이 다르다는 점도 기억하세요.

### ❌ 오류 2: 카메라 권한 없음 (macOS)

```
OpenCV: not authorized to capture video (status 0), requesting...
RuntimeError: 웹캠을 열 수 없습니다.
```

- **원인**: macOS는 카메라를 쓰는 앱마다 허용을 따로 받습니다. 코드를 실행한 터미널 앱(Terminal, iTerm, VS Code 등)에 권한이 없었습니다.
- **해결**:
  1. **시스템 설정 → 개인정보 보호 및 보안 → 카메라**에서 사용 중인 터미널 앱을 켭니다.
  2. 터미널 앱을 완전히 종료(⌘Q)했다가 다시 열고 실행합니다.

---

## 7. 정리

| | 분류 | 객체인식 | 분할 | 객체추적 |
|---|---|---|---|---|
| 호출 | `model(frame)` | `model(frame)` | `model(frame)` | `model.track(frame, persist=True)` |
| 결과 위치 | `result.probs` | `result.boxes` | `result.masks` | `result.boxes.id` |
| 위치 정보 | ❌ | 박스 4개 값 | 윤곽점 수백 개 | 박스 + ID |
| 활용 예 | 불량품 판별 | CCTV 사람 감지 | 배경 제거, 의료 영상 | 유동 인구 계수, 스포츠 분석 |

---

## 8. 도전 과제

1. **모델 크기 바꾸기**: `yolo26n.pt` → `yolo26s.pt`로 바꾸고 정확도와 속도(FPS) 차이를 비교해 보세요.
2. **특정 클래스만 찾기**: `model(frame, classes=[0])`로 사람(class 0)만 탐지해 보세요.
3. **사람 수 세기**: `2_detection.py`에서 화면 속 사람이 몇 명인지 화면에 글자로 표시해 보세요.
4. **누끼 따기**: `3_segmentation.py`에서 사람 마스크만 남기고 배경을 검은색으로 바꿔 보세요.
5. **트래커 비교**: `4_tracking.py`의 트래커를 `botsort.yaml`로 바꾸고, 손으로 얼굴을 가렸다 뗄 때 ID가 유지되는지 비교해 보세요.

---

## 9. 용어 정리

### AI · 컴퓨터 비전 용어

| 용어 | 뜻 |
|---|---|
| **컴퓨터 비전** (Computer Vision) | 컴퓨터가 이미지나 영상을 보고 내용을 이해하게 하는 AI 분야 |
| **YOLO** (You Only Look Once) | 이미지를 한 번만 보고 물체를 찾아내는 빠른 객체인식 모델 계열. YOLO26은 그 최신 세대 |
| **Ultralytics** | YOLO 모델을 만들고 배포하는 회사이자, 그 모델을 쉽게 쓰게 해주는 Python 라이브러리 이름 |
| **모델** (Model) | 많은 데이터로 미리 학습시켜 둔 AI의 "두뇌". 여기서는 `.pt` 파일 하나가 모델 하나 |
| **사전학습 모델** (Pre-trained Model) | 다른 사람이 이미 대량의 데이터로 학습시켜 둔 모델. 바로 가져다 쓸 수 있음 |
| **추론** (Inference) | 학습된 모델에 새 이미지를 넣어 결과를 얻는 과정. "학습"과 반대되는 말 |
| **클래스** (Class) | 모델이 구분할 수 있는 대상의 종류. 예: person, chair, cup |
| **신뢰도** (Confidence, conf) | 모델이 결과를 얼마나 확신하는지 0~1 사이로 나타낸 값. 0.89 = 89% 확신 |
| **임계값** (Threshold) | 결과를 남길지 버릴지 가르는 기준값. `conf=0.5`면 50% 미만은 버림 |
| **바운딩 박스** (Bounding Box) | 물체를 감싸는 사각형. 왼쪽 위·오른쪽 아래 좌표 4개로 표현 |
| **마스크** (Mask) | 이미지에서 물체에 해당하는 픽셀만 표시한 영역. 분할의 결과물 |
| **폴리곤** (Polygon) | 여러 점을 이어 만든 다각형. 마스크의 외곽선을 점 좌표로 나타낸 것 |
| **누끼** | 사진에서 배경을 지우고 대상만 오려내는 작업을 부르는 디자인 업계 용어 |
| **트래커** (Tracker) | 프레임 사이에서 같은 물체를 찾아 같은 ID를 이어 붙이는 알고리즘. ByteTrack, BoT-SORT 등 |
| **ID** | 추적 중인 물체마다 붙는 고유 번호. 물체가 움직여도 유지됨 |
| **오탐** (False Positive) | 없는 물체를 있다고 잘못 찾는 것 |
| **ImageNet** | 1000개 클래스, 100만 장 이상의 이미지로 된 데이터셋. 분류 모델 학습에 사용 |
| **COCO** | 80개 클래스의 일상 사물 이미지 데이터셋. 객체인식·분할 모델 학습에 사용 |
| **Teachable Machine** | 구글이 만든 웹 도구. 코딩 없이 내 데이터로 분류 모델을 만들 수 있음 |

### 영상 처리 용어

| 용어 | 뜻 |
|---|---|
| **OpenCV** (cv2) | 이미지·영상을 다루는 대표 라이브러리. 웹캠 읽기, 창 띄우기, 도형 그리기 담당 |
| **프레임** (Frame) | 영상을 이루는 한 장 한 장의 사진. 영상은 프레임이 빠르게 이어진 것 |
| **FPS** (Frames Per Second) | 1초에 처리하는 프레임 수. 높을수록 부드러움 |
| **해상도** | 이미지의 가로×세로 픽셀 수. 이 실습 웹캠은 1920×1080 |
| **좌표** | 이미지에서 위치를 나타내는 `(x, y)` 값. **왼쪽 위가 (0, 0)** 이고, 오른쪽으로 x, 아래로 y가 커짐 |
| **픽셀** (Pixel) | 이미지를 이루는 가장 작은 점 하나 |

### Python · 개발 환경 용어

| 용어 | 뜻 |
|---|---|
| **라이브러리 / 패키지** | 다른 사람이 만들어 둔 코드 묶음. `pip install`로 설치해서 `import`로 불러씀 |
| **pip** | Python 패키지 설치 도구 |
| **가상환경** (venv) | 프로젝트마다 패키지를 따로 설치해 두는 독립된 공간. 프로젝트끼리 버전 충돌을 막아줌 |
| **인터프리터** (Interpreter) | Python 코드를 실행하는 프로그램. 가상환경마다 따로 있음 |
| **PyTorch** | 딥러닝 모델을 실행하는 라이브러리. ultralytics가 내부적으로 사용 |
| **`.pt` 파일** | PyTorch 모델을 저장한 파일 |
| **MPS** (Metal Performance Shaders) | Apple 실리콘 Mac의 GPU로 연산을 빠르게 해주는 기능 |
| **ModuleNotFoundError** | `import`하려는 패키지를 찾지 못했을 때 나는 오류. 대부분 설치가 안 됐거나 다른 Python으로 실행한 경우 |
| **Traceback** | 오류가 났을 때 어느 파일 몇 번째 줄에서 문제가 생겼는지 보여주는 기록 |

### Git · GitHub 용어

| 용어 | 뜻 |
|---|---|
| **Git** | 파일의 변경 이력을 저장하고 관리하는 버전 관리 도구 |
| **GitHub** | Git 저장소를 인터넷에 올려 공유하는 서비스 |
| **저장소** (Repository, repo) | 프로젝트 파일과 변경 이력이 담긴 공간 |
| **커밋** (Commit) | 현재 파일 상태를 하나의 기록으로 저장하는 것 |
| **푸시** (Push) | 내 컴퓨터의 커밋을 GitHub에 올리는 것 |
| **GitHub CLI** (gh) | 터미널에서 명령어로 GitHub를 다루는 도구. 저장소 생성, 업로드 등 가능 |
| **`.gitignore`** | Git에 올리지 않을 파일 목록. 이 프로젝트에서는 `.venv`와 `.pt` 파일 제외 |
| **`requirements.txt`** | 프로젝트에 필요한 패키지 목록. `pip install -r requirements.txt`로 한 번에 설치 |

---

## 파일 구성

```
.
├── 1_classification.py   # 분류
├── 2_detection.py        # 객체인식
├── 3_segmentation.py     # 분할
├── 4_tracking.py         # 객체추적
├── requirements.txt      # 필요 패키지 목록
├── .gitignore            # Git 제외 목록
└── README.md             # 강의노트 (이 문서)
```

## 참고 자료

- Ultralytics 공식 문서: https://docs.ultralytics.com
- YOLO26 모델 소개: https://docs.ultralytics.com/models/yolo26/
- OpenCV Python 튜토리얼: https://docs.opencv.org/
