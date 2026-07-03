# YOLO Head Tracking

YOLO 기반 머리 탐지와 ByteTrack 기반 객체 추적을 결합해 사람의 머리 위치, 이동 방향, 얼굴 방향 벡터를 시각화하는 컴퓨터 비전 프로젝트입니다.

## 프로젝트 개요

커스텀 학습한 head detection 모델과 YOLO pose 모델을 함께 사용합니다. 머리 bounding box에는 tracking ID를 부여하고, pose keypoint의 코/눈 좌표 또는 머리 중심-코 벡터를 활용해 바라보는 방향을 화살표로 표시합니다.

## 주요 기능

- 커스텀 YOLO head detector 학습
- ByteTrack 기반 프레임 간 tracking ID 유지
- 머리 중심 이동 방향 추정
- YOLO pose keypoint 기반 얼굴 방향 벡터 표시
- head detector + pose detector dual model pipeline
- 웹 대시보드용 tactical monitoring 화면 구성

## 기술 스택

- **Computer Vision**: OpenCV
- **Object Detection**: Ultralytics YOLO
- **Tracking**: ByteTrack
- **Pose Estimation**: YOLO Pose
- **Language**: Python, HTML/CSS/JavaScript

## 프로젝트 구조

```text
.
├── train.py                  # 커스텀 head detector 학습
├── predict.py                # 기본 추론/추적 실행
├── predict_track.py          # 이동 방향 궤적 기반 tracking 시각화
├── face_direction_track.py   # pose keypoint 기반 얼굴 방향 표시
├── dual_model_track.py       # head detector + pose detector 결합
├── index.html                # 결과 영상을 보여주는 웹 모니터링 UI
├── imgs/                     # README/테스트용 이미지
└── requirements.txt
```

## 핵심 구현 내용

### 1. Head Tracking Pipeline
학습된 head detector로 머리 영역을 찾고, `model.track(..., tracker="bytetrack.yaml")`를 사용해 프레임 간 동일 객체 ID를 유지합니다. 최근 중심 좌표를 누적해 이동 방향을 계산하고 화살표로 표시합니다.

### 2. Face Direction Estimation
YOLO pose 모델의 keypoint 중 코와 눈 좌표를 사용해 얼굴 방향 벡터를 계산합니다. dual model 버전에서는 head box 내부의 코 keypoint를 매칭해 머리 박스와 포즈 결과를 연결합니다.

### 3. Demo Dashboard
`index.html`은 추적 결과 영상을 tactical command center 스타일로 보여주는 단일 HTML 대시보드입니다. 시스템 로그와 짐벌 각도 값은 프론트엔드에서 실시간처럼 갱신됩니다.

## 실행 방법

```bash
pip install -r requirements.txt
```

```bash
# 커스텀 head detector 학습
python train.py

# 추적 결과 저장
python predict_track.py

# 얼굴 방향 표시
python face_direction_track.py

# head detector와 pose detector 결합
python dual_model_track.py
```

커스텀 모델을 실행하려면 `./runs/detect/train2/weights/best.pt` 경로에 학습된 head detector 가중치가 있어야 합니다.

---
YOLO와 ByteTrack을 활용해 사람의 머리 위치와 방향성을 추적하는 실험 프로젝트입니다.
