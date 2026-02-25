import cv2
import os
from collections import defaultdict
from ultralytics import YOLO

# 1. 학습된 내 모델 불러오기 (경로 확인 필수!)
model = YOLO("./runs/detect/train2/weights/best.pt")
video_path = "./videos/test3.mp4"  # 테스트 영상 경로
cap = cv2.VideoCapture(video_path)

# FPS 가져오기 (만약 영상을 못 읽어와서 0이 나오면 기본값 30으로 강제 세팅)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30.0

# 💡 수정 포인트: 루프 밖에서는 VideoWriter를 만들지 않고 비워둠!
out = None

# 2. 객체(머리)들의 과거 위치(궤적)를 저장할 딕셔너리
track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. 모델로 추적 실행 (persist=True로 해야 프레임 간 ID가 유지됨)
    # 아까 말한 오탐 방지를 위해 conf=0.6 적용!
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml", conf=0.7, verbose=False
    )
    annotated_frame = results[0].plot()

    # 💡 수정 포인트: 첫 번째 프레임을 그렸을 때, 그 '진짜 크기'를 기준으로 저장기를 딱 한 번만 생성함
    if out is None:
        height, width, _ = annotated_frame.shape
        # mp4v 코덱 사용 (만약 이것도 안 되면 '*XVID' 로 바꾸고 파일명을 .avi로 바꿔봐)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        save_path = os.path.join(os.getcwd(), "tracking_result.mp4")

        print("\n" + "=" * 50)
        print(f"🎬 비디오 저장을 시작합니다!")
        print(f"📍 저장 위치: {save_path}")
        print(f"📐 해상도: {width}x{height} @ {fps}FPS")
        print("=" * 50 + "\n")

        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # --- (이하 방향 예측 및 화살표 그리는 로직 동일) ---
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            current_center = (int(x), int(y))
            track_history[track_id].append(current_center)

            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)

            if len(track_history[track_id]) >= 10:
                past_center = track_history[track_id][0]
                dx = current_center[0] - past_center[0]
                dy = current_center[1] - past_center[1]
                distance = (dx**2 + dy**2) ** 0.5

                if distance > 10:
                    end_point = (
                        int(current_center[0] + dx * 2),
                        int(current_center[1] + dy * 2),
                    )
                    cv2.arrowedLine(
                        annotated_frame,
                        current_center,
                        end_point,
                        (0, 0, 255),
                        3,
                        tipLength=0.3,
                    )

    # 이제 100% 해상도가 맞물린 프레임이 안전하게 저장됨
    out.write(annotated_frame)

    cv2.imshow("Head Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
