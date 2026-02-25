import cv2
import os
from ultralytics import YOLO

# 1. 뼈대와 얼굴 주요 포인트를 찾는 Pose 모델 불러오기
model = YOLO("yolov8n-pose.pt")

video_path = "./videos/test3.mp4"  # 테스트 영상 경로
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30.0

out = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. 포즈 모델로 추적 실행
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, verbose=False)
    
    # 뼈대와 박스가 기본적으로 그려진 프레임 가져오기
    annotated_frame = results[0].plot()

    # 안전한 영상 저장을 위한 셋업 (첫 프레임에서 딱 한 번 실행)
    if out is None:
        height, width, _ = annotated_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        save_path = os.path.join(os.getcwd(), 'face_direction_result.mp4')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # 3. 키포인트(관절 및 얼굴 포인트) 정보가 인식되었다면
    if results[0].keypoints is not None:
        # GPU에 있는 데이터를 CPU로 내리고 Numpy 배열로 변환
        keypoints = results[0].keypoints.data.cpu().numpy() 

        for kp in keypoints:
            # YOLO Pose 모델의 인덱스 규칙: 0(코), 1(왼쪽 눈), 2(오른쪽 눈)
            nose = kp[0]
            l_eye = kp[1]
            r_eye = kp[2]

            # [x좌표, y좌표, 신뢰도(정확도)] 중 신뢰도가 0.5 이상으로 확실히 보일 때만 계산
            if nose[2] > 0.5 and l_eye[2] > 0.5 and r_eye[2] > 0.5:
                
                # 4. 두 눈의 중심점 좌표 계산
                eye_center_x = (l_eye[0] + r_eye[0]) / 2
                eye_center_y = (l_eye[1] + r_eye[1]) / 2

                # 5. 얼굴 방향 벡터 계산 (코 좌표 - 두 눈의 중심점 좌표)
                dx = nose[0] - eye_center_x
                dy = nose[1] - eye_center_y

                # 화살표 시작점 (코 위치)
                start_point = (int(nose[0]), int(nose[1]))
                
                # 화살표 끝점 (방향 벡터를 3~4배 길게 뻗어 예측 방향 표시)
                end_point = (int(nose[0] + dx * 4), int(nose[1] + dy * 4))

                # 코에서 시선 방향으로 뻗어나가는 붉은색 화살표 그리기
                cv2.arrowedLine(annotated_frame, start_point, end_point, (0, 0, 255), 3, tipLength=0.3)

    out.write(annotated_frame)

    cv2.imshow("Face Direction Tracker", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()