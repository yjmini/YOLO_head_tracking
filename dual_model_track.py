import cv2
import os
from ultralytics import YOLO

# 1. 두 개의 모델 모두 5060 Ti VRAM에 올리기
head_model = YOLO("./runs/detect/train2/weights/best.pt") # 내가 직접 구운 머리 추적 모델
pose_model = YOLO("yolov8n-pose.pt")                   # 방향 벡터를 뽑아줄 포즈 모델

video_path = "./videos/test3.mp4"  # 테스트 영상 경로
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps: fps = 30.0

out = None

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    annotated_frame = frame.copy()

    # 2. 내 모델로 머리 추적 (Tracking - ID 부여용)
    head_results = head_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.6, verbose=False)
    
    # 3. 포즈 모델로 관절 찾기 (Detection - 방향 벡터용, 트래킹 불필요)
    pose_results = pose_model(frame, conf=0.5, verbose=False)

    # 포즈 모델이 찾은 키포인트들 미리 뽑아두기
    all_keypoints = []
    if pose_results[0].keypoints is not None:
        all_keypoints = pose_results[0].keypoints.data.cpu().numpy()

    # 비디오 저장기 초기화 (최초 1회)
    if out is None:
        height, width, _ = annotated_frame.shape
        out = cv2.VideoWriter(os.path.join(os.getcwd(), 'dual_tracking_result.mp4'), 
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 4. 화면에 머리가 인식되었다면 매칭 시작
    if head_results[0].boxes.id is not None:
        # 이번에는 포함 여부를 쉽게 검사하기 위해 xyxy (좌상단, 우하단 좌표) 포맷 사용
        boxes = head_results[0].boxes.xyxy.cpu().numpy() 
        track_ids = head_results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            hx1, hy1, hx2, hy2 = box # 머리 박스의 좌표
            
            # 머리 네모 박스 그리기
            cv2.rectangle(annotated_frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 144, 30), 2)
            cv2.putText(annotated_frame, f"ID: {track_id}", (int(hx1), int(hy1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 144, 30), 2)

            # 5. 이 머리 박스 안에 들어가는 '코'가 있는지 교차 검증
            for kp in all_keypoints:
                nose = kp[0]
                nx, ny, n_conf = nose

                # 💡 수정 1: 눈의 좌표는 아예 안 쓰고, 코(nose)가 박스 안에 있고 확실히 보일 때만 실행
                if (hx1 <= nx <= hx2) and (hy1 <= ny <= hy2) and (n_conf > 0.5):
                    
                    # 💡 수정 2: '두 눈' 대신 '머리 박스의 정중앙' 좌표를 구합니다.
                    head_center_x = (hx1 + hx2) / 2
                    head_center_y = (hy1 + hy2) / 2
                    
                    # 💡 수정 3: 머리 중심에서 코가 치우친 방향 벡터 계산
                    dx = nx - head_center_x
                    dy = ny - head_center_y
                    
                    # 화살표 그리기 (방향 벡터에 곱하는 숫자로 화살표 길이를 조절하세요)
                    start_point = (int(nx), int(ny))
                    end_point = (int(nx + dx * 2.5), int(ny + dy * 2.5)) 
                    
                    cv2.arrowedLine(annotated_frame, start_point, end_point, (0, 0, 255), 3, tipLength=0.3)
                    
                    # 짝을 찾았으니 현재 머리에 대한 검색 종료
                    break

    out.write(annotated_frame)
    cv2.imshow("Dual Model Tracker (Head + Direction)", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
if out is not None: out.release()
cv2.destroyAllWindows()