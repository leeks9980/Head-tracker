import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
prev_time = 0

# 좌우 기준을 잡기 위한 랜드마크 인덱스
# 왼쪽 귀/광대 바깥쪽: 234, 오른쪽 귀/광대 바깥쪽: 454, 코끝: 1
LEFT_EDGE_IDX = 234
RIGHT_EDGE_IDX = 454
NOSE_IDX = 1

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # 좌우 반전 (거울 모드)
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    yaw_direction = "Center"
    yaw_ratio = 0.0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # 1. 주요 랜드마크 픽셀 좌표 추출
            left_x = landmarks[LEFT_EDGE_IDX].x * w
            right_x = landmarks[RIGHT_EDGE_IDX].x * w
            nose_x = landmarks[NOSE_IDX].x * w

            # 2. 얼굴 중앙 X 좌표 계산 (양쪽 끝점의 중간)
            center_x = (left_x + right_x) / 2.0
            
            # 3. 얼굴 전체 폭 (스케일 정규화를 위함)
            face_width = right_x - left_x

            if face_width > 0:
                # 4. 코끝이 중앙에서 가로 방향으로 치우친 정도를 얼굴 폭으로 나눔 (-0.5 ~ 0.5 사이 값)
                yaw_ratio = (nose_x - center_x) / face_width

                # 5. 임계값(Threshold)을 기준으로 방향 판정 (필요에 따라 민감도 조절 가능)
                threshold = 0.08  # 값이 클수록 덜 민감함
                if yaw_ratio < -threshold:
                    yaw_direction = "Left"
                elif yaw_ratio > threshold:
                    yaw_direction = "Right"
                else:
                    yaw_direction = "Center"

            # 시각화: 주요 포인트 및 중심선 표시
            cv2.circle(image, (int(center_x), int((landmarks[LEFT_EDGE_IDX].y + landmarks[RIGHT_EDGE_IDX].y) * h / 2)), 5, (0, 255, 255), -1)
            cv2.circle(image, (int(nose_x), int(landmarks[NOSE_IDX].y * h)), 5, (0, 0, 255), -1)
            cv2.line(image, (int(center_x), 0), (int(center_x), h), (255, 0, 0), 1)

    # 화면 정보 출력
    cv2.putText(image, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Direction: {yaw_direction} (Ratio: {yaw_ratio:.3f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow('Lightweight Yaw Tracking', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()