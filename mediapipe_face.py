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

# 3D 헤드 포즈 추정을 위한 3D 모델 표준 좌표계 (Generic 3D face points)
# 코끝, 턱, 왼쪽 눈 코너, 오른쪽 눈 코너, 왼쪽 입꼬리, 오른쪽 입꼬리
object_3d_points = np.float32([
    [0.0, 0.0, 0.0],          # 코끝 (Nose tip)
    [0.0, -330.0, -65.0],     # 턱 (Chin)
    [-225.0, 170.0, -135.0],  # 왼쪽 눈 왼쪽 끝 (Left eye left corner)
    [225.0, 170.0, -135.0],   # 오른쪽 눈 오른쪽 끝 (Right eye right corner)
    [-150.0, -150.0, -125.0], # 왼쪽 입꼬리 (Left mouth corner)
    [150.0, -150.0, -125.0]   # 오른쪽 입꼬리 (Right mouth corner)
])

def draw_virtual_space(yaw, pitch, roll):
    """
    [시각화 2] 가상 공간(검은 배경)에 가상의 얼굴/축(Axes)을 만들어 
    실제 머리 회전 각도에 따라 움직이는 시각화 창 생성
    """
    canvas_size = 400
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    # 화면 중심점
    center = (canvas_size // 2, canvas_size // 2)
    
    # 라디안 변환
    y_rad = np.radians(yaw)
    p_rad = np.radians(pitch)
    r_rad = np.radians(roll)
    
    # 3D 회전 행렬 생성 (Yaw, Pitch, Roll 조합)
    # 간단한 3D 프로젝션을 위한 회전 매트릭스 계산
    cos_y, sin_y = np.cos(y_rad), np.sin(y_rad)
    cos_p, sin_p = np.cos(p_rad), np.sin(p_rad)
    cos_r, sin_r = np.cos(r_rad), np.sin(r_rad)
    
    # 가상의 머리 방향을 나타내는 3D 축 (Axis) 정의
    axis_length = 80
    axes = np.float32([
        [axis_length, 0, 0],  # X축 (빨강 - 오른쪽)
        [0, axis_length, 0],  # Y축 (초록 - 위쪽)
        [0, 0, axis_length]   # Z축 (파랑 - 앞쪽)
    ])
    
    # 회전 적용
    rotated_axes = []
    for axis in axes:
        x, y, z = axis
        # Pitch 회전 (X축 기준)
        y1 = y * cos_p - z * sin_p
        z1 = y * sin_p + z * cos_p
        # Yaw 회전 (Y축 기준)
        x2 = x * cos_y + z1 * sin_y
        z2 = -x * sin_y + z1 * cos_y
        # Roll 회전 (Z축 기준)
        x3 = x2 * cos_r - y1 * sin_r
        y3 = x2 * sin_r + y1 * cos_r
        
        # 2D 평면으로 투영 (원근감 무시하고 중심 기준 이동)
        proj_x = int(center[0] + x3)
        proj_y = int(center[1] - y3)  # 이미지 좌표계 Y는 아래가 양수이므로 반전
        rotated_axes.append((proj_x, proj_y))
    
    # 가상 공간에 기준 십자가 그리기
    cv2.circle(canvas, center, 5, (255, 255, 255), -1)
    
    # 축 그리기 (X: 빨강, Y: 초록, Z: 파랑)
    cv2.line(canvas, center, rotated_axes[0], (0, 0, 255), 3)   # Yaw 방향
    cv2.line(canvas, center, rotated_axes[1], (0, 255, 0), 3)   # Pitch 방향
    cv2.line(canvas, center, rotated_axes[2], (255, 0, 0), 3)   # Roll 방향
    
    # 각도 텍스트 표시
    cv2.putText(canvas, f"Yaw (LR): {int(yaw)} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(canvas, f"Pitch (UD): {int(pitch)} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(canvas, f"Roll (Tilt): {int(roll)} deg", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return canvas

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    yaw, pitch, roll = 0, 0, 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 1. 2D 이미지 상의 핵심 랜드마크 좌표 추출 (인덱스 매핑)
            # 코끝(1), 턱(152), 왼쪽 눈 끝(33), 오른쪽 눈 끝(263), 왼쪽 입꼬리(61), 오른쪽 입꼬리(291)
            landmark_ids = [1, 152, 33, 263, 61, 291]
            image_2d_points = []
            
            for idx in landmark_ids:
                pt = face_landmarks.landmark[idx]
                image_2d_points.append([pt.x * w, pt.y * h])
            
            image_2d_points = np.float32(image_2d_points)

            # 2. 카메라 내부 매개변수 가정 (Camera Matrix)
            focal_length = w
            center_x, center_y = w / 2, h / 2
            camera_matrix = np.array([
                [focal_length, 0, center_x],
                [0, focal_length, center_y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            distortion_coeffs = np.zeros((4, 1)) # 왜곡 없음 가정

            # 3. solvePnP를 통한 회전/이동 벡터 계산
            success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
                object_3d_points, 
                image_2d_points, 
                camera_matrix, 
                distortion_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success_pnp:
                # 회전 벡터를 오일러 각도(Euler Angles)로 변환
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                pose_mat = cv2.hconcat([rotation_matrix, translation_vector])
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                
                pitch = euler_angles[0][0]
                yaw = euler_angles[1][0]
                roll = euler_angles[2][0]

                # [시각화 1] 웹캠 화면 위에 머리 방향 축(Axis) 그리기
                axis_length = 50
                axis_3d = np.float32([
                    [axis_length, 0, 0], 
                    [0, axis_length, 0], 
                    [0, 0, -axis_length]
                ]).reshape(-1, 3)
                
                axis_2d, _ = cv2.projectPoints(axis_3d, rotation_vector, translation_vector, camera_matrix, distortion_coeffs)
                
                # 기준 코끝 위치
                nose_2d = (int(image_2d_points[0][0]), int(image_2d_points[0][1]))
                
                # 선 그리기 (X: 빨강, Y: 초록, Z: 파랑)
                cv2.line(image, nose_2d, tuple(axis_2d[0].ravel().astype(int)), (0, 0, 255), 3) # Yaw
                cv2.line(image, nose_2d, tuple(axis_2d[1].ravel().astype(int)), (0, 255, 0), 3) # Pitch
                cv2.line(image, nose_2d, tuple(axis_2d[2].ravel().astype(int)), (255, 0, 0), 3) # Roll

    # [시각화 1 창] 웹캠 화면 출력
    cv2.putText(image, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Yaw: {int(yaw)} | Pitch: {int(pitch)} | Roll: {int(roll)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow('1. Webcam Tracking View', image)

    # [시각화 2 창] 가상 공간 View 생성 및 출력
    virtual_view = draw_virtual_space(yaw, pitch, roll)
    cv2.imshow('2. Virtual Space Head View', virtual_view)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()