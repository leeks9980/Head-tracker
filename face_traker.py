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

# 랜드마크 인덱스 정의
LEFT_EDGE_IDX = 234
RIGHT_EDGE_IDX = 454
NOSE_IDX = 1
TOP_IDX = 10
CHIN_IDX = 152

def draw_virtual_cube(img, yaw_ratio, pitch_ratio):
    h, w, _ = img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    cx, cy = w // 2, h // 2

    # 화면 중앙 정면 기준선 및 가이드 표시
    for y_pos in range(0, h, 20):
        cv2.line(canvas, (cx, y_pos), (cx, y_pos + 10), (50, 50, 50), 1)
    for x_pos in range(0, w, 20):
        cv2.line(canvas, (x_pos, cy), (x_pos + 10, cy), (50, 50, 50), 1)
    
    cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

    # 3D 큐브의 기본 정점 정의
    size = 70
    points_3d = np.array([
        [-size, -size, -size], # 0: 뒤-왼-위
        [ size, -size, -size], # 1: 뒤-오-위
        [ size,  size, -size], # 2: 뒤-오-아래
        [-size,  size, -size], # 3: 뒤-왼-아래
        [-size, -size,  size], # 4: 앞-왼-위 (Front Face)
        [ size, -size,  size], # 5: 앞-오-위 (Front Face)
        [ size,  size,  size], # 6: 앞-오-아래 (Front Face)
        [-size,  size,  size]  # 7: 앞-왼-아래 (Front Face)
    ], dtype=np.float32)
    
    # 회전 각도 변환 (Pitch 방향 반전 반영)
    theta_y = np.radians(yaw_ratio * 150)    # Yaw (Y축 회전)
    theta_x = np.radians(pitch_ratio * 150)  # Pitch (X축 회전 - 반전됨)
    
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    
    # Y축 회전 행렬 (Yaw)
    rotation_y = np.array([
        [ cos_y, 0, sin_y],
        [     0, 1,     0],
        [-sin_y, 0, cos_y]
    ], dtype=np.float32)
    
    # X축 회전 행렬 (Pitch)
    rotation_x = np.array([
        [1,     0,      0],
        [0, cos_x, -sin_x],
        [0, sin_x,  cos_x]
    ], dtype=np.float32)
    
    # 두 회전 행렬 합성
    combined_rotation = np.dot(rotation_y, rotation_x)
    
    projected_points = []
    for pt in points_3d:
        rotated_pt = np.dot(combined_rotation, pt)
        z_depth = rotated_pt[2] + 400 
        focal_length = 400
        
        x2d = int(cx + (rotated_pt[0] * focal_length / z_depth))
        y2d = int(cy + (rotated_pt[1] * focal_length / z_depth))
        projected_points.append((x2d, y2d))
        
    # 앞면(Front Face) 채우기
    front_face_idx = [4, 5, 6, 7]
    front_points_2d = [projected_points[i] for i in front_face_idx]
    cv2.fillPoly(canvas, [np.array(front_points_2d, np.int32)], (255, 200, 100))
    
    # 정면 화살표 표시
    f_p4, f_p6 = front_points_2d[0], front_points_2d[2]
    front_center_x = int((f_p4[0] + f_p6[0]) / 2)
    front_center_y = int((f_p4[1] + f_p6[1]) / 2)
    
    cv2.arrowedLine(canvas, (front_center_x, front_center_y), (front_center_x, front_center_y - 40), (0, 0, 255), 3, tipLength=0.4)
    cv2.putText(canvas, "FRONT", (front_center_x - 25, front_center_y - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 큐브 모서리 및 선 연결
    edges = [
        (0,1), (1,2), (2,3), (3,0), 
        (4,5), (5,6), (6,7), (7,4), 
        (0,4), (1,5), (2,6), (3,7)  
    ]
    
    for edge in edges:
        pt1 = projected_points[edge[0]]
        pt2 = projected_points[edge[1]]
        color = (150, 100, 50) if edge in [(4,5), (5,6), (6,7), (7,4)] else (100, 100, 100)
        cv2.line(canvas, pt1, pt2, color, 2)
        
    for pt in projected_points:
        cv2.circle(canvas, pt, 4, (0, 0, 200), -1)
        
    cv2.putText(canvas, f"Yaw Ratio: {yaw_ratio:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(canvas, f"Pitch Ratio: {pitch_ratio:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
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

    yaw_ratio = 0.0
    pitch_ratio = 0.0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # 1. Yaw (좌우) 계산
            left_x = landmarks[LEFT_EDGE_IDX].x * w
            right_x = landmarks[RIGHT_EDGE_IDX].x * w
            nose_x = landmarks[NOSE_IDX].x * w
            center_x = (left_x + right_x) / 2.0
            face_width = right_x - left_x

            if face_width > 0:
                yaw_ratio = (nose_x - center_x) / face_width

            # 2. Pitch (상하) 계산 -> [수정] 위아래 반전을 위해 마이너스(-) 부호 추가
            top_y = landmarks[TOP_IDX].y * h
            chin_y = landmarks[CHIN_IDX].y * h
            nose_y = landmarks[NOSE_IDX].y * h
            vertical_center = (top_y + chin_y) / 2.0
            face_height = chin_y - top_y

            if face_height > 0:
                pitch_ratio = -((nose_y - vertical_center) / face_height)

            # 시각화 포인트
            cv2.circle(image, (int(center_x), int((landmarks[LEFT_EDGE_IDX].y + landmarks[RIGHT_EDGE_IDX].y) * h / 2)), 5, (0, 255, 255), -1)
            cv2.circle(image, (int(nose_x), int(nose_y)), 5, (0, 0, 255), -1)
            cv2.line(image, (int(center_x), 0), (int(center_x), h), (255, 0, 0), 1)
            cv2.line(image, (0, int(vertical_center)), (w, int(vertical_center)), (0, 255, 0), 1)

    cv2.putText(image, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Yaw: {yaw_ratio:.3f} | Pitch: {pitch_ratio:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow('Webcam Face Mesh', image)
    
    virtual_window = draw_virtual_cube(image, yaw_ratio, pitch_ratio)
    cv2.imshow('Virtual Object 3D Rotation', virtual_window)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()