import cv2
import mediapipe as mp
import numpy as np
import time
import socket
import struct

# --- OpenTrack FreePIE UDP 통신 설정 ---
OPENTRACK_IP = "127.0.0.1"
OPENTRACK_PORT = 4242  
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

LEFT_EDGE_IDX = 234
RIGHT_EDGE_IDX = 454
NOSE_IDX = 1
TOP_IDX = 10
CHIN_IDX = 152

def draw_virtual_cube(img, yaw_deg, pitch_deg, send_yaw, send_pitch):
    h, w, _ = img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    for y_pos in range(0, h, 20):
        cv2.line(canvas, (cx, y_pos), (cx, y_pos + 10), (50, 50, 50), 1)
    for x_pos in range(0, w, 20):
        cv2.line(canvas, (x_pos, cy), (x_pos + 10, cy), (50, 50, 50), 1)
    
    cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

    size = 70
    points_3d = np.array([
        [-size, -size, -size], [ size, -size, -size], 
        [ size,  size, -size], [-size,  size, -size], 
        [-size, -size,  size], [ size, -size,  size], 
        [ size,  size,  size], [-size,  size,  size]  
    ], dtype=np.float32)
    
    # 각도 변화폭을 완만하게 제한하여 과도한 회전 방지
    theta_y = np.radians(yaw_deg)    
    theta_x = np.radians(pitch_deg)  
    
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    
    rotation_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
    rotation_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float32)
    combined_rotation = np.dot(rotation_y, rotation_x)
    
    projected_points = []
    for pt in points_3d:
        rotated_pt = np.dot(combined_rotation, pt)
        z_depth = rotated_pt[2] + 400 
        x2d = int(cx + (rotated_pt[0] * 400 / z_depth))
        y2d = int(cy + (rotated_pt[1] * 400 / z_depth))
        projected_points.append((x2d, y2d))
        
    front_face_idx = [4, 5, 6, 7]
    front_points_2d = [projected_points[i] for i in front_face_idx]
    cv2.fillPoly(canvas, [np.array(front_points_2d, np.int32)], (255, 200, 100))
    
    for edge in [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]:
        cv2.line(canvas, projected_points[edge[0]], projected_points[edge[1]], (100, 100, 100), 2)
        
    cv2.putText(canvas, f"Yaw: {yaw_deg:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, f"Pitch: {pitch_deg:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(canvas, f"[Sent] Yaw: {send_yaw:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(canvas, f"[Sent] Pitch: {send_pitch:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(canvas, f"[Sent] Roll: 0.0 (Fixed)", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
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

    yaw_deg = 0.0
    pitch_deg = 0.0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_x = landmarks[LEFT_EDGE_IDX].x * w
            right_x = landmarks[RIGHT_EDGE_IDX].x * w
            nose_x = landmarks[NOSE_IDX].x * w
            center_x = (left_x + right_x) / 2.0
            face_width = right_x - left_x

            if face_width > 0:
                # 360도 대신 현실적인 고개 회전각 제한(-90도 ~ 90도) 및 감도 조절 계수 적용
                yaw_ratio = (nose_x - center_x) / face_width
                yaw_deg = float(np.clip(yaw_ratio * 120.0, -90.0, 90.0))

            top_y = landmarks[TOP_IDX].y * h
            chin_y = landmarks[CHIN_IDX].y * h
            nose_y = landmarks[NOSE_IDX].y * h
            vertical_center = (top_y + chin_y) / 2.0
            face_height = chin_y - top_y

            if face_height > 0:
                pitch_ratio = -((nose_y - vertical_center) / face_height)
                pitch_deg = float(np.clip(pitch_ratio * 120.0, -90.0, 90.0))

    x = 0.0
    y = 0.0
    z = 0.0
    send_yaw = float(yaw_deg)
    send_pitch = float(pitch_deg)
    send_roll = 0.0 

    # FreePIE 표준 규격에 맞춘 24바이트 패킷 (6개의 32비트 float, Little-endian)
    packet = struct.pack('<dddddd', x, y, z, send_yaw, send_pitch, send_roll)
    udp_sock.sendto(packet, (OPENTRACK_IP, OPENTRACK_PORT))

    cv2.putText(image, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Webcam Face Mesh', image)
    
    virtual_window = draw_virtual_cube(image, yaw_deg, pitch_deg, send_yaw, send_pitch)
    cv2.imshow('Virtual Object 3D Rotation', virtual_window)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
udp_sock.close()