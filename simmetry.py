import cv2
import mediapipe as mp
import math
import time

# 初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

symmetric_pairs = [
    (33, 263), (133, 362), (61, 291), (234, 454), (127, 356),
]
top_idx = 10
bottom_idx = 152
center_idx = 1

def get_rotation_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return -math.atan2(dy, dx)

def rotate_point(x, y, cx, cy, angle):
    dx, dy = x - cx, y - cy
    x_rot = dx * math.cos(angle) - dy * math.sin(angle)
    y_rot = dx * math.sin(angle) + dy * math.cos(angle)
    return x_rot + cx, y_rot + cy

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# 測定時間（秒）
MEASURE_DURATION = 8
max_score = 0.0
start_time = time.time()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > MEASURE_DURATION:
        break  # 終了時間を超えたら測定停止

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = (landmarks[33].x * w, landmarks[33].y * h)
        right_eye = (landmarks[263].x * w, landmarks[263].y * h)
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        angle = get_rotation_angle(left_eye, right_eye)

        rotated = {}
        for i, lm in enumerate(landmarks):
            x, y = lm.x * w, lm.y * h
            rotated[i] = rotate_point(x, y, center[0], center[1], angle)

        top = rotated[top_idx]
        bottom = rotated[bottom_idx]
        face_height = distance(top, bottom)
        center_x = rotated[center_idx][0]

        total_diff = 0
        for l_idx, r_idx in symmetric_pairs:
            lx = rotated[l_idx][0]
            rx = rotated[r_idx][0]
            norm_l = abs(lx - center_x) / face_height
            norm_r = abs(rx - center_x) / face_height
            diff = abs(norm_l - norm_r)

            z_l = landmarks[l_idx].z
            z_r = landmarks[r_idx].z
            z_diff = abs(z_l - z_r)
            z_penalty = z_diff * 2.0

            total_diff += diff + z_penalty

        raw_score = max(0.0, 1.0 - total_diff / len(symmetric_pairs))
        symmetry_score = raw_score * 100

        if symmetry_score > max_score:
            max_score = symmetry_score  # 最高スコア更新

        # スコア表示
        cv2.putText(
            frame,
            f"Live Score: {symmetry_score:.1f}/100",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Max Score: {max_score:.1f}/100",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Time: {MEASURE_DURATION - int(elapsed_time)}s",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION
        )
    else:
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Symmetry Measuring (8s)", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# 測定終了後、最高スコアを表示
print(f"\n✅ 測定終了：最高スコアは {max_score:.1f} / 100")



