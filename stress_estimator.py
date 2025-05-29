import cv2
import mediapipe as mp
import math
import json
import os
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

REFERENCE_FILE = "reference_face.json"  # 保存された標準顔ランドマーク

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def load_reference():
    if not os.path.exists(REFERENCE_FILE):
        print("❌ reference_face.json が見つかりません。先に保存してください。")
        return None
    with open(REFERENCE_FILE, "r") as f:
        return json.load(f)

def compute_stress_score(landmarks, reference, w, h):
    top = landmarks[10]
    bottom = landmarks[152]
    left = landmarks[234]
    right = landmarks[454]
    face_height = distance(top.x * w, top.y * h, bottom.x * w, bottom.y * h)
    face_width = distance(left.x * w, left.y * h, right.x * w, right.y * h)

    eye_now = distance(
        landmarks[145].x * w, landmarks[145].y * h,
        landmarks[159].x * w, landmarks[159].y * h
    ) / face_height

    brow_now = distance(
        landmarks[55].x * w, landmarks[55].y * h,
        landmarks[285].x * w, landmarks[285].y * h
    ) / face_width

    ref_eye = reference["eye_open"]
    ref_brow = reference["brow_dist"]

    eye_score = max(0, min(1, (ref_eye - eye_now) / ref_eye * 5))
    brow_score = max(0, min(1, (ref_brow - brow_now) / ref_brow * 5))

    stress_index = round((eye_score + brow_score) / 2 * 100, 1)
    return stress_index

# カメラ起動
cap = cv2.VideoCapture(0)
reference = load_reference()

if reference is None:
    cap.release()
    exit()

scores = []
start_time = time.time()
duration = 10  # 測定秒数

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    elapsed = time.time() - start_time
    label = ""
    color = (200, 200, 200)

    if results.multi_face_landmarks and elapsed <= duration:
        landmarks = results.multi_face_landmarks[0].landmark
        stress_index = compute_stress_score(landmarks, reference, w, h)
        scores.append(stress_index)

        label = f"Measuring... {int(elapsed)}/{duration}s"
        color = (255, 255, 0)
    elif elapsed > duration:
        if scores:
            avg_score = round(sum(scores) / len(scores), 1)
            label = f"Avg Stress: {avg_score}/100"
            color = (0, 0, 255) if avg_score >= 67 else (0, 255, 0)
        else:
            label = "No face detected"
            color = (0, 0, 255)
    else:
        label = f"Waiting for face... {int(elapsed)}/{duration}s"
        color = (100, 100, 100)

    cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if results.multi_face_landmarks:
        mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION)

    cv2.imshow("Stress Estimation (10s Avg)", frame)

    # 測定が終わったら、結果を表示し続け、Escで終了
    if elapsed > duration:
        if cv2.waitKey(5) & 0xFF == 27:
            break
        continue

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


















