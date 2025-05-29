import cv2
import mediapipe as mp
import math
import json
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

REFERENCE_FILE = "reference_face.json"

def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def save_reference(landmarks):
    top = landmarks[10]
    bottom = landmarks[152]
    left = landmarks[234]
    right = landmarks[454]
    face_height = distance(top, bottom)
    face_width = distance(left, right)

    if face_height == 0 or face_width == 0:
        print("⚠ 顔の高さまたは幅が0。保存できません。")
        return

    # 特定部位の正規化値を保存
    eye_open = distance(landmarks[145], landmarks[159]) / face_height
    brow_dist = distance(landmarks[55], landmarks[285]) / face_width

    reference_data = {
        "eye_open": eye_open,
        "brow_dist": brow_dist
    }

    try:
        with open(REFERENCE_FILE, "w") as f:
            json.dump(reference_data, f, indent=2)
        print("✅ reference_face.json を保存しました。")
    except Exception as e:
        print(f"❌ ファイル保存に失敗しました: {e}")

cap = cv2.VideoCapture(0)
print("📸 顔が検出されたら自動保存 / Escで終了")

saved = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks and not saved:
        print("📍 顔が検出されました。保存します...")
        save_reference(results.multi_face_landmarks[0].landmark)
        saved = True

    if results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION)

    cv2.putText(frame, "Saving face automatically...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Auto Save Reference Face", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        print("🛑 終了します。")
        break

cap.release()
cv2.destroyAllWindows()






