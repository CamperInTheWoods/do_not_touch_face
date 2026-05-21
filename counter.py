"""
얼굴 손대기 카운터
손이 얼굴 윤곽선 영역에 닿을 때마다 카운트합니다.
종료: Q 키
"""

import urllib.request
from pathlib import Path
from datetime import datetime
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

CAMERA_INDEX = 0
MODEL_DIR  = Path(__file__).parent / "models"
LOG_FILE   = Path(__file__).parent / "logs" / "history.txt"

FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# 얼굴 윤곽선 랜드마크 인덱스 (MediaPipe Face Mesh 468점 기준)
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
             397,365,379,378,400,377,152,148,176,149,150,136,
             172,58,132,93,234,127,162,21,54,103,67,109]


def download_models():
    MODEL_DIR.mkdir(exist_ok=True)
    for url, path in [(FACE_MODEL_URL, FACE_MODEL_PATH), (HAND_MODEL_URL, HAND_MODEL_PATH)]:
        if not path.exists():
            print(f"모델 다운로드 중: {path.name} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  완료: {path.name}")


def get_face_polygon(face_landmarks, w, h):
    points = []
    for i in FACE_OVAL:
        lm = face_landmarks[i]
        points.append([int(lm.x * w), int(lm.y * h)])
    return np.array(points, dtype=np.int32)


def hand_in_face(hand_landmarks, face_polygon, w, h):
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        if cv2.pointPolygonTest(face_polygon, (px, py), False) >= 0:
            return True
    return False


def save_log(start_dt, elapsed_sec, count):
    elapsed_min = elapsed_sec / 60
    per_min = count / elapsed_min if elapsed_min > 0 else 0

    entry = (
        f"{'='*40}\n"
        f"날짜       : {start_dt.strftime('%Y-%m-%d')}\n"
        f"시작 시각  : {start_dt.strftime('%H:%M:%S')}\n"
        f"소요 시간  : {int(elapsed_min)}분 {int(elapsed_sec % 60)}초\n"
        f"접촉 횟수  : {count}회\n"
        f"분당 접촉  : {per_min:.1f}회/분\n"
    )

    print("\n" + entry)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
    print(f"저장됨: {LOG_FILE}")


def main():
    download_models()

    face_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    count = 0
    touching = False
    start_time = time.time()
    start_dt = datetime.now()

    print("실행 중... (Q 키로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        face_result = face_landmarker.detect(mp_image)
        hand_result = hand_landmarker.detect(mp_image)

        face_polygon = None
        if face_result.face_landmarks:
            face_polygon = get_face_polygon(face_result.face_landmarks[0], w, h)
            cv2.polylines(frame, [face_polygon], isClosed=True, color=(0, 200, 0), thickness=2)

        currently_touching = False
        if hand_result.hand_landmarks:
            for hand_lm in hand_result.hand_landmarks:
                for lm in hand_lm:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 100, 0), -1)
                if face_polygon is not None and hand_in_face(hand_lm, face_polygon, w, h):
                    currently_touching = True

        if currently_touching and not touching:
            count += 1
        touching = currently_touching

        # 경과 시간 계산
        elapsed = int(time.time() - start_time)
        elapsed_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

        # 반투명 배경
        overlay = frame.copy()
        box_h = 130 if touching else 90
        cv2.rectangle(overlay, (10, 10), (420, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        color = (0, 0, 255) if touching else (255, 255, 255)
        cv2.putText(frame, f"Touch count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"Elapsed: {elapsed_str}", (20, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        if touching:
            cv2.putText(frame, "TOUCHING!", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Do Not Touch Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    hand_landmarker.close()

    save_log(start_dt, time.time() - start_time, count)


if __name__ == "__main__":
    main()
