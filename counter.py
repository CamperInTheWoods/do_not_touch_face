"""
얼굴 손대기 카운터
손이 얼굴 영역에 닿을 때마다 카운트합니다.
종료: Q 키
"""

import urllib.request
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

CAMERA_INDEX = 0
MODEL_DIR = Path(__file__).parent / "models"

FACE_MODEL_PATH = MODEL_DIR / "face_detector.tflite"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download_models():
    MODEL_DIR.mkdir(exist_ok=True)
    for url, path in [(FACE_MODEL_URL, FACE_MODEL_PATH), (HAND_MODEL_URL, HAND_MODEL_PATH)]:
        if not path.exists():
            print(f"모델 다운로드 중: {path.name} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  완료: {path.name}")


def get_face_box(detection, w, h):
    bbox = detection.bounding_box
    x1 = bbox.origin_x
    y1 = bbox.origin_y
    x2 = x1 + bbox.width
    y2 = y1 + bbox.height
    pad_x = int(bbox.width  * 0.1)
    pad_y = int(bbox.height * 0.1)
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y


def hand_in_face(hand_landmarks, face_box, w, h):
    x1, y1, x2, y2 = face_box
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        if x1 < px < x2 and y1 < py < y2:
            return True
    return False


def main():
    download_models()

    face_options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        min_detection_confidence=0.6
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    face_detector  = mp_vision.FaceDetector.create_from_options(face_options)
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    count = 0
    touching = False

    print("실행 중... (Q 키로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        face_result = face_detector.detect(mp_image)
        hand_result = hand_landmarker.detect(mp_image)

        face_box = None
        if face_result.detections:
            face_box = get_face_box(face_result.detections[0], w, h)
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        currently_touching = False
        if hand_result.hand_landmarks:
            for hand_lm in hand_result.hand_landmarks:
                # 손 랜드마크 그리기
                for lm in hand_lm:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 100, 0), -1)
                if face_box and hand_in_face(hand_lm, face_box, w, h):
                    currently_touching = True

        if currently_touching and not touching:
            count += 1
        touching = currently_touching

        color = (0, 0, 255) if touching else (255, 255, 255)
        cv2.putText(frame, f"Touch count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        if touching:
            cv2.putText(frame, "TOUCHING!", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Do Not Touch Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()
    hand_landmarker.close()
    print(f"총 얼굴 접촉 횟수: {count}")


if __name__ == "__main__":
    main()
