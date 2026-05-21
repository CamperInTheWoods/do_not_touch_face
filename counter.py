"""
얼굴 손대기 카운터
손이 얼굴 영역에 닿을 때마다 카운트합니다.
종료: Q 키
"""

import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CAMERA_INDEX = 0


def get_face_box(detection, w, h):
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)
    # 얼굴 영역을 조금 여유있게
    pad = int(bh * 0.1)
    return x - pad, y - pad, x + bw + pad, y + bh + pad


def hand_in_face(hand_landmarks, face_box, w, h):
    x1, y1, x2, y2 = face_box
    for lm in hand_landmarks.landmark:
        px = int(lm.x * w)
        py = int(lm.y * h)
        if x1 < px < x2 and y1 < py < y2:
            return True
    return False


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    count = 0
    touching = False  # 이전 프레임 접촉 여부 (중복 카운트 방지)

    with mp_face.FaceDetection(min_detection_confidence=0.6) as face_det, \
         mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands_det:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_result  = face_det.process(rgb)
            hands_result = hands_det.process(rgb)

            face_box = None
            if face_result.detections:
                detection = face_result.detections[0]
                face_box = get_face_box(detection, w, h)
                x1, y1, x2, y2 = face_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            currently_touching = False
            if hands_result.multi_hand_landmarks:
                for hand_lm in hands_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                    if face_box and hand_in_face(hand_lm, face_box, w, h):
                        currently_touching = True

            # 닿기 시작하는 순간만 카운트
            if currently_touching and not touching:
                count += 1
            touching = currently_touching

            # 화면 표시
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
    print(f"총 얼굴 접촉 횟수: {count}")


if __name__ == "__main__":
    main()
