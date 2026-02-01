import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

canvas = None

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        hand_lms = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # overlay (nothing on canvas yet)
    output = frame.copy()
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    output[mask] = canvas[mask]

    cv2.imshow("Air Draw", output)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()