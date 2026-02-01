import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

canvas = None
prev_point = None

BRUSH = 12
COLOR = (255, 255, 255)  # white ink

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
        lm = res.multi_hand_landmarks[0].landmark

        x = int(lm[8].x * w)
        y = int(lm[8].y * h)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        if prev_point is None:
            prev_point = (x, y)

            cv2.line(canvas, prev_point, (x, y), COLOR, BRUSH)
            prev_point = (x, y)
        else:
            prev_point = None
    else:
        prev_point = None

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