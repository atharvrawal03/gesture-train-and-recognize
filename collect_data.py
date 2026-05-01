#!/usr/bin/env python3
# gesture data collector – press 1-7 to record
import cv2, mediapipe as mp, numpy as np, pickle, math

GESTURES = {
    49: 'point_up',    # 1
    50: 'fist_hand',   # 2
    51: 'rock_on',     # 3
    52: 'palm_open',   # 4
    53: 'two_fingers', # 5
    54: 'thumbs_up',   # 6
    55: 'peace_sign'   # 7
}

def normalize(hand):
    wrist = hand.landmark[0]
    mid = hand.landmark[9]
    hand_size = math.hypot(wrist.x - mid.x, wrist.y - mid.y) + 1e-6
    feat = []
    for lm in hand.landmark:
        feat.append((lm.x - wrist.x) / hand_size)
        feat.append((lm.y - wrist.y) / hand_size)
        feat.append((lm.z - wrist.z) / hand_size)
    return feat

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)
data, labels = [], []

print("Show your hand. Press keys 1 to 7 to record. Press q to quit.")
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        key = cv2.waitKey(1) & 0xFF
        if key in GESTURES:
            feat = normalize(hand)
            data.append(feat)
            labels.append(GESTURES[key])
            print(f"✅ {GESTURES[key]} recorded – total {len(data)} samples")
        elif key == ord('q'):
            break
    cv2.imshow("Gesture Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
with open('gesture_data.pkl', 'wb') as f:
    pickle.dump((data, labels), f)
print(f"\nSaved {len(data)} samples to gesture_data.pkl")
