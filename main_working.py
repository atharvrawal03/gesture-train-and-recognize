#!/usr/bin/env python3
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
import pyautogui, numpy as np, time, subprocess, math, threading, pyttsx3, joblib
from datetime import datetime
from collections import deque

# Load model if exists
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    MODEL = True
    print("✅ Loaded trained model")
except:
    MODEL = False
    print("⚠️ No model – using rule‑based")

# Settings
SCREEN_W, SCREEN_H = pyautogui.size()
SMOOTH = 0.2
HISTORY = 5
BLINK_THRESH = 0.2
DOUBLE_TIME = 0.5
COOLDOWN = {'left':0.5,'right':0.5,'rock':1.0,'fist':0.8,'scroll':0.1,'volume':0.05,'mode':1.0}
MODES = ['MOUSE','MEDIA','SYSTEM']
current_mode = 'MOUSE'

hands = mp_hands.Hands(max_num_hands=1)
face_mesh = mp_face_mesh.FaceMesh()

# State
cursor = [SCREEN_W//2, SCREEN_H//2]
drag = False
last = {}
gest_buf = deque(maxlen=HISTORY)
blink = {'closed':False,'last':0,'cnt':0}
pause_until = 0
volume = 50
fps = 0
prev_time = time.time()

# Voice
tts = pyttsx3.init()
tts.setProperty('rate',150)
def speak(text):
    threading.Thread(target=lambda: (tts.say(text), tts.runAndWait()), daemon=True).start()

# Normalization (same as training)
def norm(hand):
    wrist = hand.landmark[0]
    mid = hand.landmark[9]
    size = math.hypot(wrist.x-mid.x, wrist.y-mid.y) + 1e-6
    f = []
    for lm in hand.landmark:
        f.extend([(lm.x-wrist.x)/size, (lm.y-wrist.y)/size, (lm.z-wrist.z)/size])
    return np.array(f).reshape(1,-1)

# Rule‑based fallback helpers
def finger_states(hand, handed):
    s = {}
    s['Index'] = hand.landmark[8].y < hand.landmark[6].y
    s['Middle'] = hand.landmark[12].y < hand.landmark[10].y
    s['Ring'] = hand.landmark[16].y < hand.landmark[14].y
    s['Pinky'] = hand.landmark[20].y < hand.landmark[18].y
    if handed == "Right":
        s['Thumb'] = hand.landmark[4].x < hand.landmark[3].x
    else:
        s['Thumb'] = hand.landmark[4].x > hand.landmark[3].x
    return s

def hand_size_rule(hand):
    w = hand.landmark[0]; m = hand.landmark[9]
    return math.hypot(w.x-m.x, w.y-m.y)

def rule_gesture(states, hand):
    sz = hand_size_rule(hand); thresh = sz * 0.4
    t,i,m = hand.landmark[4], hand.landmark[8], hand.landmark[12]
    left = math.hypot(t.x-i.x, t.y-i.y) < thresh
    right = math.hypot(t.x-m.x, t.y-m.y) < thresh
    if all(states.values()): return 'palm_open', left, right
    if not any(states.values()): return 'fist_hand', left, right
    if states['Index'] and not states['Middle']: return 'point_up', left, right
    if states['Index'] and states['Middle']: return 'two_fingers', left, right
    if states['Thumb'] and states['Pinky'] and not states['Index'] and not states['Middle']: return 'rock_on', left, right
    if states['Thumb'] and not any([states['Index'],states['Middle'],states['Ring'],states['Pinky']]): return 'thumbs_up', left, right
    return 'unknown', left, right

def double_blink(face_lm, h, w):
    p1,p2,p3,p4 = face_lm[33], face_lm[133], face_lm[159], face_lm[145]
    ear = (abs(p3.y-p4.y)*h) / (abs(p1.x-p2.x)*w + 1e-6)
    now = time.time()
    if ear < BLINK_THRESH and not blink['closed']:
        blink['closed'] = True
        if now - blink['last'] < DOUBLE_TIME:
            blink['cnt'] += 1
        else:
            blink['cnt'] = 1
        blink['last'] = now
        if blink['cnt'] >= 2:
            blink['cnt'] = 0
            return True
    elif ear >= BLINK_THRESH:
        blink['closed'] = False
    return False

def allow(action):
    now = time.time()
    if now - last.get(action,0) < COOLDOWN.get(action,0.3):
        return False
    last[action] = now
    return True

def set_mode(m):
    global current_mode
    if m in MODES: current_mode = m; speak(f"{m} mode")

# Main loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found.")
    exit(1)

print("🎮 Gesture Control Started. Open palm = cycle modes. Double blink = screenshot. q=quit.")
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h,w = frame.shape[:2]
    now = time.time()
    fps = 1/(now - prev_time + 1e-6); prev_time = now

    # Blink detection
    face_res = face_mesh.process(rgb)
    if face_res.multi_face_landmarks:
        if double_blink(face_res.multi_face_landmarks[0].landmark, h, w):
            pyautogui.screenshot(f"screen_{datetime.now():%Y%m%d_%H%M%S}.png")
            speak("Screenshot")
            cv2.putText(frame,"📸 SCREENSHOT",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # Hand gestures
    if now < pause_until:
        cv2.putText(frame,"PAUSED",(w//2-50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    else:
        hand_res = hands.process(rgb)
        if hand_res.multi_hand_landmarks:
            hand = hand_res.multi_hand_landmarks[0]
            handed = hand_res.multi_handedness[0].classification[0].label
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Gesture recognition
            if MODEL:
                feat = norm(hand)
                feat_scaled = scaler.transform(feat)
                pred = model.predict(feat_scaled)[0]
                gesture = le.inverse_transform([pred])[0]
                states = finger_states(hand, handed)
                _, left_pinch, right_pinch = rule_gesture(states, hand)
                cv2.putText(frame, f"ML: {gesture}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
            else:
                states = finger_states(hand, handed)
                gesture, left_pinch, right_pinch = rule_gesture(states, hand)

            gest_buf.append(gesture)
            stable = max(set(gest_buf), key=gest_buf.count)

            # Mode switch
            if stable == 'palm_open' and not left_pinch and not right_pinch and allow('mode'):
                next_mode = MODES[(MODES.index(current_mode)+1)%len(MODES)]
                set_mode(next_mode)

            # ===== MOUSE =====
            if current_mode == 'MOUSE':
                if stable == 'point_up' and not left_pinch and not right_pinch:
                    tx = int(hand.landmark[8].x * SCREEN_W)
                    ty = int(hand.landmark[8].y * SCREEN_H)
                    cx = int(cursor[0]*(1-SMOOTH) + tx*SMOOTH)
                    cy = int(cursor[1]*(1-SMOOTH) + ty*SMOOTH)
                    pyautogui.moveTo(cx,cy); cursor = [cx,cy]
                if left_pinch and allow('left'): pyautogui.click(); speak("Click")
                if right_pinch and allow('right'): pyautogui.rightClick(); speak("Right click")
                if stable == 'two_fingers':
                    if not drag: pyautogui.mouseDown(); drag = True
                    pyautogui.moveTo(int(hand.landmark[8].x*SCREEN_W), int(hand.landmark[8].y*SCREEN_H))
                elif drag: pyautogui.mouseUp(); drag = False
                if stable == 'point_up' and not left_pinch and not right_pinch:
                    y_pos = hand.landmark[8].y
                    if (y_pos<0.2 or y_pos>0.8) and allow('scroll'):
                        pyautogui.scroll(3 if y_pos<0.2 else -3)

            # ===== MEDIA =====
            elif current_mode == 'MEDIA':
                if stable == 'point_up':
                    t,i = hand.landmark[4], hand.landmark[8]
                    dist = math.hypot(t.x-i.x, t.y-i.y)
                    new_vol = int(np.clip(np.interp(dist,[0.02,0.2],[0,100]),0,100))
                    if abs(new_vol-volume)>2 and allow('volume'):
                        volume = new_vol
                        subprocess.run(['amixer','set','Master',f"{volume}%"], stderr=subprocess.DEVNULL)
                        speak(f"Volume {volume}")
                if stable == 'fist_hand' and allow('fist'):
                    subprocess.run(['xdotool','key','XF86AudioPlay'], stderr=subprocess.DEVNULL)
                    speak("Play pause")
                if stable == 'rock_on' and allow('rock'):
                    subprocess.run(['xdotool','key','XF86AudioNext'], stderr=subprocess.DEVNULL)
                    speak("Next track")
                if stable == 'thumbs_up' and allow('thumb'):
                    subprocess.run(['xdotool','key','XF86AudioPrev'], stderr=subprocess.DEVNULL)
                    speak("Previous track")

            # ===== SYSTEM =====
            else:
                if stable == 'rock_on' and allow('rock'):
                    pyautogui.hotkey('alt','tab'); speak("Switch apps")
                if stable == 'palm_open' and allow('open_palm'):
                    pyautogui.hotkey('win','d'); speak("Show desktop")
                if stable == 'fist_hand' and allow('fist_lock'):
                    subprocess.run(['gnome-screensaver-command','-l'], stderr=subprocess.DEVNULL)
                    speak("Screen locked")
                if stable == 'thumbs_up' and allow('thumb'):
                    pyautogui.hotkey('ctrl','alt','t'); speak("Terminal")

        else:
            if drag: pyautogui.mouseUp(); drag = False

    # GUI overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (320,130), (0,0,0), -1)
    frame = cv2.addWeighted(overlay,0.3,frame,0.7,0)
    cv2.putText(frame, f"Mode: {current_mode}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
    if 'stable' in locals(): cv2.putText(frame, f"Gest: {stable}", (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1)
    if current_mode=='MEDIA': cv2.putText(frame, f"Vol: {volume}%", (10,120), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1)
    cv2.putText(frame, "q:quit  open palm:mode", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
    cv2.imshow('Team Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
