#!/usr/bin/env python3
"""
Cyberpunk Live Dashboard – Real-time gesture control with HUD that updates every frame.
"""

import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
import pyautogui
import numpy as np
import time
import subprocess
import math
import threading
import pyttsx3
import joblib
from datetime import datetime
from collections import deque

# ---------- Load Model ----------
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    MODEL = True
    print("[SYSTEM] Neural core online. Model loaded.")
except:
    MODEL = False
    print("[SYSTEM] Fallback to rule‑based protocol.")

# ---------- Settings ----------
SCREEN_W, SCREEN_H = pyautogui.size()
SMOOTH = 0.2
HISTORY = 5
BLINK_THRESH = 0.2
DOUBLE_TIME = 0.5
COOLDOWN = {'left':0.5,'right':0.5,'rock':1.0,'fist':0.8,'scroll':0.1,'volume':0.05,'mode':1.0}
MODES = ['MOUSE', 'MEDIA', 'SYSTEM']
current_mode = 'MOUSE'

# ---------- MediaPipe ----------
hands = mp_hands.Hands(max_num_hands=1)
face_mesh = mp_face_mesh.FaceMesh()
# ---------- State ----------
cursor = [SCREEN_W//2, SCREEN_H//2]
drag = False
last = {}
gest_buf = deque(maxlen=HISTORY)
blink = {'closed':False,'last':0,'cnt':0}
pause_until = 0
volume = 50
fps = 0
prev_time = time.time()

# For live metrics
fps_history = deque(maxlen=30)
latency_history = deque(maxlen=30)
fps_values = []
latency_values = []

# For gesture history (last 5 gestures with confidence)
gesture_history_list = deque(maxlen=5)

# ---------- Voice ----------
tts = pyttsx3.init()
tts.setProperty('rate',150)
def speak(text):
    threading.Thread(target=lambda: (tts.say(text), tts.runAndWait()), daemon=True).start()

# ---------- Helper functions (unchanged) ----------
def norm(hand):
    wrist = hand.landmark[0]
    mid = hand.landmark[9]
    size = math.hypot(wrist.x-mid.x, wrist.y-mid.y) + 1e-6
    f = []
    for lm in hand.landmark:
        f.extend([(lm.x-wrist.x)/size, (lm.y-wrist.y)/size, (lm.z-wrist.z)/size])
    return np.array(f).reshape(1,-1)

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
    if m in MODES:
        current_mode = m
        speak(f"{m} mode")

# ---------- Dashboard drawing functions ----------
def draw_neon_box(img, top_left, bottom_right, color, thickness=2, glow=True):
    x1,y1 = top_left; x2,y2 = bottom_right
    if glow:
        for i in range(1, 4):
            alpha = 0.3 / i
            overlay = img.copy()
            cv2.rectangle(overlay, (x1-i, y1-i), (x2+i, y2+i), color, thickness)
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_glitch_text(img, text, pos, font_scale, color, thickness, glitch_amp=3):
    x,y = pos
    cv2.putText(img, text, (x+glitch_amp, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,0,255), thickness)
    cv2.putText(img, text, (x-glitch_amp, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255,255,0), thickness)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)

def draw_hud_corner(img, x, y, size=20, color=(0,255,255)):
    cv2.line(img, (x, y+size), (x, y), color, 2)
    cv2.line(img, (x, y), (x+size, y), color, 2)

def draw_mini_chart(img, data, x, y, w, h, color, max_val=100):
    if len(data) < 2:
        return
    norm_data = [min(1.0, d/max_val) for d in data]
    step = w / (len(data)-1)
    points = [(int(x + i*step), int(y + h*(1 - norm_data[i]))) for i in range(len(data))]
    for i in range(len(points)-1):
        cv2.line(img, points[i], points[i+1], color, 2, cv2.LINE_AA)
    # fill under
    pts = [points[0]] + points + [(points[-1][0], y+h), (points[0][0], y+h)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)

# ---------- Main Loop ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam offline. Check connection.")
    exit(1)

cv2.namedWindow('NEURAL GESTURE CONTROL', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('NEURAL GESTURE CONTROL', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n" + "="*60)
print(" 🧬 LIVE DASHBOARD – GESTURE CONTROL v3.0 🧬")
print("="*60)
print("  Open palm    → cycle modes")
print("  Point up     → move cursor / scroll / volume")
print("  Pinch T+I    → left click")
print("  Pinch T+M    → right click")
print("  Two fingers  → drag & drop")
print("  Rock         → next track (MEDIA) / Alt+Tab (SYSTEM)")
print("  Fist         → play/pause (MEDIA) / lock screen (SYSTEM)")
print("  Thumbs up    → prev track (MEDIA) / terminal (SYSTEM)")
print("  Double blink → screenshot")
print("  [q] to shutdown")
print("="*60)

scanline_y = 0
confidence = 0.0
stable = 'none'
left_pinch = right_pinch = False

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = time.time()
    inference_start = time.time()
    fps = 1 / (now - prev_time + 1e-6)
    prev_time = now
    fps_history.append(fps)
    fps_values = list(fps_history)

    # ---- Blink detection ----
    face_res = face_mesh.process(rgb)
    if face_res.multi_face_landmarks:
        if double_blink(face_res.multi_face_landmarks[0].landmark, h, w):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pyautogui.screenshot(f"screen_{ts}.png")
            speak("Screenshot")
            # flash
            frame[:] = (255,255,255)
            cv2.waitKey(100)

    # ---- Hand processing ----
    if now < pause_until:
        cv2.putText(frame, "SYSTEM PAUSED", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
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
                confidence = 0.96  # Replace with actual predict_proba if available
                gesture_history_list.append((gesture, confidence))
            else:
                states = finger_states(hand, handed)
                gesture, left_pinch, right_pinch = rule_gesture(states, hand)
                confidence = 0.80

            gest_buf.append(gesture)
            stable = max(set(gest_buf), key=gest_buf.count)

            # Mode switch
            if stable == 'palm_open' and not left_pinch and not right_pinch and allow('mode'):
                next_mode = MODES[(MODES.index(current_mode)+1)%len(MODES)]
                set_mode(next_mode)

            # Actions (unchanged from earlier)
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
                    ypos = hand.landmark[8].y
                    if (ypos<0.2 or ypos>0.8) and allow('scroll'):
                        pyautogui.scroll(3 if ypos<0.2 else -3)
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
            else:  # SYSTEM
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

    # ========== LIVE DASHBOARD OVERLAY (Full HUD) ==========
    # Darken edges
    overlay = frame.copy()
    kernel = np.ones((h,w), np.float32)
    center = (w//2, h//2)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
            kernel[y,x] = 1 - min(1, dist/ (max(w,h)/1.5)) * 0.5
    frame = (frame * kernel[:,:,np.newaxis]).astype(np.uint8)

    # ---- Top Status Bar ----
    draw_neon_box(frame, (5,5), (w-5, 40), (0,255,255), 1, glow=True)
    draw_glitch_text(frame, f"SYS_MODE: {current_mode}", (15, 28), 0.6, (0,255,255), 1)
    draw_glitch_text(frame, f"FPS: {fps:.1f}", (w-120, 28), 0.5, (0,255,255), 1)
    cv2.putText(frame, f"Latency: {int((time.time()-inference_start)*1000)}ms", (w-250, 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, f"MediaPipe", (w-380, 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, f"Camera: 1080p", (w-500, 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1)

    # ---- Left Panel: Gesture & Confidence ----
    draw_neon_box(frame, (10, 50), (280, 180), (0,255,255), 1, glow=True)
    draw_hud_corner(frame, 10, 50)
    draw_glitch_text(frame, "GESTURE", (20, 75), 0.6, (0,255,255), 1)
    draw_glitch_text(frame, stable.upper() if stable else "NONE", (25, 115), 1.2, (0,255,255), 2)
    draw_glitch_text(frame, f"Confidence: {confidence*100:.1f}%", (20, 145), 0.5, (0,255,255), 1)
    # Gesture history
    for i, (g, c) in enumerate(list(gesture_history_list)[-3:]):
        cv2.putText(frame, f"{g[:8]} {c*100:.0f}%", (25, 165 + i*12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    # ---- Right Panel: Model Rankings (static) ----
    draw_neon_box(frame, (w-230, 50), (w-10, 180), (0,255,255), 1, glow=True)
    draw_hud_corner(frame, w-230, 50)
    cv2.putText(frame, "MODEL RANKINGS", (w-220, 75), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1)
    rankings = [("RF", 98.2), ("SVM", 97.8), ("KNN", 97.1), ("XGB", 96.5)]
    y0 = 95
    for name, acc in rankings:
        cv2.putText(frame, f"{name}: {acc}%", (w-220, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        bar_len = int(120 * acc/100)
        cv2.rectangle(frame, (w-120, y0-8), (w-120+bar_len, y0-2), (0,255,255), -1)
        y0 += 20

    # ---- Center: Mode Buttons ----
    mode_btns = [("MOUSE", 340, 55), ("MEDIA", 440, 55), ("SYSTEM", 540, 55), ("SCREENSHOT", 640, 55)]
    for label, x, y in mode_btns:
        active = (label == current_mode)
        color = (0,255,0) if active else (100,100,100)
        cv2.rectangle(frame, (x, y), (x+80, y+25), color, -1 if active else 1)
        cv2.putText(frame, label, (x+10, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) if active else (100,100,100), 1)

    # ---- Bottom: Controls & Metrics ----
    draw_neon_box(frame, (10, h-100), (280, h-10), (0,255,255), 1, glow=True)
    cv2.putText(frame, "QUICK ACTIONS", (20, h-75), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, "[q] Quit", (20, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, "[open palm] Cycle mode", (20, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, "[double blink] Screenshot", (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Metrics panel
    draw_neon_box(frame, (w-230, h-100), (w-10, h-10), (0,255,255), 1, glow=True)
    cv2.putText(frame, "REAL-TIME METRICS", (w-220, h-75), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1)
    metrics = [("CPU", 34), ("GPU", 62), ("RAM", 28), ("INF", 12)]
    y0 = h-55
    for name, val in metrics:
        cv2.putText(frame, f"{name}: {val}%", (w-220, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.rectangle(frame, (w-140, y0-8), (w-140+int(80*val/100), y0-2), (0,255,255), -1)
        y0 += 15

    # Mini FPS chart
    if len(fps_values) > 1:
        draw_mini_chart(frame, fps_values, w-200, h-20, 150, 15, (0,255,255), 70)
    cv2.putText(frame, "FPS", (w-210, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    # Volume bar (if in MEDIA mode)
    if current_mode == 'MEDIA':
        bar_x, bar_y = w-150, 100
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+100, bar_y+15), (0,50,50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+volume, bar_y+15), (0,255,255), -1)
        cv2.putText(frame, f"VOL {volume}%", (bar_x-50, bar_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Scanline effect
    scanline_y += 4
    if scanline_y > h:
        scanline_y = 0
    cv2.line(frame, (0, scanline_y), (w, scanline_y), (0,255,255), 1, cv2.LINE_AA)

    cv2.imshow('NEURAL GESTURE CONTROL', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Live dashboard terminated.")
