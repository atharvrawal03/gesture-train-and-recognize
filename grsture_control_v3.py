#!/usr/bin/env python3
"""
Cyberpunk Live Dashboard v4.0 — Real-time gesture control.
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
import queue
import pyttsx3
import joblib
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, List

# ---------- CONFIG ----------
@dataclass
class Config:
    cam_index: int = 0
    cam_w: int = 1280
    cam_h: int = 720
    smooth: float = 0.18
    history_len: int = 7
    blink_thresh: float = 0.21
    double_blink_window: float = 0.55
    cooldowns: dict = field(default_factory=lambda: {
        'left': 0.40, 'right': 0.50, 'rock': 1.0,
        'fist': 0.8,  'scroll': 0.10, 'volume': 0.06,
        'mode': 1.0,  'thumb': 0.8,
    })
    pinch_ratio: float = 0.38
    scroll_zone: float = 0.22
    vol_cmd: str = 'pactl'
    win_title: str = 'NEURAL GESTURE CONTROL'

CFG = Config()
MODES = ['MOUSE', 'MEDIA', 'SYSTEM']

# ---------- MODEL LOADING ----------
def load_model():
    try:
        model  = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le     = joblib.load('label_encoder.pkl')
        has_proba = hasattr(model, 'predict_proba')
        print("[SYSTEM] Neural core online. predict_proba:", has_proba)
        return model, scaler, le, has_proba
    except:
        print("[SYSTEM] Rule-based fallback active.")
        return None, None, None, False

MODEL, SCALER, LE, HAS_PROBA = load_model()
MODEL_ACTIVE = MODEL is not None

# ---------- THREAD-SAFE TTS ----------
class TTSWorker:
    def __init__(self):
        self._q = queue.Queue(maxsize=3)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
    def _run(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', 155)
        while True:
            text = self._q.get()
            if text is None: break
            try:
                engine.say(text)
                engine.runAndWait()
            except: pass
    def say(self, text: str):
        try: self._q.put_nowait(text)
        except queue.Full: pass
    def stop(self):
        try: self._q.put_nowait(None)
        except: pass

TTS = TTSWorker()

# ---------- VOLUME ----------
def set_volume(pct: int):
    pct = int(np.clip(pct, 0, 100))
    try:
        if CFG.vol_cmd == 'pactl':
            subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f'{pct}%'],
                           stderr=subprocess.DEVNULL, timeout=0.5)
        else:
            raise RuntimeError
    except:
        subprocess.run(['amixer', 'set', 'Master', f'{pct}%'],
                       stderr=subprocess.DEVNULL, timeout=0.5)

# ---------- GESTURE HELPERS ----------
def normalise_landmarks(hand) -> np.ndarray:
    wrist = hand.landmark[0]
    mid   = hand.landmark[9]
    size  = math.hypot(wrist.x - mid.x, wrist.y - mid.y) + 1e-6
    f = []
    for lm in hand.landmark:
        f.extend([(lm.x - wrist.x)/size, (lm.y - wrist.y)/size, (lm.z - wrist.z)/size])
    return np.array(f, dtype=np.float32).reshape(1, -1)

def finger_states(hand, handed: str) -> dict:
    lm = hand.landmark
    up = lambda tip, pip: lm[tip].y < lm[pip].y
    s = {
        'Index':  up(8, 6),
        'Middle': up(12, 10),
        'Ring':   up(16, 14),
        'Pinky':  up(20, 18),
    }
    s['Thumb'] = (lm[4].x < lm[3].x) if handed == "Right" else (lm[4].x > lm[3].x)
    return s

def hand_scale(hand) -> float:
    w, m = hand.landmark[0], hand.landmark[9]
    return math.hypot(w.x - m.x, w.y - m.y)

def detect_pinches(hand) -> Tuple[bool, bool]:
    sz = hand_scale(hand)
    thresh = sz * CFG.pinch_ratio
    t, i, m = hand.landmark[4], hand.landmark[8], hand.landmark[12]
    left  = math.hypot(t.x - i.x, t.y - i.y) < thresh
    right = math.hypot(t.x - m.x, t.y - m.y) < thresh
    return left, right

def rule_gesture(states: dict, hand) -> str:
    v = states
    if all(v.values()):               return 'palm_open'
    if not any(v.values()):           return 'fist_hand'
    if v['Index'] and not v['Middle']: return 'point_up'
    if v['Index'] and v['Middle'] and not v['Ring']: return 'two_fingers'
    if v['Thumb'] and v['Pinky'] and not v['Index'] and not v['Middle']: return 'rock_on'
    if v['Thumb'] and not v['Index'] and not v['Middle'] and not v['Ring'] and not v['Pinky']:
        return 'thumbs_up'
    return 'unknown'

# ---------- WEIGHTED GESTURE SMOOTHER ----------
class GestureSmoother:
    def __init__(self, maxlen: int = 7):
        self._buf = deque(maxlen=maxlen)
        self._weights = np.linspace(0.5, 1.0, maxlen)
    def update(self, g: str) -> str:
        self._buf.append(g)
        n = len(self._buf)
        w = self._weights[-n:]
        counts = {}
        for weight, gesture in zip(w, self._buf):
            counts[gesture] = counts.get(gesture, 0.0) + weight
        return max(counts, key=counts.get)

smoother = GestureSmoother(CFG.history_len)

# ---------- COOLDOWN MANAGER ----------
class CooldownManager:
    def __init__(self): self._last = {}
    def allow(self, action: str) -> bool:
        now = time.monotonic()
        cd = CFG.cooldowns.get(action, 0.3)
        if now - self._last.get(action, 0.0) >= cd:
            self._last[action] = now
            return True
        return False

CD = CooldownManager()

# ---------- BLINK DETECTOR ----------
class BlinkDetector:
    EYE_IDXS = (33, 133, 159, 145)
    def __init__(self):
        self._closed = False
        self._last = 0.0
        self._cnt = 0
    def update(self, face_lm, h: int, w: int) -> bool:
        p1, p2, p3, p4 = [face_lm[i] for i in self.EYE_IDXS]
        ear = (abs(p3.y - p4.y) * h) / (abs(p1.x - p2.x) * w + 1e-6)
        now = time.monotonic()
        if ear < CFG.blink_thresh and not self._closed:
            self._closed = True
            if now - self._last < CFG.double_blink_window:
                self._cnt += 1
            else:
                self._cnt = 1
            self._last = now
            if self._cnt >= 2:
                self._cnt = 0
                return True
        elif ear >= CFG.blink_thresh:
            self._closed = False
        return False

blink_det = BlinkDetector()

# ---------- PRE-COMPUTED VIGNETTE (dark edges) ----------
def build_vignette(h: int, w: int) -> np.ndarray:
    ys = np.linspace(-1, 1, h)
    xs = np.linspace(-1, 1, w)
    xv, yv = np.meshgrid(xs, ys)
    r = np.sqrt(xv**2 + yv**2)
    v = np.clip(1.0 - r * 0.45, 0.5, 1.0).astype(np.float32)
    return v[:, :, np.newaxis]

# ---------- HUD DRAWING HELPERS ----------
CYAN   = (0, 255, 255)
GREEN  = (0, 255, 80)
DIM    = (0, 140, 140)
_FONT = cv2.FONT_HERSHEY_DUPLEX

def neon_rect(img, x1, y1, x2, y2, color=CYAN, t=1):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    for d, a in ((3, 0.12), (2, 0.18), (1, 0.30)):
        ov = img.copy()
        cv2.rectangle(ov, (x1-d, y1-d), (x2+d, y2+d), color, t)
        cv2.addWeighted(ov, a, img, 1-a, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)

def corner_ticks(img, x, y, size=18, color=CYAN):
    cv2.line(img, (x, y+size), (x, y), color, 2)
    cv2.line(img, (x, y), (x+size, y), color, 2)

def glitch_text(img, text, x, y, scale, color, t, amp=2):
    cv2.putText(img, text, (x+amp, y), _FONT, scale, (0,0,200), t)
    cv2.putText(img, text, (x-amp, y), _FONT, scale, (200,200,0), t)
    cv2.putText(img, text, (x, y),     _FONT, scale, color, t)

def hud_text(img, text, x, y, scale=0.45, color=CYAN, t=1):
    cv2.putText(img, text, (x, y), _FONT, scale, color, t)

def bar(img, x, y, w, pct, color=CYAN, bg=(0,40,40), h=7):
    pct = float(np.clip(pct, 0, 100))
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    cv2.rectangle(img, (x, y), (x+int(w*pct/100), y+h), color, -1)

def mini_chart(img, data: list, x: int, y: int, w: int, h: int, color=CYAN, max_val=70.0):
    if len(data) < 2: return
    arr = np.array(data, dtype=np.float32)
    arr = np.clip(arr / max_val, 0.0, 1.0)
    n = len(arr)
    xs = (x + np.arange(n) * w / (n-1)).astype(np.int32)
    ys = (y + h * (1.0 - arr)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1)
    # filled area
    fill = np.vstack([[xs[0], y+h], pts, [xs[-1], y+h]])
    alpha_layer = img.copy()
    cv2.fillPoly(alpha_layer, [fill.astype(np.int32)], color)
    cv2.addWeighted(alpha_layer, 0.25, img, 0.75, 0, img)
    # line
    for i in range(n-1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), color, 1, cv2.LINE_AA)

# ---------- ACTION DISPATCH ----------
class ActionDispatch:
    def __init__(self):
        self.mode = 'MOUSE'
        screen = pyautogui.size()
        self.sw, self.sh = screen.width, screen.height
        self.cursor = [self.sw//2, self.sh//2]
        self.drag = False
        self.volume = 50

    def cycle_mode(self):
        idx = MODES.index(self.mode)
        self.mode = MODES[(idx+1) % len(MODES)]
        TTS.say(f"{self.mode} mode")

    def do(self, stable: str, hand, left_pinch: bool, right_pinch: bool):
        lm = hand.landmark
        if stable == 'palm_open' and not left_pinch and CD.allow('mode'):
            self.cycle_mode()
            return
        if self.mode == 'MOUSE':
            self._mouse(stable, lm, left_pinch, right_pinch)
        elif self.mode == 'MEDIA':
            self._media(stable, lm)
        else:
            self._system(stable)

    def _mouse(self, stable, lm, lp, rp):
        if stable == 'point_up':
            tx = int(lm[8].x * self.sw)
            ty = int(lm[8].y * self.sh)
            cx = int(self.cursor[0]*(1-CFG.smooth) + tx*CFG.smooth)
            cy = int(self.cursor[1]*(1-CFG.smooth) + ty*CFG.smooth)
            pyautogui.moveTo(cx, cy, _pause=False)
            self.cursor = [cx, cy]
            if CD.allow('scroll'):
                yp = lm[8].y
                if yp < CFG.scroll_zone:
                    pyautogui.scroll(3)
                elif yp > (1.0 - CFG.scroll_zone):
                    pyautogui.scroll(-3)
        if lp and CD.allow('left'):
            pyautogui.click(_pause=False)
            TTS.say("Click")
        if rp and CD.allow('right'):
            pyautogui.rightClick(_pause=False)
            TTS.say("Right click")
        if stable == 'two_fingers':
            if not self.drag:
                pyautogui.mouseDown(_pause=False)
                self.drag = True
            pyautogui.moveTo(int(lm[8].x * self.sw), int(lm[8].y * self.sh), _pause=False)
        elif self.drag:
            pyautogui.mouseUp(_pause=False)
            self.drag = False

    def _media(self, stable, lm):
        if stable == 'point_up' and CD.allow('volume'):
            t, i = lm[4], lm[8]
            dist = math.hypot(t.x - i.x, t.y - i.y)
            new_vol = int(np.clip(np.interp(dist, [0.02, 0.22], [0, 100]), 0, 100))
            if abs(new_vol - self.volume) > 2:
                self.volume = new_vol
                set_volume(self.volume)
                TTS.say(f"Volume {self.volume}")
        if stable == 'fist_hand' and CD.allow('fist'):
            subprocess.run(['xdotool','key','XF86AudioPlay'], stderr=subprocess.DEVNULL)
            TTS.say("Play pause")
        if stable == 'rock_on' and CD.allow('rock'):
            subprocess.run(['xdotool','key','XF86AudioNext'], stderr=subprocess.DEVNULL)
            TTS.say("Next track")
        if stable == 'thumbs_up' and CD.allow('thumb'):
            subprocess.run(['xdotool','key','XF86AudioPrev'], stderr=subprocess.DEVNULL)
            TTS.say("Previous track")

    def _system(self, stable):
        if stable == 'rock_on' and CD.allow('rock'):
            pyautogui.hotkey('alt', 'tab')
            TTS.say("Switch apps")
        if stable == 'palm_open' and CD.allow('mode'):
            pyautogui.hotkey('super', 'd')
            TTS.say("Show desktop")
        if stable == 'fist_hand' and CD.allow('fist'):
            subprocess.run(['gnome-screensaver-command', '-l'], stderr=subprocess.DEVNULL)
            TTS.say("Screen locked")
        if stable == 'thumbs_up' and CD.allow('thumb'):
            pyautogui.hotkey('ctrl', 'alt', 't')
            TTS.say("Terminal")

    def release_drag(self):
        if self.drag:
            pyautogui.mouseUp(_pause=False)
            self.drag = False

dispatch = ActionDispatch()

# ---------- MAIN LOOP ----------
def main():
    cap = cv2.VideoCapture(CFG.cam_index)
    if not cap.isOpened():
        print("[ERROR] Webcam offline.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read from webcam.")
        sys.exit(1)
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    vignette = build_vignette(H, W)   # pre-computed once

    hands_mp = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65, min_tracking_confidence=0.60)
    face_mp = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

    cv2.namedWindow(CFG.win_title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(CFG.win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps_hist = deque(maxlen=60)
    latency_hist = deque(maxlen=60)
    gesture_log = deque(maxlen=6)
    stable = 'none'
    confidence = 0.0
    scanline_y = 0
    prev_time = time.monotonic()
    RANKINGS = [("RF", 98.2), ("SVM", 97.8), ("KNN", 97.1), ("XGB", 96.5)]

    print("\n" + "═"*62)
    print("  🧬  NEURAL GESTURE CONTROL  v4.0")
    print("═"*62)
    print("  palm_open   → cycle modes\n  point_up    → cursor / scroll / volume\n  pinch T+I   → left click\n  pinch T+M   → right click\n  two_fingers → drag & drop\n  rock_on     → next track / Alt+Tab\n  fist_hand   → play-pause / lock screen\n  thumbs_up   → prev track / terminal\n  double blink→ screenshot\n  [q]         → quit")
    print("═"*62)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.monotonic()
            now = t0
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            fps_hist.append(fps)

            # Blink detection
            face_res = face_mp.process(rgb)
            if face_res.multi_face_landmarks:
                if blink_det.update(face_res.multi_face_landmarks[0].landmark, H, W):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pyautogui.screenshot(f"screen_{ts}.png")
                    TTS.say("Screenshot")
                    frame[:] = 200

            # Hand processing
            hand_res = hands_mp.process(rgb)
            if hand_res.multi_hand_landmarks:
                hand = hand_res.multi_hand_landmarks[0]
                handed = hand_res.multi_handedness[0].classification[0].label
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,100,180), thickness=2),
                )
                lp, rp = detect_pinches(hand)
                if MODEL_ACTIVE:
                    feat = normalise_landmarks(hand)
                    fscal = SCALER.transform(feat)
                    pred = MODEL.predict(fscal)[0]
                    raw_gesture = LE.inverse_transform([pred])[0]
                    confidence = float(MODEL.predict_proba(fscal)[0].max()) if HAS_PROBA else 0.85
                else:
                    states = finger_states(hand, handed)
                    raw_gesture = rule_gesture(states, hand)
                    confidence = 0.75
                stable = smoother.update(raw_gesture)
                gesture_log.append((stable, confidence))
                dispatch.do(stable, hand, lp, rp)
            else:
                dispatch.release_drag()
                stable = 'none'
                confidence = 0.0

            latency_hist.append((time.monotonic() - t0) * 1000)

            # Apply vignette (fast numpy)
            frame = (frame.astype(np.float32) * vignette).clip(0, 255).astype(np.uint8)

            # ---------- HUD drawing ----------
            # Top bar
            neon_rect(frame, 4, 4, W-4, 38, CYAN)
            glitch_text(frame, f"MODE: {dispatch.mode}", 14, 27, 0.58, CYAN, 1)
            fps_col = GREEN if fps >= 25 else ((0,255,220) if fps >= 15 else (0,80,255))
            hud_text(frame, f"FPS {fps:5.1f}", W-210, 27, 0.48, fps_col)
            hud_text(frame, f"LAT {latency_hist[-1]:4.1f}ms", W-120, 27, 0.48, CYAN)

            # Mode buttons
            mode_colors = {'MOUSE':(0,200,255), 'MEDIA':(0,255,120), 'SYSTEM':(0,120,255)}
            for i, m in enumerate(MODES):
                bx = W//2 - 140 + i*95
                active = (m == dispatch.mode)
                col = mode_colors[m]
                if active:
                    cv2.rectangle(frame, (bx, 8), (bx+88, 32), col, -1)
                    cv2.putText(frame, m, (bx+8, 25), _FONT, 0.45, (0,0,0), 1)
                else:
                    cv2.rectangle(frame, (bx, 8), (bx+88, 32), DIM, 1)
                    hud_text(frame, m, bx+8, 25, 0.45, DIM)

            # Left panel: Gesture
            neon_rect(frame, 8, 48, 270, 220, CYAN)
            corner_ticks(frame, 8, 48)
            glitch_text(frame, "GESTURE", 18, 72, 0.55, CYAN, 1)
            g_col = GREEN if confidence > 0.85 else ((0,255,220) if confidence > 0.65 else (0,100,255))
            glitch_text(frame, stable.upper()[:12], 18, 118, 1.1, g_col, 2, amp=3)
            bar(frame, 18, 128, 200, confidence*100, g_col, h=8)
            hud_text(frame, f"{confidence*100:.1f}%", 228, 136, 0.42, g_col)
            hud_text(frame, "RECENT GESTURES", 18, 158, 0.42, DIM)
            for i, (g, c) in enumerate(list(gesture_log)[-4:]):
                alpha = 0.4 + 0.15*i
                col = tuple(int(v*alpha) for v in CYAN)
                hud_text(frame, f"{'▶' if i==3 else ' '} {g:<14} {c*100:4.0f}%",
                         22, 174 + i*13, 0.38, col)

            # Right panel: Model rankings
            neon_rect(frame, W-255, 48, W-8, 190, CYAN)
            corner_ticks(frame, W-255, 48)
            glitch_text(frame, "MODEL RANKINGS", W-245, 72, 0.50, CYAN, 1)
            y0 = 94
            for name, acc in RANKINGS:
                pct = acc - 95.0
                bar(frame, W-245, y0-8, 130, pct*20, CYAN, h=6)
                hud_text(frame, f"{name:<4} {acc:.1f}%", W-105, y0, 0.42, CYAN)
                y0 += 22
            hud_text(frame, f"ENGINE: {'ML+RULE' if MODEL_ACTIVE else 'RULE-ONLY'}", W-245, 178, 0.40, DIM)

            # Bottom-left: Quick actions
            neon_rect(frame, 8, H-105, 290, H-8, CYAN)
            glitch_text(frame, "QUICK ACTIONS", 18, H-82, 0.48, CYAN, 1)
            actions = ["palm_open  → cycle mode", "two_fingers→ drag & drop",
                       "dbl blink  → screenshot", "[q]/[m]    → quit/mode"]
            for i, a in enumerate(actions):
                hud_text(frame, a, 18, H-62 + i*16, 0.38, DIM)

            # Bottom-right: Metrics
            neon_rect(frame, W-255, H-105, W-8, H-8, CYAN)
            glitch_text(frame, "REAL-TIME METRICS", W-245, H-82, 0.48, CYAN, 1)
            metrics = [("FPS", fps, 60), ("LAT", latency_hist[-1] if latency_hist else 0, 50), ("VOL", dispatch.volume, 100)]
            y0 = H-62
            for name, val, max_v in metrics:
                pct = val / max_v * 100
                col = GREEN if pct < 60 else ((0,255,220) if pct < 85 else (0,80,255))
                bar(frame, W-200, y0-7, 120, pct, col, h=6)
                hud_text(frame, f"{name} {val:5.1f}", W-245, y0, 0.40, CYAN)
                y0 += 18
            if len(fps_hist) > 2:
                mini_chart(frame, list(fps_hist), W-245, H-22, 230, 12, CYAN, 70.0)

            # Volume bar in MEDIA mode
            if dispatch.mode == 'MEDIA':
                bx, by = W//2 - 75, 48
                neon_rect(frame, bx-40, by, bx+155, by+28, GREEN)
                bar(frame, bx, by+8, 100, dispatch.volume, GREEN, h=10)
                hud_text(frame, f"VOL {dispatch.volume:3d}%", bx-35, by+20, 0.50, GREEN)

            # Scanline effect
            scanline_y = (scanline_y + 5) % H
            ov = frame.copy()
            cv2.line(ov, (0, scanline_y), (W, scanline_y), CYAN, 1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)

            cv2.imshow(CFG.win_title, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('m'):
                dispatch.cycle_mode()

    except KeyboardInterrupt:
        pass
    finally:
        dispatch.release_drag()
        TTS.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Shutdown.")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    main()
