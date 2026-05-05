
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import mediapipe.python.solutions as mp_solutions
mp_hands = mp_solutions.hands
mp_face = mp_solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)
face_mesh = mp_face.FaceMesh()


