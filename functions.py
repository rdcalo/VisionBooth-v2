import math
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    return thumb_tip.y < thumb_mcp.y - 0.05  

def is_thumb_down(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    return thumb_tip.y > thumb_mcp.y + 0.05  

def calculate_distance_xyz(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 +
                     (point1.y - point2.y) ** 2 +
                     (point1.z - point2.z) ** 2)

def calculate_distance_xy(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def calculate_normalized_distance(x1, y1, x2, y2, image_width, image_height):
    dx = (x1 - x2) * image_width
    dy = (y1 - y2) * image_height
    return math.sqrt(dx ** 2 + dy ** 2)

def calculate_scale(face_landmarks, image_width, image_height):
    LEFT_EYE = 159
    RIGHT_EYE = 386
    left_eye = face_landmarks.landmark[LEFT_EYE]
    right_eye = face_landmarks.landmark[RIGHT_EYE]
    interocular_distance = calculate_normalized_distance(
        left_eye.x, left_eye.y, right_eye.x, right_eye.y, image_width, image_height
    )
    return interocular_distance

def calculate_angle_2points(landmark1, landmark2):
    return np.arctan2(landmark2.y - landmark1.y, landmark2.x - landmark1.x) * 180 / np.pi

def calculate_angle_3points(point1, point2, point3):
    vector1 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
    vector2 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]
    dot_product = sum([vector1[i] * vector2[i] for i in range(3)])
    norm1 = math.sqrt(sum([vector1[i] ** 2 for i in range(3)]))
    norm2 = math.sqrt(sum([vector2[i] ** 2 for i in range(3)]))
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = max(min(cos_angle, 1), -1)
    angle = math.degrees(math.acos(cos_angle))
    return angle

def is_finger_extended(hand_landmarks, finger_tip, finger_pip):
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y

def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    if thumb_tip.y < wrist.y and index_tip.y > thumb_tip.y:
        angle = calculate_angle_3points(thumb_mcp, thumb_tip, wrist)
        if angle < 4:
            return True
    return False

def detect_smile(face_landmarks):
    return False

def detect_puckered_lips(face_landmarks):
    return False

def detect_raised_eyebrows(face_landmarks):
    return False

SIGMA_DETECTOR = None
