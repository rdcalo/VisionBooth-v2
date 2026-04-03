import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def count_fingers(hand_landmarks, handedness):
    fingers = []

    if handedness.classification[0].label == "Right":
        fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x <
                       hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)
    else:
        fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x >
                       hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)


    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
                   hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y <
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y <
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y)

    return fingers.count(True)
