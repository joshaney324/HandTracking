import cv2
import mediapipe as mp
import time
import numpy as np
import csv

number_of_classes = 3
class_labels = np.eye(number_of_classes)


################# COLLECT 1 DATA ##############################

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
start_time = time.time()
one_data = []

while cap.isOpened() and time.time() - start_time < 1:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand = []
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                hand.append([x, y, z])
            # print(hand_landmarks)
            # print(hand)
            one_data.append(hand)
            # print("----------------------------------------")

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

one_data = np.array(one_data)
one_data_flat = one_data.reshape(len(one_data), -1)
labels = np.tile(class_labels[0], (len(one_data_flat), 1))
one_data = np.concatenate((one_data_flat, labels), axis=1)

################# COLLECT 2 DATA ##############################

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
start_time = time.time()
two_data = []

while cap.isOpened() and time.time() - start_time < 1:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand = []
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                hand.append([x, y, z])
            # print(hand_landmarks)
            # print(hand)
            two_data.append(hand)
            # print("----------------------------------------")

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

two_data = np.array(two_data)
two_data_flat = two_data.reshape(len(two_data), -1)
labels = np.tile(class_labels[1], (len(two_data_flat), 1))
two_data = np.concatenate((two_data_flat, labels), axis=1)

################# COLLECT 3 DATA ##############################

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
start_time = time.time()
three_data = []

while cap.isOpened() and time.time() - start_time < 1:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand = []
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                hand.append([x, y, z])
            # print(hand_landmarks)
            # print(hand)
            three_data.append(hand)
            # print("----------------------------------------")

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

three_data = np.array(three_data)
three_data_flat = three_data.reshape(len(three_data), -1)
labels = np.tile(class_labels[1], (len(three_data_flat), 1))
three_data = np.concatenate((two_data_flat, labels), axis=1)

######################## Save to CSV ######################################

print(one_data.shape)
print(two_data.shape)
print(three_data.shape)

csv_file = "../data/data.csv"
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    # Write data
    writer.writerows(one_data)
    writer.writerows(two_data)
    writer.writerows(three_data)

print(f"Data saved to {csv_file}!")

