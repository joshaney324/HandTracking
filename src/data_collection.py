import cv2
import mediapipe as mp
import time
import numpy as np
import csv

number_of_classes = 5
class_labels = np.eye(number_of_classes)



csv_file = "../data/data.csv"
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    for i in range(number_of_classes):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        mp_draw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)
        start_time = time.time()
        hand_data = []

        while cap.isOpened() and time.time() - start_time < 10:
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
                    hand_data.append(hand)
                    # print("----------------------------------------")

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        hand_data = np.array(hand_data)
        hand_data_flat = hand_data.reshape(len(hand_data), -1)
        labels = np.tile(class_labels[i], (len(hand_data_flat), 1))
        hand_data = np.concatenate((hand_data_flat, labels), axis=1)

        # Write data
        writer.writerows(hand_data)

print(f"Data saved to {csv_file}!")

