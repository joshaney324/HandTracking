import csv
import time
import mediapipe as mp
import numpy as np
import cv2
import os


def decode_one_hot(one_hot_vector, labels):
    index = np.argmax(one_hot_vector)
    return labels[index]


def collect_data(csv_title, labels):
    num_classes = len(labels)

    # Create one-hot encoding dictionary
    one_hot_dict = {
        label: np.eye(num_classes)[i].tolist()
        for i, label in enumerate(labels)
    }

    csv_file = csv_title

    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)

        for i, label in enumerate(labels):
            print(f"Collecting class '{label}' ({i + 1}/{num_classes})")
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands()
            mp_draw = mp.solutions.drawing_utils

            cap = cv2.VideoCapture(0)
            start_time = time.time()
            hand_data = []

            while cap.isOpened() and time.time() - start_time < 20:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(frame_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        hand = []
                        for landmark in hand_landmarks.landmark:
                            x, y, z = landmark.x, landmark.y, landmark.z
                            hand.extend([x, y, z])
                        hand_data.append(hand)

                cv2.putText(frame, f"Label: {label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                remaining = int(20 - (time.time() - start_time))
                cv2.putText(frame, f"Time Left: {remaining}s", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("Hand Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            hand_data = np.array(hand_data)

            labeled_data = [row.tolist() + [labels[i]] for row in hand_data]

            writer.writerows(labeled_data)

    print(f"\nData saved to {csv_file}")
    return one_hot_dict
