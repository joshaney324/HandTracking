from model import complete_model
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp

data = np.loadtxt('../data/data.csv', delimiter=",")
np.random.shuffle(data)

datapoints = data[:, :-5]
labels = data[:, -5:]

X_train, X_test, y_train, y_test = train_test_split(datapoints, labels, test_size=0.2, random_state=42)

model = complete_model(X_train, y_train, X_test, y_test, 3)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
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
            hand = np.array(hand)
            hand = hand.reshape(1, -1)
            prediction = model.predict(hand)
            predicted_class = np.argmax(prediction) + 1
            print(predicted_class)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
