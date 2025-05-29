import time
from src.base_hand_tracking.model import complete_model
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('TkAgg')


data = np.loadtxt('../../data/alphabet_data.csv', delimiter=",", dtype=object)

np.random.shuffle(data)

datapoints = data[:, :-1].astype(float)
labels = data[:, -1].astype(str)
unique_labels = np.unique(labels)
num_classes = len(np.unique(labels))
eye = np.eye(num_classes)
one_hot_dict = {}

for i, label in enumerate(np.unique(labels)):
    one_hot_dict[label] = eye[i]

one_hot_labels = []
for label in labels:
    one_hot_labels.append(one_hot_dict[label])

one_hot_labels = np.array(one_hot_labels)

X_train, X_test, y_train, y_test = train_test_split(datapoints, one_hot_labels, test_size=0.2, random_state=42)

model = complete_model(X_train, y_train, X_test, y_test, 10)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

typed_text = ""
last_predicted = None
last_time = 0
hold_threshold = 3.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    h, w, _ = frame.shape
    progress = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_list = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_list = [landmark.y * h for landmark in hand_landmarks.landmark]
            x_min, x_max = int(min(x_list)), int(max(x_list))
            y_min, y_max = int(min(y_list)), int(max(y_list))

            hand = []
            for lm in hand_landmarks.landmark:
                hand.extend([lm.x, lm.y, lm.z])
            hand = np.array(hand).reshape(1, -1)

            prediction = model.predict(hand)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            predicted_letter = unique_labels[predicted_class]
            current_time = time.time()

            if predicted_letter == last_predicted:
                time_held = current_time - last_time
                progress = min(time_held / hold_threshold, 1.0)
                if time_held >= hold_threshold:
                    if predicted_letter == "space":
                        typed_text += " "
                    elif predicted_letter == "delete":
                        typed_text = typed_text[:-1]
                    else:
                        typed_text += predicted_letter

                    last_time = current_time + 2
                    last_predicted = None
            else:
                last_predicted = predicted_letter
                last_time = current_time
                progress = 0

            cv2.rectangle(frame, (x_min - 30, y_min - 30), (x_max + 10, y_max + 10), (0, 255, 0), 2)

            label_text = f"Letter: {unique_labels[predicted_class]} ({confidence:.1f}%)"
            cv2.putText(frame, label_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.rectangle(frame, (10, h - 60), (w - 10, h - 10), (50, 50, 50), -1)
    cv2.putText(frame, f"Text: {typed_text}", (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    bar_x, bar_y = 10, h - 70
    bar_width, bar_height = w - 20, 10
    filled_width = int(bar_width * progress)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (150, 150, 150), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()