from model import complete_model
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system
import matplotlib.pyplot as plt

data = np.loadtxt('../../data/number_data.csv', delimiter=",")
np.random.shuffle(data)

datapoints = data[:, :-1]
labels = data[:, -1]
# scaler = StandardScaler()
# scaled_X = scaler.fit_transform(datapoints)
# pca = PCA(n_components=2)
# pca.fit(scaled_X)
# X_pca = pca.transform(scaled_X)
#
# explained_variance = pca.explained_variance_ratio_
# print("Explained variance ratio:", explained_variance)
#
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1])
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA Result')
# plt.grid(True)
# plt.show()

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            h, w, _ = frame.shape
            x_list = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_list = [landmark.y * h for landmark in hand_landmarks.landmark]
            x_min, x_max = int(min(x_list)), int(max(x_list))
            y_min, y_max = int(min(y_list)), int(max(y_list))

            hand = []
            for lm in hand_landmarks.landmark:
                hand.extend([lm.x, lm.y, lm.z])
            hand = np.array(hand).reshape(1, -1)

            prediction = model.predict(hand)
            print(prediction)
            predicted_class = np.argmax(prediction) + 1
            confidence = np.max(prediction) * 100

            cv2.rectangle(frame, (x_min - 30, y_min - 30), (x_max + 10, y_max + 10), (0, 255, 0), 2)

            label_text = f"Number of Fingers: {predicted_class} ({confidence:.1f}%)"
            cv2.putText(frame, label_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()