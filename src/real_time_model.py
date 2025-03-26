import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp

data = np.loadtxt('../data/data.csv', delimiter=",")
np.random.shuffle(data)


datapoints = data[:, :-3]
labels = data[:, -3:]

loss_fn = 'categorical_crossentropy' if labels.shape[1] > 1 else 'sparse_categorical_crossentropy'

X_train, X_test, y_train, y_test = train_test_split(datapoints, labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(datapoints.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

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
