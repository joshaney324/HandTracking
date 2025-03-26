import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

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

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")