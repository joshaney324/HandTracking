import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, output_shape):
    def model_builder(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_shape,)))

        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(layers.Dense(units=hp_units, activation='relu'))

        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(layers.Dense(units=hp.Int(f'units_{i}', 32, 512, step=32), activation='relu'))

        model.add(layers.Dense(output_shape, activation='softmax'))  # Output layer

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    return model_builder


def hyperparameter_search(X_train, y_train, X_test, y_test, max_trials):
    tuner = kt.RandomSearch(
        build_model(len(X_train[0]), len(y_train[0])),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='tuning_dir',
        project_name='dense_model_tuning'
    )

    tuner.search(X_train, y_train, epochs=32, validation_data=(X_test, y_test), verbose=1)

    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model


def complete_model(X_train, y_train, X_test, y_test, max_trials):
    model = hyperparameter_search(X_train, y_train, X_test, y_test, max_trials)
    return model
