import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def build_model(dropout):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(28 * 28, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


dropout_values = [0, 0.3, 0.9]
dropout_accuracy = []

for dropout in dropout_values:
    model = build_model(dropout)

    checkpoint = ModelCheckpoint(
        f"best_model_dropout_{dropout}.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-3)
    tensorboard = TensorBoard(
        log_dir=f"./logs_dropout_{dropout}",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )

    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping, tensorboard],
    )

    best_model = tf.keras.models.load_model(f"best_model_dropout_{dropout}.keras")
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    print(f"Test Accuracy for Dropout {dropout}: {test_accuracy}")

    best_model.save(f"final_model_dropout_{dropout}.keras")
    dropout_accuracy.append(test_accuracy)

best_dropout = dropout_values[np.argmax(dropout_accuracy)]
print(
    f"Best Dropout: {best_dropout} with accuracy {dropout_accuracy[np.argmax(dropout_accuracy)]}"
)
