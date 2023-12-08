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

drop_th = 0.3

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(28 * 28, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(drop_th),
        layers.Dense(128, activation="relu"),
        layers.Dropout(drop_th),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

checkpoint = ModelCheckpoint(
    "best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-3)

tensorboard = TensorBoard(
    log_dir="./logs", histogram_freq=0, write_graph=True, write_images=False
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint, early_stopping, tensorboard],
)

best_model = tf.keras.models.load_model("best_model.keras")
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

best_model.save("final_model.keras")
