import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("kc_house_data.csv")

scaler = MinMaxScaler()
X = scaler.fit_transform(data[["sqft_living"]])
y = scaler.fit_transform(data[["price"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(
    Dense(1, input_dim=1, activation="linear", kernel_regularizer=l2(0.01))
)  # 1 hidden layer with L2 regularization


def custom_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return 10 * mse_loss


model.compile(optimizer="adam", loss=custom_loss)

history = model.fit(
    X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
)

test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

model.save("regression_model.keras")
