import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

file_path = r"C:\Users\vinuta.s_cmrit\Downloads\jena_climate_2009_2016.csv\jena_climate_2009_2016.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Shape of dataset:", df.shape)

print("\nFirst five records:")
print(df.head())

temperature = df["T (degC)"].values

print("\nNumber of temperature observations:", len(temperature))
print("\nMissing values:")
print(df["T (degC)"].isnull().sum())


plt.figure(figsize=(12, 5))

plt.plot(temperature[:2000])

plt.title("Temperature Time Series")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")

plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))

temperature_scaled = scaler.fit_transform(
    temperature.reshape(-1, 1)
)

print("Normalization completed")
time_steps = 24

X = []
y = []

for i in range(time_steps, len(temperature_scaled)):

    X.append(
        temperature_scaled[i-time_steps:i, 0]
    )

    y.append(
        temperature_scaled[i, 0]
    )
X = np.array(X)
y = np.array(y)

print("\nSequence creation completed")

print("X shape before reshaping:", X.shape)
print("y shape:", y.shape)

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

print("X shape after reshaping:", X.shape)

train_size = int(len(X) * 0.80)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(time_steps, 1)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(50)
)

model.add(Dropout(0.2))

model.add(Dense(1))
model.summary()

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
plt.figure(figsize=(10, 5))
plt.plot(
    history.history["loss"],
    label="Training Loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

predicted = model.predict(X_test)
predicted_temperature = scaler.inverse_transform(
    predicted
)

actual_temperature = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

rmse = np.sqrt(
    mean_squared_error(
        actual_temperature,
        predicted_temperature
    )
)

mae = mean_absolute_error(
    actual_temperature,
    predicted_temperature
)

print("\nModel Evaluation")

print("RMSE:", rmse)

print("MAE:", mae)

plt.figure(figsize=(12, 6))

plt.plot(
    actual_temperature[:500],
    label="Actual Temperature"
)

plt.plot(
    predicted_temperature[:500],
    label="Predicted Temperature"
)

plt.title(
    "Actual vs Predicted Temperature"
)

plt.xlabel("Time")

plt.ylabel("Temperature (°C)")

plt.legend()

plt.show()
