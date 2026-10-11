import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
path = r"C:\Users\gouth\Downloads\archive"
train = pd.read_csv(path + r"\train.csv")
test = pd.read_csv(path + r"\test.csv")

train = train.sample(n=min(10000, len(train)), random_state=42)
test = test.sample(n=min(2000, len(test)), random_state=42)
X_train = (
    train.iloc[:, 1].fillna("").astype(str) + " " +
    train.iloc[:, 2].fillna("").astype(str)
)
X_test = (
    test.iloc[:, 1].fillna("").astype(str) + " " +
    test.iloc[:, 2].fillna("").astype(str)
)

y_train = train.iloc[:, 0].astype(int).to_numpy() - 1
y_test = test.iloc[:, 0].astype(int).to_numpy() - 1
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train = pad_sequences(
    tokenizer.texts_to_sequences(X_train), maxlen=60
)
X_test = pad_sequences(
    tokenizer.texts_to_sequences(X_test), maxlen=60
)

model = Sequential([
    Embedding(5000, 32),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(4, activation="softmax")
])

model.build(input_shape=(None, 100))
model.summary()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

for metric in ["accuracy", "loss"]:
    plt.plot(history.history[metric], label="Training")
    plt.plot(history.history["val_" + metric], label="Validation")
    plt.title(metric.capitalize())
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Accuracy:", round(accuracy * 100, 2), "%")
predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=["World", "Sports", "Business", "Sci/Tech"],
    zero_division=0
))
