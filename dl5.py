import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from sklearn.metrics import classification_report, confusion_matrix

train_path = r"C:\Users\Student\Downloads\AG_News_Dataset_Local\AG_News\train.csv"
test_path = r"C:\Users\Student\Downloads\AG_News_Dataset_Local\AG_News\test.csv"

train_data = pd.read_csv(
    train_path,
    header=None,
    names=["label", "title", "description"]
)

test_data = pd.read_csv(
    test_path,
    header=None,
    names=["label", "title", "description"]
)

print("Training data:")
print(train_data.head())

print("\nTesting data:")
print(test_data.head())

X_train = (
    train_data["title"] + " " +
    train_data["description"]
)

X_test = (
    test_data["title"] + " " +
    test_data["description"]
)

y_train = train_data["label"]
y_test = test_data["label"]

y_train = y_train - 1
y_test = y_test - 1

vocab_size = 10000

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

print("\nShape of training data:", X_train_pad.shape)
print("Shape of testing data:", X_test_pad.shape)

num_classes = 4

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_length
    ),
    LSTM(128),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.build(input_shape=(None, max_length))

model.summary()

history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

loss, accuracy = model.evaluate(
    X_test_pad,
    y_test
)

print("\nTest Accuracy:", accuracy * 100, "%")

y_pred = model.predict(X_test_pad)

y_pred_labels = np.argmax(
    y_pred,
    axis=1
)

class_names = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
]

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred_labels,
        target_names=class_names
    )
)
