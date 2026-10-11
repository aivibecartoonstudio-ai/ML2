import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

dataset_path = r"C:\Users\laya.n_cmrit\Downloads\cifar-10-python (4)\cifar-10-batches-py"

with open(os.path.join(dataset_path, "batches.meta"), "rb") as f:
    meta = pickle.load(f, encoding="bytes")

class_names = [x.decode("utf-8") for x in meta[b"label_names"]]

print("Classes:")
print(class_names)


def load_batch(file_name):

    with open(os.path.join(dataset_path, file_name), "rb") as f:
        data = pickle.load(f, encoding="bytes")

    images = data[b"data"]
    labels = np.array(data[b"labels"])

    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, labels


X_train = []
y_train = []

for i in range(1, 6):

    images, labels = load_batch("data_batch_" + str(i))

    X_train.append(images)
    y_train.append(labels)

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

print("Training images:", X_train.shape)

X_test, y_test = load_batch("test_batch")

print("Testing images:", X_test.shape)


X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]

X_train = tf.image.resize(X_train, (96, 96))
X_test = tf.image.resize(X_test, (96, 96))

X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(96, 96, 3),
    alpha=0.35
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(96, 96, 3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)

outputs = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs, outputs)


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1
)


loss, accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=16
)

print("\nTest Accuracy:", accuracy)

predictions = model.predict(X_test[:10])

predicted_classes = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_test[i].numpy().astype("uint8"))

    actual = class_names[y_test[i]]
    predicted = class_names[predicted_classes[i]]

    plt.title(
        "Actual: " + actual +
        "\nPredicted: " + predicted
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
