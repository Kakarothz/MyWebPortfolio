"""Small TensorFlow/Keras classifier using the built-in MNIST dataset."""
from pathlib import Path
import json
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
out = Path(__file__).parent / "artifacts"
out.mkdir(exist_ok=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
model.save(out / "mnist.keras")
(out / "metrics.json").write_text(json.dumps({"test_loss": loss, "test_accuracy": accuracy}, indent=2))
print({"test_loss": loss, "test_accuracy": accuracy})
