from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

app = Flask(__name__)

# -------------------------
# Attention Layer
# -------------------------
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal")
        self.b = self.add_weight(shape=(input_shape[1], 1), initializer="zeros")

    def call(self, inputs):
        score = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        context = inputs * weights
        context = tf.reduce_sum(context, axis=1)
        return context, weights

# -------------------------
# Load Model
# -------------------------
model = load_model("model.h5", custom_objects={"Attention": Attention}, compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1]

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form["text"]

        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')

        pred, _ = model.predict(padded)
        pred = float(pred[0][0])

        if pred > 0.5:
            label = "Positive"
            confidence = pred
        else:
            label = "Negative"
            confidence = 1 - pred

        result = f"{label} ({confidence*100:.1f}%)"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)