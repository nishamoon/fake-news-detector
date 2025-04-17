from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("../models/saved_model.h5")
with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 500
THRESHOLD = 0.5  # You can tweak this value if needed

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Limit input length to 2000 characters
        news_text = request.form['news_text'][:2000]

        # Preprocess text
        seq = tokenizer.texts_to_sequences([news_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        # Make prediction
        prediction = model.predict(padded)[0][0]
        confidence = round(float(prediction) * 100, 2)

        # Determine label and confidence display
        if prediction >= THRESHOLD:
            label = "Real News 📰"
            confidence_display = confidence
        else:
            label = "Fake News ⚠️"
            confidence_display = 100 - confidence

        return render_template("result.html", prediction=label, confidence=confidence_display)

if __name__ == '__main__':
    app.run(debug=True)
