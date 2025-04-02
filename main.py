from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("sentiment_model_98_2.pkl")
vectorizer = joblib.load("vectorizer_98_2.pkl")

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['Review']  # Get user input from form
        
        # Transform input using vectorizer
        review_vectorized = vectorizer.transform([review])
        prediction = model.predict(review_vectorized)[0]
        
        # Convert prediction to label
        if prediction == 2:
            sentiment = 'Positive'
        elif prediction == 1:
            sentiment = 'Neutral'
        else:
            sentiment = 'Negative'

        #sentiment = "Positive" if prediction == 1 else "Negative"
        
        return render_template('index.html', prediction_text=f"Sentiment: {sentiment}")

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)



"""
from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('sentiment_model_98.pkl')
vectorizer = joblib.load('vectorizer_98.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Getting JSON data from request
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_vectorized = vectorizer.transform([text])  # Transform text
    prediction = model.predict(text_vectorized)[0]  # Predict sentiment

    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"sentiment": sentiment})



if __name__ == '__main__':
    app.run(debug=True)

"""

"""
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return "Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from request
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_vectorized = vectorizer.transform([text])  # Transform text
    prediction = model.predict(text_vectorized)[0]  # Predict sentiment

    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

"""