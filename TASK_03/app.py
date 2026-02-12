from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    attendance = float(request.form['attendance'])
    previous_marks = float(request.form['previous_marks'])

    features = np.array([[hours, attendance, previous_marks]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    result = "Pass" if prediction == 1 else "Fail"

    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
