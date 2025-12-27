from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("credit_fraud_model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API Running on Colab"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [data["Time"], data["Amount"]]
    for i in range(1, 29):
        features.append(data[f"V{i}"])

    features = np.array(features).reshape(1, -1)
    probability = model.predict_proba(features)[0][1]

    return jsonify({"fraud_probability": float(probability)})

if __name__ == "__main__":
    app.run()
