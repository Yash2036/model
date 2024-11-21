from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get features from the form
    try:
        features = [float(x) for x in request.form.values()]
        features = np.array([features])
        prediction = model.predict(features)[0]
        return render_template("index.html", prediction_text=f"Prediction: {prediction}")
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
