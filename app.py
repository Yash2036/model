from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = [
        int(request.form['A1_Score']),
        int(request.form['A2_Score']),
        int(request.form['A3_Score']),
        int(request.form['A4_Score']),
        int(request.form['A5_Score']),
        int(request.form['A6_Score']),
        int(request.form['A7_Score']),
        int(request.form['A8_Score']),
        int(request.form['A9_Score']),
        int(request.form['A10_Score']),
        int(request.form['age']),
        1 if request.form['gender'] == 'm' else 0,  # Gender encoding
        request.form['ethnicity'],
        1 if request.form['jaundice'] == 'yes' else 0,
        1 if request.form['austim'] == 'yes' else 0
    ]

    # Prediction logic (adjust according to the model's input expectations)
    prediction = model.predict([input_data])[0]
    
    # Render the result back to the user
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
