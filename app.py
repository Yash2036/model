from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess categorical data
def preprocess_input(input_data):
    # Encoding categorical features: Gender and Ethnicity
    categorical_features = ['gender', 'ethnicity']
    
    # Example categories based on the dataset you provided
    # This should be consistent with how your model was trained
    categories = {
        'gender': ['Male', 'Female'],
        'ethnicity': ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other', 'White-European']
    }
    
    # Convert categorical data to numerical using LabelEncoder
    for feature in categorical_features:
        encoder = LabelEncoder()
        encoder.fit(categories[feature])
        input_data[feature] = encoder.transform([input_data[feature]])[0]
    
    return input_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect the form data
    try:
        input_data = {
            'A1_Score': int(request.form['A1_Score']),
            'A2_Score': int(request.form['A2_Score']),
            'A3_Score': int(request.form['A3_Score']),
            'A4_Score': int(request.form['A4_Score']),
            'A5_Score': int(request.form['A5_Score']),
            'A6_Score': int(request.form['A6_Score']),
            'A7_Score': int(request.form['A7_Score']),
            'A8_Score': int(request.form['A8_Score']),
            'A9_Score': int(request.form['A9_Score']),
            'gender': request.form['gender'],
            'ethnicity': request.form['ethnicity']
        }
    except KeyError as e:
        return f"Missing input data: {str(e)}", 400

    # Preprocess the input data (convert categorical to numerical)
    input_data = preprocess_input(input_data)

    # Prepare data for prediction
    input_features = [input_data[key] for key in input_data]

    # Perform the prediction
    prediction = model.predict([input_features])[0]
    
    # Return the result
    result = 'Autism' if prediction == 1 else 'No Autism'
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    # Bind to 0.0.0.0 for deployment so it's not restricted to localhost
    app.run(host='0.0.0.0', port=5000)
