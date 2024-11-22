import os
import pickle
import numpy as np
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the encoders for gender and ethnicity
with open('encoders.pkl', 'rb') as encoder_file:
    encoders = pickle.load(encoder_file)

# Function to preprocess input data
def preprocess_input(input_data):
    try:
        # Validate and encode 'gender' field
        if input_data['gender'] not in ['Male', 'Female']:
            raise ValueError("Invalid gender value. Expected 'Male' or 'Female'.")
        
        input_data['gender_encoded'] = encoders['gender'].transform([input_data['gender']])[0]

        # Validate and encode 'ethnicity' field
        valid_ethnicities = encoders['ethnicity'].classes_.tolist()
        if input_data['ethnicity'] not in valid_ethnicities:
            raise ValueError(f"Invalid ethnicity value. Expected one of {valid_ethnicities}.")

        input_data['ethnicity_encoded'] = encoders['ethnicity'].transform([input_data['ethnicity']])[0]

        # Ensure all other numeric fields are integers (0 or 1)
        for key in [f'A{i}_Score' for i in range(1, 10)]:
            input_data[key] = int(input_data[key])
            if input_data[key] not in [0, 1]:
                raise ValueError(f"Invalid score value for {key}. Expected 0 or 1.")
        
        # Construct input for prediction
        processed_data = [
            input_data['gender_encoded'],
            input_data['ethnicity_encoded'],
            input_data['A1_Score'],
            input_data['A2_Score'],
            input_data['A3_Score'],
            input_data['A4_Score'],
            input_data['A5_Score'],
            input_data['A6_Score'],
            input_data['A7_Score'],
            input_data['A8_Score'],
            input_data['A9_Score']
        ]

    except ValueError as e:
        print(f"Error during encoding: {e}")
        raise ValueError(f"Error during encoding: {e}")

    return np.array(processed_data).reshape(1, -1)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        input_data = {
            'gender': request.form.get('gender', None),
            'ethnicity': request.form.get('ethnicity', None),
            'A1_Score': request.form.get('A1_Score', None),
            'A2_Score': request.form.get('A2_Score', None),
            'A3_Score': request.form.get('A3_Score', None),
            'A4_Score': request.form.get('A4_Score', None),
            'A5_Score': request.form.get('A5_Score', None),
            'A6_Score': request.form.get('A6_Score', None),
            'A7_Score': request.form.get('A7_Score', None),
            'A8_Score': request.form.get('A8_Score', None),
            'A9_Score': request.form.get('A9_Score', None)
        }

        # Validate that all fields are present
        if None in input_data.values():
            raise ValueError("All input fields are required.")

        # Preprocess the input
        processed_input = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(processed_input)[0]
        prediction_text = "Positive for Autism" if prediction == 1 else "Negative for Autism"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        # Handle any exceptions and display an error message
        error_message = f"An error occurred while processing your request: {e}"
        print(error_message)  # This will print the error to the console for debugging
        return render_template('index.html', prediction_text=error_message)

# Run the app
if __name__ == "__main__":
    # Get the port number from the environment variable (default to 5000 if not set)
    port = int(os.environ.get("PORT", 5000))
    # Bind the app to '0.0.0.0' to allow external access
    app.run(host='0.0.0.0', port=port)
