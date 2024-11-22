from flask import Flask, request, render_template
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')  # Load the encoder

# Home route to render the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect the form data
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

        # Preprocess input_data using the encoder
        input_data = preprocess_input(input_data)

        # Collect all the numerical features (including encoded gender and ethnicity)
        features = [
            input_data['A1_Score'], input_data['A2_Score'], input_data['A3_Score'], 
            input_data['A4_Score'], input_data['A5_Score'], input_data['A6_Score'],
            input_data['A7_Score'], input_data['A8_Score'], input_data['A9_Score'],
            input_data['gender_encoded'], input_data['ethnicity_encoded']
        ]

        # Predict using the model
        prediction = model.predict([features])[0]

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        # Log the error and return a user-friendly message
        print(f"Error: {e}")
        return "An error occurred while processing your request. Please try again."

# Function to preprocess input data using the encoder
def preprocess_input(input_data):
    # Example of data preprocessing using the encoder
    try:
        # Prepare categorical data for encoding
        categorical_data = [[input_data['gender'], input_data['ethnicity']]]

        # Use the encoder to transform categorical data
        encoded_data = encoder.transform(categorical_data)

        # Replace the original categorical values with the encoded values
        input_data['gender_encoded'] = encoded_data[0][0]
        input_data['ethnicity_encoded'] = encoded_data[0][1]

        # Remove original gender and ethnicity fields
        del input_data['gender']
        del input_data['ethnicity']

    except Exception as e:
        print(f"Encoding Error: {e}")
        raise ValueError("Error during encoding. Please ensure inputs are correct.")

    return input_data

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
