import pandas as pd
import pickle

# Load the encoder from the encoders.pkl file
with open('encoders.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Define a helper function to safely preprocess input data
def preprocess_input(input_data):
    try:
        # Validate and encode 'gender' field
        if input_data['gender'] not in ['Male', 'Female']:
            raise ValueError("Invalid gender value. Expected 'Male' or 'Female'.")
        
        input_data['gender_encoded'] = encoder['gender'].transform([input_data['gender']])[0]  # Assuming encoder for gender

        # Validate and encode 'ethnicity' field (ensure input matches one of the known categories)
        valid_ethnicities = ['White-European', 'Asian', 'Black', 'Hispanic']  # Update with actual valid categories
        if input_data['ethnicity'] not in valid_ethnicities:
            raise ValueError(f"Invalid ethnicity value. Expected one of {valid_ethnicities}.")
        
        input_data['ethnicity_encoded'] = encoder['ethnicity'].transform([input_data['ethnicity']])[0]  # Assuming encoder for ethnicity

        # Ensure all other numeric fields are integers
        input_data['A1_Score'] = int(input_data['A1_Score'])
        input_data['A2_Score'] = int(input_data['A2_Score'])
        input_data['A3_Score'] = int(input_data['A3_Score'])
        input_data['A4_Score'] = int(input_data['A4_Score'])
        input_data['A5_Score'] = int(input_data['A5_Score'])
        input_data['A6_Score'] = int(input_data['A6_Score'])
        input_data['A7_Score'] = int(input_data['A7_Score'])
        input_data['A8_Score'] = int(input_data['A8_Score'])
        input_data['A9_Score'] = int(input_data['A9_Score'])

    except ValueError as e:
        print(f"Error during encoding: {e}")
        raise ValueError(f"Error during encoding: {e}")

    return input_data
