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
        
        # Debugging: Print input data to logs
        print("Received Input Data:", input_data)
        
        # Validate the scores
        validate_scores(input_data)
        
        # Preprocess the input data
        input_data = preprocess_input(input_data)
        print("Preprocessed Input Data:", input_data)

        # Prepare data for prediction
        input_features = [input_data[key] for key in input_data]
        
        # Perform the prediction
        prediction = model.predict([input_features])[0]
        
        # Debugging: Print prediction result to logs
        print("Prediction Result:", prediction)
        
    except (KeyError, ValueError) as e:
        print(f"Error processing input data: {str(e)}")  # Log the error
        return f"Invalid input data: {str(e)}", 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Log unexpected errors
        return "Internal Server Error", 500

    # Return the result
    result = 'Autism' if prediction == 1 else 'No Autism'
    return render_template('result.html', prediction=result)
