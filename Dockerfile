# Use Python base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask application
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
