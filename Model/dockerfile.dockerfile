# Use Python base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy all project files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
