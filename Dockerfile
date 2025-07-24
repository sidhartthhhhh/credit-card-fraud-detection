# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# --- ADD THIS LINE ---
# Install system dependency required by lightgbm
RUN apt-get update && apt-get install -y libgomp1

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run your application
CMD ["python", "src/app.py"]