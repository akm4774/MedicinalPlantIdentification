# Use a Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt file to the container
COPY requirements.txt requirements.txt

# Install the dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]
