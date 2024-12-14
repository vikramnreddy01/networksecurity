FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install DagsHub
RUN pip install --no-cache-dir dagshub

# Log in to DagsHub using the provided token
RUN dagshub login --token 28e8ded312555c788f14d7fd4083a1ca490f2c68

# Install AWS CLI
RUN apt-get update -y && apt-get install -y awscli

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python3", "app.py"]
