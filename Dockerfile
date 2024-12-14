FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install AWS CLI
RUN apt-get update -y && apt-get install -y awscli

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DagsHub
RUN pip install --no-cache-dir dagshub

# Set environment variable for DagsHub token
ENV DAGSHUB_TOKEN=28e8ded312555c788f14d7fd4083a1ca490f2c68

# Log in to DagsHub (this will use the token set above)
RUN dagshub login --token $DAGSHUB_TOKEN

# Command to run the application
CMD ["python3", "app.py"]
