FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install git and other dependencies
RUN apt-get update -y && \
    apt-get install -y git awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install dagshub
RUN pip install --no-cache-dir dagshub

# Command to run the application
CMD ["python", "app.py"]
