# Use an official, lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for some ML libraries and PyArrow)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and component definitions into the container
COPY src/ /app/src/
COPY components.py /app/

# Set the Python path so modules can be imported correctly
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Keep the container running in a way that KFP can inject commands
ENTRYPOINT ["python3"]