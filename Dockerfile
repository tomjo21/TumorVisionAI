# Use an official Python runtime as a parent image
# 3.10 is a good balance for TensorFlow compatibility
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# libgl1-mesa-glx and libglib2.0-0 are often required for opencv/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches using gunicorn
# 4 workers is a reasonable starting point, bind to 0.0.0.0
# Run download_models.py then start app
CMD ["/bin/bash", "-c", "python download_models.py && gunicorn --bind 0.0.0.0:7860 --workers 2 --timeout 120 app:app"]
