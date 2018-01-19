# Refer to Docker documentation
# https://docs.docker.com/get-started/

# Use an official Python runtime as parent image
FROM python:3.6-slim

# Set working directory to /app
WORKDIR /usr/src/deflux

# Install supervisord
RUN apt-get update && apt-get install -y supervisor

# Copy requirements and install to enable caching on rebuilds
COPY requirements.txt /usr/src/deflux

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /usr/src/deflux

# Define environment variables
ENV FLASK_APP=deflux.py
ENV FLASK_DEBUG=1
ENV PYTHONUNBUFFERED=0

# Run supervisord
CMD ["/usr/bin/supervisord"]
