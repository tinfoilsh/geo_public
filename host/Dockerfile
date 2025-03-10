FROM python:3.12-slim

# Install system dependencies required for NVIDIA SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the host directory contents
COPY . ./host/

# Environment variables
ENV FLASK_APP=host/host.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-m", "host.wsgi"]