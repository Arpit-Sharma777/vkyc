# Use Python Slim
FROM python:3.9-slim

# 1. Install System Dependencies for OpenCV (Crucial!)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Python Dependencies
COPY requirements.txt .
# Use --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose Port 5001
EXPOSE 5001

# Run the AI Service (Disable Debug mode for Docker)
CMD ["python", "app.py"]