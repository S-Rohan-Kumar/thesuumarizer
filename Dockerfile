# Use the official Python slim image for a smaller footprint
FROM python:3.13

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-tam \
    tesseract-ocr-tel \
    tesseract-ocr-kan \
    pkg-config \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (create this separately)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TESSERACT_PATH=/usr/bin/tesseract
ENV RENDER_DISK_PATH=/app
ENV TEMP_DIR=/app/temp
ENV PORT=7000

RUN tesseract --version && ffmpeg -version
# Create temp directory
RUN mkdir -p /app/temp && chmod 777 /app/temp
RUN mkdir -p templates static
# Expose the port Render expects (Render uses 10000 by default, but we'll use 7000 as per your app)
EXPOSE 7000

# Command to run the application with Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --timeout 120 app1:app"]
