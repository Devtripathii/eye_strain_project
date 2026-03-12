FROM python:3.11-slim

# System deps required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Models and user data are volume-mounted at runtime — not baked into image
VOLUME ["/app/models", "/app/data", "/app/outputs"]

EXPOSE 8501

# EYEGUARD_* env vars can override any config.py constant at runtime
ENTRYPOINT ["streamlit", "run", "apps/user_app.py", \
            "--server.address=0.0.0.0", \
            "--server.port=8501", \
            "--server.headless=true", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false"]