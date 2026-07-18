# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860
FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# IMPORTANT: place your trained model file here before building, e.g.:
#   sargassum-app/sargassum_model_beach1.h5
# (not committed here since it's a large binary you already have on Drive)

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
