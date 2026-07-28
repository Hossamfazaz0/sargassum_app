# Works on both Hugging Face Spaces (Docker SDK, fixed port 7860) and
# Render (dynamic $PORT assigned at runtime) — CMD is shell-form so $PORT
# gets expanded; falls back to 7860 if $PORT isn't set (e.g. on HF Spaces).
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

# The model is downloaded at container startup from Hugging Face Hub
# (HF_MODEL_REPO / HF_MODEL_FILE env vars in app.py), so no local .h5
# needs to be committed here.

EXPOSE 7860

# Shell form (not exec/JSON-array form) so $PORT is expanded at runtime.
# Render sets $PORT itself; falls back to 7860 for local/HF Spaces runs.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}