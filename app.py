"""
Sargassum Morphotype Classifier — FastAPI backend
----------------------------------------------------
Serves the custom static/index.html frontend and exposes POST /predict,
matching what that frontend's fetch('/predict', ...) call expects:

    Request:  multipart/form-data, field name "file" (an image)
    Response: {
        "prediction": "<display name>",
        "confidence": <float 0-1>,
        "probabilities": {"SNN": <float>, "SFF": <float>, "SNW": <float>}
    }

Inference matches the training script: preprocessing baked into the model
(Lambda layer), TTA (7 augmented crops + 1 center resize, averaged),
temperature scaling (T fit during calibration).
"""

import os
import time
import numpy as np
import cv2
import albumentations as A
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from huggingface_hub import hf_hub_download, list_repo_files
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efn_preprocess

# -----------------------------
# CONFIG — matches training script
# -----------------------------
HF_REPO_ID = "Hossamfazaz/sargassum_enhanced_version"
IMG_SIZE = 300
CLASS_NAMES = ["SN1", "SF3", "SN8"]  # internal order — MUST match training's CLASS_TO_INDEX
CONFIDENCE_THRESHOLD = 0.75
TEMPERATURE = 0.852
N_TTA = 7

DISPLAY_CODE = {"SN1": "SNN", "SF3": "SFF", "SN8": "SNW"}
DISPLAY_FULL = {
    "SN1": "Sargassum natans natans",
    "SF3": "Sargassum fluitans fluitans",
    "SN8": "Sargassum natans wingei",
}


def display_name(internal_cls):
    return f"{DISPLAY_FULL[internal_cls]} ({DISPLAY_CODE[internal_cls]})"


# -----------------------------
# Load model from Hugging Face Hub at startup
# -----------------------------
print(f"Locating model file in {HF_REPO_ID} ...")
repo_files = list_repo_files(HF_REPO_ID)
h5_files = [f for f in repo_files if f.endswith(".h5")]
if not h5_files:
    raise RuntimeError(f"No .h5 file found in {HF_REPO_ID}. Files present: {repo_files}")
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=h5_files[0])

print("Loading model...")
model = load_model(model_path, custom_objects={"efn_preprocess": efn_preprocess}, compile=False)
print("Model loaded.")

tta_aug = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.88, 1.0), ratio=(0.95, 1.05), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.08, 0.08, p=0.3),
])


def temperature_scale(probs, temperature=TEMPERATURE):
    log_p = np.log(probs + 1e-8)
    scaled = np.exp(log_p / temperature)
    return scaled / scaled.sum()


# -----------------------------
# Simple in-memory rate limiting (per-session, fine for a pilot)
# -----------------------------
request_log = {}
RATE_LIMIT_PER_MIN = 10


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    window = request_log.setdefault(ip, [])
    window[:] = [t for t in window if now - t < 60]
    if len(window) >= RATE_LIMIT_PER_MIN:
        return False
    window.append(now)
    return True


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Sargassum Morphotype Classifier")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Limite de requêtes atteinte — réessayez dans une minute.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        raw_bytes = await file.read()
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        src = np.array(image)

        imgs = [tta_aug(image=src)["image"].astype(np.float32) for _ in range(N_TTA)]
        imgs.append(cv2.resize(src, (IMG_SIZE, IMG_SIZE)).astype(np.float32))
        batch = np.array(imgs)

        raw_probs = model.predict(batch, verbose=0).mean(axis=0)
        probs = temperature_scale(raw_probs)

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        top_display = display_name(CLASS_NAMES[top_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            prediction_label = f"⚠️ Vérification nécessaire — {top_display}"
        else:
            prediction_label = f"✅ {top_display}"

        probabilities = {
            DISPLAY_CODE[CLASS_NAMES[i]]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de l'image : {e}")


# Mount static AFTER routes so "/" and "/predict" above take priority
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))