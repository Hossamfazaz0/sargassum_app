# ================================================================
# 🌐 SARGASSUM CLASSIFIER — FASTAPI SERVING APP
# ================================================================
# Serves the fine-tuned EfficientNetB3 beach model (sargassum_model_beach1.h5)
# behind a REST API + a simple upload page, for deployment on Hugging Face
# Spaces (Docker SDK) or any Python host (Render, Railway, etc).
#
# Mirrors EXACTLY the loading + preprocessing logic from the Colab
# single-image tester, so predictions match what you validated locally:
#   - mixed_float16 policy set BEFORE loading (weight layout must match)
#   - two-stage load: direct load_model() -> fallback rebuild + load_weights
#   - preprocessing: BGR->RGB, resize to (300,300), float32
#     (NO manual normalization — efn_preprocess is baked into the model
#     via the Lambda layer, so we must NOT double-preprocess here)
# ================================================================

import os
import io
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, mixed_precision
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB3, preprocess_input as efn_preprocess)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ================================================================
# ⚙️ SETTINGS — must match training/testing exactly
# ================================================================
MODEL_PATH  = os.environ.get("MODEL_PATH", "sargassum_model_beach1.h5")
CLASS_NAMES = ['SN1', 'SF3', 'SN8']
IMG_SIZE    = 300
NUM_CLASSES = len(CLASS_NAMES)

# If the model isn't present locally (e.g. it's too large for the Git repo),
# download it from a Hugging Face Hub MODEL repo at container startup.
# Set these two env vars on Render if you go this route; leave unset if
# you're committing the .h5 directly (e.g. via Git LFS) instead.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")       # e.g. "your-username/sargassum-model"
HF_MODEL_FILE = os.environ.get("HF_MODEL_FILE", "sargassum_model_beach1.h5")


def ensure_model_downloaded():
    """Download the .h5 from Hugging Face Hub if it's not already present locally."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already present at {MODEL_PATH}")
        return
    if not HF_MODEL_REPO:
        return  # nothing to do; load_model_on_startup will raise a clear error
    print(f"⬇️  Downloading {HF_MODEL_FILE} from {HF_MODEL_REPO} ...")
    from huggingface_hub import hf_hub_download
    downloaded_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILE)
    # hf_hub_download caches the file elsewhere; symlink/copy it to MODEL_PATH
    import shutil
    shutil.copy(downloaded_path, MODEL_PATH)
    print(f"✅ Model downloaded to {MODEL_PATH}")

# Same mixed precision policy used during training — must be set BEFORE
# building/loading the model or weight shapes/dtypes can mismatch.
mixed_precision.set_global_policy('mixed_float16')

app = FastAPI(title="Sargassum Classifier API")

# Allow the frontend (served from anywhere) to call this API.
# Tighten allow_origins to your actual website domain once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None  # loaded on startup


def build_model():
    """Rebuild the exact training architecture (fallback if direct load fails)."""
    base = EfficientNetB3(weights=None, include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(efn_preprocess)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    return Model(inp, out)


@app.on_event("startup")
def load_model_on_startup():
    global model

    ensure_model_downloaded()

    print(f"📦 Loading model from {MODEL_PATH} ...")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Either commit it directly "
            "(e.g. via Git LFS) or set HF_MODEL_REPO / HF_MODEL_FILE env vars "
            "to download it from Hugging Face Hub at startup."
        )

    try:
        model = load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False,
            custom_objects={'preprocess_input': efn_preprocess},
        )
        print("✅ Model loaded (direct load)")
    except Exception as e1:
        print(f"   Direct load failed ({type(e1).__name__}); trying rebuild + load_weights...")
        try:
            model = build_model()
            model.load_weights(MODEL_PATH)
            print("✅ Model loaded (rebuild + load_weights)")
        except Exception as e2:
            raise RuntimeError(
                f"Both load methods failed.\nDirect: {e1}\nRebuild: {e2}"
            )

    # Warm up the model with a dummy forward pass so the first real
    # request isn't slowed down by graph tracing.
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("🔥 Model warmed up and ready")


def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """Same preprocessing as the Colab tester: BGR decode -> RGB -> resize -> float32."""
    data = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    return img_resized


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        img = preprocess_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    probs = model.predict(img[np.newaxis, ...], verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float32)
    pred_idx = int(np.argmax(probs))

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
        },
    }


# ---- Serve the simple upload frontend at "/" ----
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
