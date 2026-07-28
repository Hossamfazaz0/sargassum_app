# ================================================================
# 🌐 SARGASSUM CLASSIFIER — FASTAPI SERVING APP
# ================================================================
# Serves the fine-tuned EfficientNetB3 "lab-first" model
# (sargassum_model_lab_first.h5) behind a REST API + upload page,
# for deployment on Hugging Face Spaces (Docker SDK).
#
# Mirrors the Colab "lab-first reliable fine-tuning" pipeline:
#   - mixed_float16 policy set BEFORE loading (weight layout must match)
#   - two-stage load: direct load_model() -> fallback rebuild + load_weights
#   - preprocessing: BGR->RGB, resize to (300,300), float32
#     (NO manual normalization — efn_preprocess is baked into the model
#     via the Lambda layer, so we must NOT double-preprocess here)
#   - TTA (7 augmented crops + 1 clean resize) is the DEFAULT inference
#     mode, matching predict_with_tta() / predict_reliable() in training
#   - Temperature scaling applied post-softmax (T fit on validation set
#     during training: T = 0.852)
#   - Safety gate: predictions below CONFIDENCE_THRESHOLD are flagged
#     "review needed" instead of being reported as reliable
# ================================================================

import os
import numpy as np
import cv2
import albumentations as A
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
MODEL_PATH  = os.environ.get("MODEL_PATH", "sargassum_model_lab_first.h5")
CLASS_NAMES = ['SN1', 'SF3', 'SN8']
IMG_SIZE    = 300
NUM_CLASSES = len(CLASS_NAMES)

# Confidence threshold for the safety gate ("reliable" vs "review needed").
# Lowered from the training default (0.85) to 0.75 per deployment request.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))

# Temperature scaling factor fit on the validation set during the
# "lab-first" fine-tuning run (see TemperatureScaler.fit() output: T = 0.852).
# If you retrain the model, re-fit this value and update it here — it's
# baked in rather than fit at serving time because that requires the
# validation set, which isn't available in the deployed container.
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.852"))

# Number of augmented TTA crops (+1 clean resize) averaged per prediction,
# matching predict_with_tta(..., n_aug=7) / predict_reliable(...) in training.
TTA_N_AUG = int(os.environ.get("TTA_N_AUG", "7"))

# Hugging Face Hub model repo to download from at container startup if the
# .h5 isn't already present locally.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "Hossamfazaz/sargassum_enhanced_version")
HF_MODEL_FILE = os.environ.get("HF_MODEL_FILE", "sargassum_model_lab_first.h5")


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
    # hf_hub_download caches the file elsewhere; copy it to MODEL_PATH
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

# TTA augmentation pipeline — identical to predict_with_tta()/predict_reliable()
# in the training script.
tta_aug = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.88, 1.0), ratio=(0.95, 1.05), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.08, 0.08, p=0.3),
])


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

    # Warm up the model with a dummy batch (TTA_N_AUG + 1 images) so the
    # first real request isn't slowed down by graph tracing.
    dummy = np.zeros((TTA_N_AUG + 1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print(f"🔥 Model warmed up and ready (TTA n_aug={TTA_N_AUG}, T={TEMPERATURE}, "
          f"threshold={CONFIDENCE_THRESHOLD})")


def decode_bytes(image_bytes: bytes) -> np.ndarray:
    """BGR decode -> RGB, full resolution (TTA crops are applied downstream)."""
    data = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def predict_with_tta_and_scaling(src_rgb: np.ndarray) -> np.ndarray:
    """
    Mirrors predict_reliable() from training:
      - TTA_N_AUG augmented crops + 1 clean resize, averaged
      - temperature scaling applied to the averaged probabilities
    Returns a (NUM_CLASSES,) float32 array of calibrated probabilities.
    """
    imgs = [tta_aug(image=src_rgb)['image'].astype(np.float32) for _ in range(TTA_N_AUG)]
    imgs.append(cv2.resize(src_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32))

    probs = model.predict(np.array(imgs), verbose=0).mean(axis=0)

    # Temperature scaling: rescale the softmax "temperature" using the
    # log-prob trick, same as TemperatureScaler.scale() in training.
    probs = np.asarray(probs, dtype=np.float32)
    log_p = np.log(probs + 1e-8)
    scaled = tf.nn.softmax(log_p / TEMPERATURE, axis=-1).numpy()
    return scaled


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tta_n_aug": TTA_N_AUG,
        "temperature": TEMPERATURE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        src_rgb = decode_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    probs = predict_with_tta_and_scaling(src_rgb)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    is_reliable = confidence >= CONFIDENCE_THRESHOLD

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "reliable": is_reliable,
        "status": "reliable" if is_reliable else "review needed",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
        },
    }


# ---- Serve the simple upload frontend at "/" ----
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")