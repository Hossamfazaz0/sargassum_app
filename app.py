"""
Sargassum Morphotype Classifier — Pilot Test App (v2)
--------------------------------------------------------
Public-facing Gradio demo for the "lab-first reliable fine-tuning" model
(LEMAR / IRD Sargassum project). Matches the updated training pipeline:

  - Preprocessing (EfficientNet preprocess_input) is baked INTO the model
    via a Lambda layer, so this app feeds raw resized images (0-255),
    NOT pre-normalized ones.
  - TTA (test-time augmentation) is the DEFAULT inference mode, matching
    predict_reliable() from the training script (7 augmented crops +
    1 center resize, averaged).
  - Temperature scaling is applied with the fixed T fit during training.
  - Safety gate: predictions below CONFIDENCE_THRESHOLD are flagged
    "Review needed" instead of stated confidently.

Model is pulled automatically from Hugging Face Hub at startup:
    Hossamfazaz/sargassum_enhanced_version
"""

import gradio as gr
import numpy as np
import cv2
import time
import albumentations as A
from huggingface_hub import hf_hub_download, list_repo_files
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efn_preprocess

# -----------------------------
# CONFIG — matches training script
# -----------------------------
HF_REPO_ID = "Hossamfazaz/sargassum_enhanced_version"
IMG_SIZE = 300
CLASS_NAMES = ["SN1", "SF3", "SN8"]  # must match CLASS_TO_INDEX order from training
CONFIDENCE_THRESHOLD = 0.75          # lowered from training default (0.85) for the pilot
TEMPERATURE = 0.852                  # fixed value fit during calibration (from training run)
N_TTA = 7                            # matches predict_reliable() in training script

# -----------------------------
# Download + load model from Hugging Face Hub
# -----------------------------
print(f"Locating model file in {HF_REPO_ID} ...")
repo_files = list_repo_files(HF_REPO_ID)
h5_files = [f for f in repo_files if f.endswith(".h5")]
if not h5_files:
    raise RuntimeError(
        f"No .h5 file found in {HF_REPO_ID}. Files present: {repo_files}. "
        "Update HF_REPO_ID / filename in app.py if the repo structure changed."
    )
model_filename = h5_files[0]
print(f"Found model file: {model_filename}. Downloading...")
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_filename)

print("Loading model...")
# The model's Lambda preprocessing layer references efn_preprocess by name,
# so it must be passed via custom_objects to reconstruct correctly.
model = load_model(model_path, custom_objects={"efn_preprocess": efn_preprocess}, compile=False)
print("Model loaded.")

# -----------------------------
# TTA augmentation — matches training script exactly
# -----------------------------
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
# Simple in-memory rate limiting (per-session, resets on restart — fine for a pilot)
# -----------------------------
request_log = {}
RATE_LIMIT_PER_MIN = 10


def check_rate_limit(request: gr.Request):
    if request is None:
        return True
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = request_log.setdefault(ip, [])
    window[:] = [t for t in window if now - t < 60]
    if len(window) >= RATE_LIMIT_PER_MIN:
        return False
    window.append(now)
    return True


def predict(image, request: gr.Request):
    if image is None:
        return "Please upload an image.", None

    if not check_rate_limit(request):
        return "Rate limit reached — please wait a minute and try again.", None

    try:
        # NOTE: no manual preprocess_input call here — the model applies it
        # internally via its Lambda layer. We only resize / feed raw pixels.
        src = np.array(image.convert("RGB"))

        imgs = [tta_aug(image=src)["image"].astype(np.float32) for _ in range(N_TTA)]
        imgs.append(cv2.resize(src, (IMG_SIZE, IMG_SIZE)).astype(np.float32))
        batch = np.array(imgs)

        raw_probs = model.predict(batch, verbose=0).mean(axis=0)
        probs = temperature_scale(raw_probs)

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        breakdown = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        if confidence < CONFIDENCE_THRESHOLD:
            label = f"⚠️ Review needed — best guess: {CLASS_NAMES[top_idx]} ({confidence:.1%})"
        else:
            label = f"✅ {CLASS_NAMES[top_idx]} ({confidence:.1%} confidence)"

        return label, breakdown

    except Exception as e:
        return f"Error processing image: {e}", None


# -----------------------------
# UI
# -----------------------------
DISCLAIMER = (
    "**Pilot / prototype model** — this tool is part of an ongoing research pilot "
    "(LEMAR / IRD) and predictions are not yet validated for definitive identification. "
    "This model is intentionally conservative: predictions below its confidence threshold "
    "are flagged **'Review needed'** rather than stated confidently — you may see this "
    "fairly often, especially on beach-collected (vs. lab) images. That's expected behavior, "
    "not a bug. Feedback is welcome and helps improve the model."
)

with gr.Blocks(title="Sargassum Morphotype Classifier — Pilot") as demo:
    gr.Markdown("# 🌿 Sargassum Morphotype Classifier")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a Sargassum image")
            submit_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            label_output = gr.Textbox(label="Prediction")
            breakdown_output = gr.Label(label="Confidence by class")

    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[label_output, breakdown_output],
    )

    gr.Markdown(
        "---\n"
        "*Was this prediction correct? Feedback helps us improve the model — "
        "contact the LEMAR Sargassum research team.*"
    )

if __name__ == "__main__":
    demo.launch()