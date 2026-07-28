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
CLASS_NAMES = ["SN1", "SF3", "SN8"]  # internal order — MUST match CLASS_TO_INDEX from training, do not reorder
CONFIDENCE_THRESHOLD = 0.75          # lowered from training default (0.85) for the pilot
TEMPERATURE = 0.852                  # fixed value fit during calibration (from training run)
N_TTA = 7                            # matches predict_reliable() in training script

# Display-only relabeling — the model's internal class order/indices above are
# unchanged (they must match what the model was trained with); this mapping
# only controls what's SHOWN to users.
DISPLAY_CODE = {"SN1": "SNN", "SF3": "SFF", "SN8": "SNW"}
DISPLAY_FULL = {
    "SN1": "Sargassum natans natans",
    "SF3": "Sargassum fluitans fluitans",
    "SN8": "Sargassum natans wingei",
}


def display_name(internal_cls):
    return f"{DISPLAY_FULL[internal_cls]} ({DISPLAY_CODE[internal_cls]})"

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
# Translations — English / Español / Français
# -----------------------------
TEXT = {
    "en": {
        "title": "# 🌿 Sargassum Morphotype Classifier",
        "disclaimer": (
            "**Pilot / prototype model** — this tool is part of an ongoing research pilot "
            "(LEMAR / IRD) and predictions are not yet validated for definitive identification. "
            "This model is intentionally conservative: predictions below its confidence threshold "
            "are flagged **'Review needed'** rather than stated confidently — you may see this "
            "fairly often, especially on beach-collected (vs. lab) images. That's expected behavior, "
            "not a bug. Feedback is welcome and helps improve the model."
        ),
        "upload_label": "Upload a Sargassum image",
        "button_label": "Classify",
        "prediction_label": "Prediction",
        "breakdown_label": "Confidence by class",
        "footer": (
            "---\n"
            "*Was this prediction correct? Feedback helps us improve the model — "
            "contact the LEMAR Sargassum research team.*"
        ),
        "please_upload": "Please upload an image.",
        "rate_limited": "Rate limit reached — please wait a minute and try again.",
        "error": "Error processing image: {e}",
        "review_needed": "⚠️ Review needed — best guess: {cls} ({conf:.1%})",
        "confident": "✅ {cls} ({conf:.1%} confidence)",
        "lang_label": "Language / Idioma / Langue",
    },
    "es": {
        "title": "# 🌿 Clasificador de Morfotipos de Sargazo",
        "disclaimer": (
            "**Modelo piloto / prototipo** — esta herramienta forma parte de un proyecto piloto de "
            "investigación en curso (LEMAR / IRD) y las predicciones aún no están validadas para "
            "una identificación definitiva. Este modelo es intencionalmente conservador: las "
            "predicciones por debajo de su umbral de confianza se marcan como **'Revisión necesaria'** "
            "en lugar de darse con seguridad — es posible que esto ocurra con frecuencia, especialmente "
            "en imágenes recolectadas en playa (frente a laboratorio). Esto es un comportamiento "
            "esperado, no un error. Los comentarios son bienvenidos y ayudan a mejorar el modelo."
        ),
        "upload_label": "Sube una imagen de sargazo",
        "button_label": "Clasificar",
        "prediction_label": "Predicción",
        "breakdown_label": "Confianza por clase",
        "footer": (
            "---\n"
            "*¿Fue correcta esta predicción? Tus comentarios ayudan a mejorar el modelo — "
            "contacta al equipo de investigación de Sargazo de LEMAR.*"
        ),
        "please_upload": "Por favor, sube una imagen.",
        "rate_limited": "Límite de solicitudes alcanzado — espera un minuto e intenta de nuevo.",
        "error": "Error al procesar la imagen: {e}",
        "review_needed": "⚠️ Revisión necesaria — mejor estimación: {cls} ({conf:.1%})",
        "confident": "✅ {cls} ({conf:.1%} de confianza)",
        "lang_label": "Language / Idioma / Langue",
    },
    "fr": {
        "title": "# 🌿 Classificateur de Morphotypes de Sargasses",
        "disclaimer": (
            "**Modèle pilote / prototype** — cet outil fait partie d'un projet pilote de recherche "
            "en cours (LEMAR / IRD) et les prédictions ne sont pas encore validées pour une "
            "identification définitive. Ce modèle est volontairement prudent : les prédictions "
            "en dessous de son seuil de confiance sont signalées **'Vérification nécessaire'** "
            "plutôt qu'affirmées avec certitude — cela peut arriver assez souvent, surtout sur "
            "des images collectées en plage (par rapport au laboratoire). C'est un comportement "
            "attendu, pas un bug. Vos retours sont les bienvenus et aident à améliorer le modèle."
        ),
        "upload_label": "Téléchargez une image de sargasses",
        "button_label": "Classifier",
        "prediction_label": "Prédiction",
        "breakdown_label": "Confiance par classe",
        "footer": (
            "---\n"
            "*Cette prédiction était-elle correcte ? Vos retours aident à améliorer le modèle — "
            "contactez l'équipe de recherche Sargasses du LEMAR.*"
        ),
        "please_upload": "Veuillez télécharger une image.",
        "rate_limited": "Limite de requêtes atteinte — veuillez patienter une minute et réessayer.",
        "error": "Erreur lors du traitement de l'image : {e}",
        "review_needed": "⚠️ Vérification nécessaire — meilleure estimation : {cls} ({conf:.1%})",
        "confident": "✅ {cls} ({conf:.1%} de confiance)",
        "lang_label": "Language / Idioma / Langue",
    },
}

LANG_CHOICES = ["English", "Español", "Français"]
LANG_CODE = {"English": "en", "Español": "es", "Français": "fr"}

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


def predict(image, lang_code, request: gr.Request):
    t = TEXT.get(lang_code, TEXT["en"])

    if image is None:
        return t["please_upload"], None

    if not check_rate_limit(request):
        return t["rate_limited"], None

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
        breakdown = {
            DISPLAY_CODE[CLASS_NAMES[i]]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        top_display = display_name(CLASS_NAMES[top_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            label = t["review_needed"].format(cls=top_display, conf=confidence)
        else:
            label = t["confident"].format(cls=top_display, conf=confidence)

        return label, breakdown

    except Exception as e:
        return t["error"].format(e=e), None


# -----------------------------
# UI
# -----------------------------
DEFAULT_LANG = "en"


def on_language_change(lang_choice):
    code = LANG_CODE.get(lang_choice, "en")
    t = TEXT[code]
    return (
        code,  # lang_state
        gr.update(value=t["title"]),
        gr.update(value=t["disclaimer"]),
        gr.update(label=t["upload_label"]),
        gr.update(value=t["button_label"]),
        gr.update(label=t["prediction_label"]),
        gr.update(label=t["breakdown_label"]),
        gr.update(value=t["footer"]),
    )


with gr.Blocks(title="Sargassum Morphotype Classifier — Pilot") as demo:
    lang_state = gr.State(DEFAULT_LANG)

    lang_dropdown = gr.Dropdown(
        choices=LANG_CHOICES, value="English",
        label=TEXT[DEFAULT_LANG]["lang_label"],
    )
    title_md = gr.Markdown(TEXT[DEFAULT_LANG]["title"])
    disclaimer_md = gr.Markdown(TEXT[DEFAULT_LANG]["disclaimer"])

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label=TEXT[DEFAULT_LANG]["upload_label"])
            submit_btn = gr.Button(TEXT[DEFAULT_LANG]["button_label"], variant="primary")
        with gr.Column():
            label_output = gr.Textbox(label=TEXT[DEFAULT_LANG]["prediction_label"])
            breakdown_output = gr.Label(label=TEXT[DEFAULT_LANG]["breakdown_label"])

    footer_md = gr.Markdown(TEXT[DEFAULT_LANG]["footer"])

    lang_dropdown.change(
        fn=on_language_change,
        inputs=[lang_dropdown],
        outputs=[
            lang_state, title_md, disclaimer_md,
            image_input, submit_btn, label_output, breakdown_output, footer_md,
        ],
    )

    submit_btn.click(
        fn=predict,
        inputs=[image_input, lang_state],
        outputs=[label_output, breakdown_output],
    )

if __name__ == "__main__":
    demo.launch()