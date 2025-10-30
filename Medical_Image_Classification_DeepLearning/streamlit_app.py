# streamlit_app.py

import os, shutil
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Optional providers (installed via requirements):
# - huggingface_hub for HF download
# - gdown for Google Drive download

MODEL_LOCAL_PATH = "final_resnet50_finetuned.h5"   # where the .h5 will live locally
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# -------------------------------
# Download helpers
# -------------------------------
def _download_from_hf(repo_id: str, filename: str, revision: str | None = None, token: str | None = None) -> str:
    """Download model from Hugging Face Hub into the current directory."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
        local_dir=".",
        local_dir_use_symlinks=False,
    )
    if os.path.abspath(path) != os.path.abspath(MODEL_LOCAL_PATH):
        shutil.copy2(path, MODEL_LOCAL_PATH)
    return MODEL_LOCAL_PATH

def _download_from_gdrive(file_id: str) -> str:
    """Download model from Google Drive using a file id."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_LOCAL_PATH, quiet=False)
    return MODEL_LOCAL_PATH

def _ensure_model_file() -> str:
    """
    Ensure model file exists locally.
    Priority:
      1) If MODEL_LOCAL_PATH exists, reuse it.
      2) Else, try Hugging Face secrets.
      3) Else, try Google Drive secrets.
    """
    if os.path.exists(MODEL_LOCAL_PATH) and os.path.getsize(MODEL_LOCAL_PATH) > 0:
        return MODEL_LOCAL_PATH

    hf_repo   = st.secrets.get("HF_REPO_ID", None)
    hf_file   = st.secrets.get("HF_FILENAME", "final_resnet50_finetuned.h5")
    hf_rev    = st.secrets.get("HF_REVISION", None)
    hf_token  = st.secrets.get("HF_TOKEN", None)      # optional for private repos
    gdrv_id   = st.secrets.get("GDRIVE_FILE_ID", None)

    if hf_repo:
        st.info(f"Downloading model from Hugging Face: {hf_repo}/{hf_file}")
        return _download_from_hf(hf_repo, hf_file, hf_rev, hf_token)

    if gdrv_id:
        st.info("Downloading model from Google Drive…")
        return _download_from_gdrive(gdrv_id)

    raise FileNotFoundError(
        "Model file not found locally and no download secrets provided. "
        "Either place the .h5 next to this script or set HF_* or GDRIVE_FILE_ID secrets."
    )

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    """Lazy, cached model loader."""
    model_path = _ensure_model_file()
    model = load_model(model_path)
    return model

# -------------------------------
# Grad-CAM Function
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), predictions.numpy()[0]

# -------------------------------
# Preprocess and Grad-CAM overlay
# -------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    array = np.array(img)
    return img, np.expand_dims(array / 255.0, axis=0)

def overlay_gradcam(image_pil, heatmap):
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image_pil)
    overlay = cv2.addWeighted(heatmap_color, 0.4, image_np, 0.6, 0)
    return overlay[..., ::-1]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Chest X-Ray Classifier with Grad-CAM")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    model = get_model()
    original_img, img_array = preprocess_image(uploaded_file)
    heatmap, preds = make_gradcam_heatmap(img_array, model, "conv5_block3_out")
    pred_class = np.argmax(preds)
    confidence = preds[pred_class]

    st.subheader(f"Prediction: {class_names[pred_class]} ({confidence:.2f})")
    st.image(original_img, caption="Original Image", use_column_width=True)

    overlay = overlay_gradcam(original_img, heatmap)
    st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)