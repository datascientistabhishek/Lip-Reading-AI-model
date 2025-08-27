# # streamlit run src/app.py
# # Streamlit UI for lipreading on GRID frames
# # Supports both Seen (s1–s20) and Unseen (s21–s25)
# # - Single video prediction (CER/WER + text)
# # - Dataset evaluation: histograms, averages, confusion matrix

# import os
# import json
# import cv2
# import torch
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# from train import (
#     LipReadingModel,
#     greedy_decode,
#     compute_cer,
#     compute_wer,
#     CHAR2IDX,
#     IDX2CHAR,
#     IMG_SIZE,
# )

# # ==============================
# # Config — dataset paths
# # ==============================
# DATASETS = {
#     "Seen (s1–s20)": {
#         "frames": r"D:\lip_reading_ai\frames",
#         "labels": r"D:\lip_reading_ai\labels.json",
#     },
#     "Unseen (s21–s25)": {
#         "frames": r"D:\lip_reading_ai\frames_unseen",
#         "labels": r"D:\lip_reading_ai\labels_new.json",
#     },
# }
# MODEL_PATH = r"D:\lip_reading_ai\lipreading_model_finetuned.pth"
# DEVICE = torch.device("cpu")

# # ------------------------------
# # Streamlit layout tweaks
# # ------------------------------
# st.set_page_config(page_title="Lip Reading App (GRID)", layout="wide")
# st.title("🎥 Lip Reading App (GRID Dataset)")

# # ------------------------------
# # Helpers
# # ------------------------------
# @st.cache_data(show_spinner=False)
# def load_labels(path: str):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# @st.cache_resource(show_spinner=True)
# def load_model(model_path: str):
#     model = LipReadingModel(vocab_size=len(CHAR2IDX) + 1).to(DEVICE)
#     state = torch.load(model_path, map_location=DEVICE)
#     if isinstance(state, dict) and "model_state" in state:
#         state = state["model_state"]
#     model.load_state_dict(state)
#     model.eval()
#     return model

# def list_speakers(root: str):
#     if not os.path.isdir(root):
#         return []
#     return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.lower().startswith("s")])

# def list_videos_for_speaker(root: str, speaker: str):
#     p = os.path.join(root, speaker)
#     if not os.path.isdir(p):
#         return []
#     return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])

# def load_frames_from_folder(folder_path: str) -> torch.Tensor:
#     frame_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
#     if not frame_files:
#         raise RuntimeError(f"No frames found in: {folder_path}")
#     frames = []
#     for f in frame_files:
#         img = cv2.imread(os.path.join(folder_path, f))
#         if img is None:
#             raise RuntimeError(f"Failed to read {f}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, IMG_SIZE)
#         img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         frames.append(img)
#     return torch.stack(frames)

# def predict_text(model, frames: torch.Tensor) -> str:
#     with torch.no_grad():
#         logits = model(frames)
#         logits = logits.permute(1, 0, 2)
#         return greedy_decode(logits[:, 0, :])

# def char_confusion_pairs(true_text: str, pred_text: str):
#     m = min(len(true_text), len(pred_text))
#     return list(true_text[:m]), list(pred_text[:m])

# # ==============================
# # Load model once
# # ==============================
# model = load_model(MODEL_PATH)

# # ==============================
# # Dataset selector
# # ==============================
# dataset_choice = st.radio("Choose Dataset", options=list(DATASETS.keys()))
# FRAMES_ROOT = DATASETS[dataset_choice]["frames"]
# LABELS_PATH = DATASETS[dataset_choice]["labels"]
# labels = load_labels(LABELS_PATH)

# # ==============================
# # Single-video Prediction
# # ==============================
# st.subheader("Single Video Prediction")

# col1, col2 = st.columns(2)
# speakers = list_speakers(FRAMES_ROOT)
# with col1:
#     speaker = st.selectbox("Speaker", options=speakers, index=0 if speakers else None)
# with col2:
#     videos = list_videos_for_speaker(FRAMES_ROOT, speaker) if speaker else []
#     video = st.selectbox("Video", options=videos, index=0 if videos else None)

# if st.button("▶️ Run Prediction", type="primary"):
#     if not speaker or not video:
#         st.error("Please select both speaker and video.")
#     else:
#         folder_path = os.path.join(FRAMES_ROOT, speaker, video)
#         try:
#             frames = load_frames_from_folder(folder_path).unsqueeze(0).to(DEVICE)
#             pred_text = predict_text(model, frames)
#         except Exception as e:
#             st.error(f"Prediction failed: {e}")
#             pred_text = None

#         if pred_text is not None:
#             key = f"{speaker}_{video}".replace("\\", "_").replace("/", "_")
#             actual = labels.get(key, "UNKNOWN")
#             st.write(f"**Prediction:** {pred_text}")
#             st.write(f"**Actual:** {actual}")

#             if actual != "UNKNOWN":
#                 cer = compute_cer(actual, pred_text)
#                 wer = compute_wer(actual, pred_text)
#                 st.write(f"**CER:** {cer:.3f}")
#                 st.write(f"**WER:** {wer:.3f}")
#             else:
#                 st.warning("Label not found → CER/WER N/A.")

# st.markdown("---")

# # ==============================
# # Dataset Evaluation & Graphs
# # ==============================
# st.subheader("📊 Dataset Evaluation")

# with st.expander("Options", expanded=False):
#     sel_speakers = st.multiselect("Speakers", options=speakers, default=speakers)
#     max_per_speaker = st.slider("Max videos per speaker", 10, 1000, 100, step=10)
#     plot_confusion = st.checkbox("Plot Confusion Matrix", value=True)

# if st.button("📈 Run Evaluation"):
#     all_cer, all_wer = [], []
#     y_true_chars, y_pred_chars = [], []

#     total_videos = sum(len(list_videos_for_speaker(FRAMES_ROOT, sp)[:max_per_speaker]) for sp in sel_speakers)
#     progress = st.progress(0.0)
#     done = 0

#     for sp in sel_speakers:
#         vids = list_videos_for_speaker(FRAMES_ROOT, sp)[:max_per_speaker]
#         for vid in vids:
#             try:
#                 folder_path = os.path.join(FRAMES_ROOT, sp, vid)
#                 frames = load_frames_from_folder(folder_path).unsqueeze(0).to(DEVICE)
#                 pred_text = predict_text(model, frames)
#                 key = f"{sp}_{vid}".replace("\\", "_").replace("/", "_")
#                 actual = labels.get(key, "UNKNOWN")

#                 if actual != "UNKNOWN":
#                     cer = compute_cer(actual, pred_text)
#                     wer = compute_wer(actual, pred_text)
#                     all_cer.append(cer)
#                     all_wer.append(wer)

#                     if plot_confusion:
#                         tc, pc = char_confusion_pairs(actual, pred_text)
#                         y_true_chars.extend(tc)
#                         y_pred_chars.extend(pc)
#             except Exception:
#                 pass
#             done += 1
#             progress.progress(done / total_videos)

#     if not all_cer:
#         st.warning("No samples evaluated.")
#     else:
#         avg_cer = float(np.mean(all_cer))
#         avg_wer = float(np.mean(all_wer))

#         st.markdown("#### Summary")
#         colA, colB = st.columns(2)

#         with colA:
#             fig0, ax0 = plt.subplots(figsize=(3, 3))
#             ax0.bar(["CER", "WER"], [avg_cer, avg_wer], color=["#ff9999", "#9999ff"])
#             for i, v in enumerate([avg_cer, avg_wer]):
#                 ax0.text(i, v + 0.01, f"{v:.2f}", ha="center")
#             ax0.set_ylim(0, max(avg_cer, avg_wer) + 0.2)
#             st.pyplot(fig0)

#         with colB:
#             fig1, ax1 = plt.subplots(figsize=(5, 3))
#             ax1.hist(all_cer, bins=20, alpha=0.6, label="CER")
#             ax1.hist(all_wer, bins=20, alpha=0.6, label="WER")
#             ax1.set_xlabel("Error Rate")
#             ax1.set_ylabel("Samples")
#             ax1.legend()
#             st.pyplot(fig1)

#         if plot_confusion and len(y_true_chars) > 0:
#             st.markdown("#### Confusion Matrix")
#             labels_cm = sorted(set(y_true_chars + y_pred_chars))
#             cm = confusion_matrix(y_true_chars, y_pred_chars, labels=labels_cm)
#             fig2, ax2 = plt.subplots(figsize=(6, 5))
#             sns.heatmap(cm, cmap="Blues", xticklabels=labels_cm, yticklabels=labels_cm, cbar=True, ax=ax2)
#             ax2.set_xlabel("Predicted")
#             ax2.set_ylabel("Actual")
#             st.pyplot(fig2)

#         st.success(f"✅ Done. Samples: {len(all_cer)} | Avg CER: {avg_cer:.3f} | Avg WER: {avg_wer:.3f}")



# streamlit run src/app.py
# Lip Reading Web App
# Supports Seen (s1–s20) and Unseen (s21–s25) datasets
# Features:
# - Single video prediction (CER/WER + transcript)
# - Dataset evaluation (summary, histograms, confusion matrix)

import os
import json
import cv2
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from train import (
    LipReadingModel,
    greedy_decode,
    compute_cer,
    compute_wer,
    CHAR2IDX,
    IMG_SIZE,
)

# ==============================
# Config — dataset paths
# ==============================
DATASETS = {
    "👀 Seen (s1–s20)": {
        "frames": r"D:\lip_reading_ai\frames",
        "labels": r"D:\lip_reading_ai\labels.json",
    },
    "🆕 Unseen (s21–s25)": {
        "frames": r"D:\lip_reading_ai\frames_unseen",
        "labels": r"D:\lip_reading_ai\labels_new.json",
    },
}
MODEL_PATH = r"D:\lip_reading_ai\lipreading_model_finetuned.pth"
DEVICE = torch.device("cpu")

# ------------------------------
# Streamlit page setup
# ------------------------------
st.set_page_config(page_title="👄 AI Lip Reader", layout="wide")
st.title("👄 SilentSpeech AI")
st.caption("Predict speech from lip movements & evaluate model performance.")

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    model = LipReadingModel(vocab_size=len(CHAR2IDX) + 1).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model

def list_speakers(root: str):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.lower().startswith("s")])

def list_videos_for_speaker(root: str, speaker: str):
    p = os.path.join(root, speaker)
    return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])

def load_frames_from_folder(folder_path: str) -> torch.Tensor:
    frame_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(folder_path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames.append(img)
    return torch.stack(frames)

def predict_text(model, frames: torch.Tensor) -> str:
    with torch.no_grad():
        logits = model(frames)
        logits = logits.permute(1, 0, 2)
        return greedy_decode(logits[:, 0, :])

def char_confusion_pairs(true_text: str, pred_text: str):
    m = min(len(true_text), len(pred_text))
    return list(true_text[:m]), list(pred_text[:m])

# ==============================
# Load model once
# ==============================
model = load_model(MODEL_PATH)

# ==============================
# Dataset selector
# ==============================
dataset_choice = st.radio("📂 Choose Dataset", options=list(DATASETS.keys()))
FRAMES_ROOT = DATASETS[dataset_choice]["frames"]
LABELS_PATH = DATASETS[dataset_choice]["labels"]
labels = load_labels(LABELS_PATH)

# ==============================
# Single-video Prediction
# ==============================
st.header("🎬 Single Video Prediction")

col1, col2 = st.columns(2)
speakers = list_speakers(FRAMES_ROOT)
with col1:
    speaker = st.selectbox("👤 Speaker", options=speakers, index=0 if speakers else None)
with col2:
    videos = list_videos_for_speaker(FRAMES_ROOT, speaker) if speaker else []
    video = st.selectbox("🎥 Video", options=videos, index=0 if videos else None)

if st.button("▶️ Run Prediction", type="primary"):
    if not speaker or not video:
        st.error("Please select both speaker and video.")
    else:
        folder_path = os.path.join(FRAMES_ROOT, speaker, video)
        try:
            frames = load_frames_from_folder(folder_path).unsqueeze(0).to(DEVICE)
            pred_text = predict_text(model, frames)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred_text = None

        if pred_text:
            key = f"{speaker}_{video}".replace("\\", "_").replace("/", "_")
            actual = labels.get(key, "UNKNOWN")

            st.markdown("### 📄 Result")
            st.write(f"**Prediction:** {pred_text}")
            st.write(f"**Actual:** {actual}")

            if actual != "UNKNOWN":
                cer = compute_cer(actual, pred_text)
                wer = compute_wer(actual, pred_text)
                st.metric("CER", f"{cer:.3f}")
                st.metric("WER", f"{wer:.3f}")
            else:
                st.warning("Label not found → CER/WER N/A.")

st.markdown("---")

# ==============================
# Dataset Evaluation & Graphs
# ==============================
st.header("📊 Model Evaluation")

with st.expander("⚙️ Options", expanded=False):
    sel_speakers = st.multiselect("🎙️ Speakers", options=speakers, default=speakers)
    max_per_speaker = st.slider("🎞️ Max videos per speaker", 10, 1000, 100, step=10)
    plot_confusion = st.checkbox("🔠 Show Confusion Matrix", value=True)

if st.button("📈 Run Evaluation"):
    all_cer, all_wer = [], []
    y_true_chars, y_pred_chars = [], []

    total_videos = sum(len(list_videos_for_speaker(FRAMES_ROOT, sp)[:max_per_speaker]) for sp in sel_speakers)
    progress = st.progress(0.0)
    done = 0

    for sp in sel_speakers:
        vids = list_videos_for_speaker(FRAMES_ROOT, sp)[:max_per_speaker]
        for vid in vids:
            try:
                folder_path = os.path.join(FRAMES_ROOT, sp, vid)
                frames = load_frames_from_folder(folder_path).unsqueeze(0).to(DEVICE)
                pred_text = predict_text(model, frames)
                key = f"{sp}_{vid}".replace("\\", "_").replace("/", "_")
                actual = labels.get(key, "UNKNOWN")

                if actual != "UNKNOWN":
                    cer = compute_cer(actual, pred_text)
                    wer = compute_wer(actual, pred_text)
                    all_cer.append(cer)
                    all_wer.append(wer)

                    if plot_confusion:
                        tc, pc = char_confusion_pairs(actual, pred_text)
                        y_true_chars.extend(tc)
                        y_pred_chars.extend(pc)
            except Exception:
                pass
            done += 1
            progress.progress(done / total_videos)

    if not all_cer:
        st.warning("No samples evaluated.")
    else:
        avg_cer, avg_wer = np.mean(all_cer), np.mean(all_wer)

        st.subheader("📈 Summary")
        colA, colB = st.columns(2)

        with colA:
            fig0, ax0 = plt.subplots(figsize=(3.2, 3.2))
            ax0.bar(["CER", "WER"], [avg_cer, avg_wer], color=["#00b4d8", "#ff6b6b"])
            for i, v in enumerate([avg_cer, avg_wer]):
                ax0.text(i, v + 0.01, f"{v:.2f}", ha="center")
            ax0.set_ylim(0, max(avg_cer, avg_wer) + 0.2)
            st.pyplot(fig0)

        with colB:
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            ax1.hist(all_cer, bins=20, alpha=0.6, label="CER", color="#00b4d8")
            ax1.hist(all_wer, bins=20, alpha=0.6, label="WER", color="#ff6b6b")
            ax1.set_xlabel("Error Rate")
            ax1.set_ylabel("Samples")
            ax1.legend()
            st.pyplot(fig1)

        if plot_confusion and len(y_true_chars) > 0:
            st.subheader("🔠 Character-level Confusion Matrix")
            labels_cm = sorted(set(y_true_chars + y_pred_chars))
            cm = confusion_matrix(y_true_chars, y_pred_chars, labels=labels_cm)
            fig2, ax2 = plt.subplots(figsize=(6, 4))  # smaller heatmap
            sns.heatmap(cm, cmap="Blues", xticklabels=labels_cm, yticklabels=labels_cm, cbar=True, ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)

        st.success(f"✅ Done. Samples: {len(all_cer)} | Avg CER: {avg_cer:.3f} | Avg WER: {avg_wer:.3f}")
