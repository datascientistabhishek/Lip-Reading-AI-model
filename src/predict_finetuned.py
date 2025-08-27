# src/predict_finetuned.py

import os
import torch
import torch.nn as nn
import cv2
import json
import csv
from train import LipReadingModel, greedy_decode, compute_cer, compute_wer, CHAR2IDX, IDX2CHAR, IMG_SIZE

# ==============================
# Config
# ==============================
FRAMES_ROOT = r"D:\lip_reading_ai\frames_unseen"    # 👈 unseen frames
LABELS_PATH = r"D:\lip_reading_ai\labels_new.json"  # 👈 new speakers labels
MODEL_PATH = "lipreading_model_finetuned.pth"       # 👈 fine-tuned model
RESULTS_FILE = "results_finetuned.csv"              # separate results file
DEVICE = torch.device("cpu")

# Load labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

# ==============================
# Load model
# ==============================
model = LipReadingModel(vocab_size=len(CHAR2IDX) + 1).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
# state can be raw state_dict or a checkpoint dict; handle both
if isinstance(state, dict) and "model_state" in state:
    state = state["model_state"]
model.load_state_dict(state)
model.eval()
print("✅ Fine-tuned model loaded.")

# ==============================
# Utils
# ==============================
def load_frames_from_folder(folder_path):
    """Reads all frames in a folder -> (T, C, H, W) float tensor [0,1]."""
    frame_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    )
    if not frame_files:
        raise RuntimeError("No frame images found in folder.")
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is None:
            raise RuntimeError(f"Failed to read image: {f}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames.append(img)
    return torch.stack(frames)

# Ensure results csv has header
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Folder", "Prediction", "Actual", "CER", "WER"])

# ==============================
# Main loop
# ==============================
while True:
    folder = input("Enter frames folder (relative to FRAMES_ROOT e.g. s21/bbjx9a, or 'quit' to exit): ").strip()
    if folder.lower() == "quit":
        break

    folder_path = os.path.join(FRAMES_ROOT, folder)
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    print(f"🔎 Running prediction on: {folder_path}")
    try:
        frames = load_frames_from_folder(folder_path).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
        with torch.no_grad():
            outputs = model(frames)              # (1, T, V)
            outputs = outputs.permute(1, 0, 2)   # (T, 1, V) for greedy_decode
            pred_text = greedy_decode(outputs[:, 0, :])

        # Ground truth key (same convention as before)
        key = folder.replace("\\", "_").replace("/", "_")
        actual_text = LABELS.get(key, "UNKNOWN")

        # Metrics
        if actual_text == "UNKNOWN":
            cer, wer = -1, -1
        else:
            cer = compute_cer(actual_text, pred_text)
            wer = compute_wer(actual_text, pred_text)

        # Print
        print(f"📝 PREDICTION: {pred_text}")
        print(f"🎯 ACTUAL:     {actual_text}")
        if cer >= 0:
            print(f"📊 CER: {cer:.3f}, WER: {wer:.3f}")
        else:
            print("📊 CER/WER: N/A (label not found)")

        # Save to CSV
        with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([folder, pred_text, actual_text, cer, wer])

        print(f"✅ Logged to {RESULTS_FILE}\n")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
