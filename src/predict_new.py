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
FRAMES_ROOT = r"D:\lip_reading_ai\frames_unseen"   # 👈 यहाँ बदल दो
LABELS_PATH = r"D:\lip_reading_ai\labels_new.json" # 👈 new speakers वाले labels
MODEL_PATH = "lipreading_model_final.pth"
RESULTS_FILE = "results_new.csv"   # ताकि पुराना results.csv safe रहे
DEVICE = torch.device("cpu")

# Load labels
with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

# ==============================
# Load model
# ==============================
model = LipReadingModel(vocab_size=len(CHAR2IDX) + 1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded.")

# ==============================
# Utils
# ==============================
def load_frames_from_folder(folder_path):
    frame_files = sorted(os.listdir(folder_path))
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(folder_path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames.append(img)
    return torch.stack(frames)

# Ensure results_new.csv has header
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
        frames = load_frames_from_folder(folder_path).unsqueeze(0).to(DEVICE)  # Add batch dim
        with torch.no_grad():
            outputs = model(frames)
            outputs = outputs.permute(1, 0, 2)
            pred_text = greedy_decode(outputs[:, 0, :])

        # Get ground truth
        key = folder.replace("\\", "_").replace("/", "_")
        actual_text = LABELS.get(key, "UNKNOWN")

        # Compute metrics
        cer = compute_cer(actual_text, pred_text) if actual_text != "UNKNOWN" else -1
        wer = compute_wer(actual_text, pred_text) if actual_text != "UNKNOWN" else -1

        # Print to terminal
        print(f"📝 PREDICTION: {pred_text}")
        print(f"🎯 ACTUAL:     {actual_text}")
        print(f"📊 CER: {cer:.3f}, WER: {wer:.3f}")

        # Save to CSV
        with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([folder, pred_text, actual_text, cer, wer])
        print(f"✅ Logged to {RESULTS_FILE}\n")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
