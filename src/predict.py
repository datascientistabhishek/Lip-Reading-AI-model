# import os
# import sys
# import json
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2

# # -----------------------
# # Config
# # -----------------------
# FRAMES_ROOT = r"D:\lip_reading_ai\frames"
# MODEL_PATH = r"D:\lip_reading_ai\lipreading_model_final.pth"
# LABELS_PATH = r"D:\lip_reading_ai\labels.json"
# DEVICE = torch.device("cpu")
# IMG_SIZE = (64, 64)

# # Vocabulary
# VOCAB = {i: chr(96 + i) for i in range(1, 27)}
# VOCAB[27] = " "  # space
# CHAR_TO_INDEX = {v: k for k, v in VOCAB.items()}
# IDX2CHAR = {k: v for v, k in CHAR_TO_INDEX.items()}

# # -----------------------
# # Model (same as train.py)
# # -----------------------
# class LipReadingModel(nn.Module):
#     def __init__(self, vocab_size=len(VOCAB) + 1):  # +1 for blank
#         super(LipReadingModel, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
#         )
#         self.rnn = nn.LSTM(64 * 16 * 16, 256, num_layers=2, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(256 * 2, vocab_size)

#     def forward(self, x):
#         B, T, C, H, W = x.size()
#         x = x.reshape(B * T, C, H, W)   # FIXED: reshape
#         x = self.cnn(x)
#         x = x.reshape(B, T, -1)
#         x, _ = self.rnn(x)
#         x = self.fc(x)
#         return x.log_softmax(2)

# # -----------------------
# # Helpers
# # -----------------------
# def load_model(path: str, device: torch.device):
#     model = LipReadingModel().to(device)
#     sd = torch.load(path, map_location=device)
#     if isinstance(sd, dict) and "model_state" in sd:
#         model.load_state_dict(sd["model_state"])
#     else:
#         model.load_state_dict(sd)
#     model.eval()
#     return model

# def load_frames(frames_folder: str, img_size=IMG_SIZE):
#     files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png'))])
#     if len(files) == 0:
#         raise FileNotFoundError(f"No image frames found in {frames_folder}")
#     imgs = []
#     for fn in files:
#         p = os.path.join(frames_folder, fn)
#         img = cv2.imread(p)
#         if img is None:
#             continue
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, img_size)
#         img = img.astype(np.float32) / 255.0
#         imgs.append(img)
#     arr = np.stack(imgs)  # (T, H, W, C)
#     tensor = torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)  # (T, C, H, W)
#     tensor = tensor.unsqueeze(0)  # (1, T, C, H, W)
#     return tensor

# def greedy_ctc_decode(log_probs: np.ndarray) -> str:
#     seq = np.argmax(log_probs, axis=1).tolist()  # per timestep index
#     collapsed = []
#     prev = None
#     for s in seq:
#         if s != prev:
#             collapsed.append(s)
#         prev = s
#     chars = []
#     for idx in collapsed:
#         if idx == 0:  # blank
#             continue
#         ch = VOCAB.get(int(idx), "")
#         if ch:
#             chars.append(ch)
#     return "".join(chars)

# # -----------------------
# # Metrics
# # -----------------------
# def edit_distance(a, b):
#     m, n = len(a), len(b)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     for i in range(m + 1):
#         dp[i][0] = i
#     for j in range(n + 1):
#         dp[0][j] = j
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             dp[i][j] = min(
#                 dp[i - 1][j] + 1,
#                 dp[i][j - 1] + 1,
#                 dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
#             )
#     return dp[m][n]

# def compute_cer(ref, hyp):
#     return edit_distance(list(ref), list(hyp)) / max(1, len(ref))

# def compute_wer(ref, hyp):
#     return edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))

# # -----------------------
# # Prediction
# # -----------------------
# def predict_from_folder(model, frames_folder, labels):
#     tensor = load_frames(frames_folder)
#     tensor = tensor.to(DEVICE)
#     with torch.no_grad():
#         out = model(tensor)  # (1, T, vocab)
#         out_np = out.squeeze(0).cpu().numpy()
#         pred = greedy_ctc_decode(out_np)

#     # ground truth lookup
#     folder_name = os.path.basename(frames_folder)
#     speaker = os.path.basename(os.path.dirname(frames_folder))
#     key = f"{speaker}_{folder_name}"
#     actual = labels.get(key, "[Not found in labels.json]")

#     # compute error if available
#     cer = compute_cer(actual, pred) if actual != "[Not found in labels.json]" else None
#     wer = compute_wer(actual, pred) if actual != "[Not found in labels.json]" else None

#     return pred, actual, cer, wer

# # -----------------------
# # CLI
# # -----------------------
# if __name__ == "__main__":
#     if not os.path.exists(MODEL_PATH):
#         print("❌ ERROR: Model not found at", MODEL_PATH)
#         sys.exit(1)

#     model = load_model(MODEL_PATH, DEVICE)
#     print("✅ Model loaded.")

#     with open(LABELS_PATH, "r") as f:
#         labels = json.load(f)

#     inp = input("Enter frames folder (relative to FRAMES_ROOT e.g. s8/lrwk8s, or full path): ").strip()

#     if os.path.isabs(inp):
#         frames_folder = inp
#     else:
#         frames_folder = os.path.join(FRAMES_ROOT, inp.replace("/", os.sep).replace("\\", os.sep))

#     if not os.path.isdir(frames_folder):
#         print("❌ Folder not found:", frames_folder)
#         sys.exit(1)

#     print("🔎 Running prediction on:", frames_folder)
#     try:
#         pred, actual, cer, wer = predict_from_folder(model, frames_folder, labels)
#         print(f"📝 PREDICTION: {pred}")
#         print(f"🎯 ACTUAL:     {actual}")
#         if cer is not None:
#             print(f"📊 CER: {cer:.3f}, WER: {wer:.3f}")
#     except Exception as e:
#         print("❌ Prediction failed:", e)
#         raise

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
FRAMES_ROOT = r"D:\lip_reading_ai\frames"
LABELS_PATH = r"D:\lip_reading_ai\labels.json"
MODEL_PATH = "lipreading_model_final.pth"
RESULTS_FILE = "results.csv"
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

# Ensure results.csv has header
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Folder", "Prediction", "Actual", "CER", "WER"])

# ==============================
# Main loop
# ==============================
while True:
    folder = input("Enter frames folder (relative to FRAMES_ROOT e.g. s8/lrwk8s, or 'quit' to exit): ").strip()
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
