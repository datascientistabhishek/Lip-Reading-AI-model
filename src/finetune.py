# finetune.py
import os
import json
import random
import cv2
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ==============================
# Paths & Config
# ==============================
FRAMES_DIR_NEW = r"D:\lip_reading_ai\frames_unseen"   # <- new speakers' frames (s21..s25)
LABELS_PATH_NEW = r"D:\lip_reading_ai\labels_new.json"  # <- labels for s21..s25

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BASE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "lipreading_checkpoint.pth")  # from previous training
BASE_MODEL_PATH = "lipreading_model_final.pth"  # fallback if checkpoint not present

FINETUNE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "lipreading_finetune_ckpt.pth")
OUTPUT_MODEL_PATH = "lipreading_model_finetuned.pth"

DEVICE = torch.device("cpu")
IMG_SIZE = (64, 64)
BATCH_SIZE = 4
EPOCHS = 6                # keep small for quick fine-tune
LR = 5e-5                 # smaller LR for fine-tuning
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
FREEZE_CNN_EPOCHS = 2     # freeze CNN for first N epochs
VAL_SPLIT = 0.1           # 10% new data for validation

# ===== same vocab as training =====
VOCAB = {i: chr(96 + i) for i in range(1, 27)}
VOCAB[27] = " "
CHAR2IDX = {v: k for k, v in VOCAB.items()}
IDX2CHAR = {k: v for v, k in CHAR2IDX.items()}

# ==============================
# Dataset (new speakers only)
# ==============================
class LipReadingDataset(Dataset):
    def __init__(self, frames_dir, labels_path, subset_keys=None):
        self.frames_dir = Path(frames_dir)
        with open(labels_path, "r", encoding="utf-8") as f:
            all_labels = json.load(f)

        # restrict to keys that actually exist in frames_dir
        present = []
        for speaker_dir in sorted([d for d in self.frames_dir.iterdir() if d.is_dir()]):
            for video_dir in sorted([d for d in speaker_dir.iterdir() if d.is_dir()]):
                key = f"{speaker_dir.name}_{video_dir.name}"
                if key in all_labels:
                    present.append(key)

        if subset_keys is not None:
            present = [k for k in present if k in subset_keys]

        self.labels = {k: all_labels[k] for k in present}
        self.samples = []
        for k in present:
            spk, vid = k.split("_", 1)
            self.samples.append((self.frames_dir / spk / vid, self.labels[k]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_path, label_text = self.samples[idx]
        frame_files = sorted(os.listdir(frames_path))
        frames = []
        for f in frame_files:
            img = cv2.imread(str(frames_path / f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            frames.append(img)
        frames_tensor = torch.stack(frames)

        label_encoded = torch.tensor(
            [CHAR2IDX[c] for c in label_text.lower() if c.isalpha() or c == " "],
            dtype=torch.long
        )
        return frames_tensor, label_encoded, frames_tensor.size(0), len(label_encoded)

def collate_fn(batch):
    frames_list, labels_list, frame_lengths, label_lengths = zip(*batch)

    max_len = max(f.size(0) for f in frames_list)
    padded_frames = []
    for f in frames_list:
        pad = max_len - f.size(0)
        if pad > 0:
            pad_tensor = torch.zeros((pad, *f.shape[1:]), dtype=f.dtype)
            f = torch.cat([f, pad_tensor], dim=0)
        padded_frames.append(f)
    padded_frames = torch.stack(padded_frames)

    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=0)
    return padded_frames, padded_labels, torch.tensor(frame_lengths), torch.tensor(label_lengths)

# ==============================
# Model (same as training)
# ==============================
class LipReadingModel(nn.Module):
    def __init__(self, vocab_size=28):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.rnn = nn.LSTM(64 * 16 * 16, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256 * 2, vocab_size)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

# ==============================
# Metrics / Decoding
# ==============================
def greedy_decode(log_probs):
    seq = torch.argmax(log_probs, dim=-1).cpu().numpy().tolist()
    collapsed, prev = [], None
    for s in seq:
        if s != prev:
            collapsed.append(s)
        prev = s
    return "".join(IDX2CHAR.get(i, "") for i in collapsed if i != 0)

def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1] + (0 if a[i-1]==b[j-1] else 1))
    return dp[m][n]

def compute_cer(ref, hyp):
    if len(ref) == 0: return 0.0 if len(hyp) == 0 else 1.0
    return edit_distance(list(ref), list(hyp)) / len(ref)

def compute_wer(ref, hyp):
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0: return 0.0 if len(hyp_w) == 0 else 1.0
    return edit_distance(ref_w, hyp_w) / len(ref_w)

# ==============================
# Split new keys into train/val
# ==============================
with open(LABELS_PATH_NEW, "r", encoding="utf-8") as f:
    labels_new = json.load(f)

# Keep only keys that exist under FRAMES_DIR_NEW
valid_keys = []
for spk_dir in Path(FRAMES_DIR_NEW).iterdir():
    if not spk_dir.is_dir(): continue
    for vid_dir in spk_dir.iterdir():
        if not vid_dir.is_dir(): continue
        k = f"{spk_dir.name}_{vid_dir.name}"
        if k in labels_new:
            valid_keys.append(k)

random.shuffle(valid_keys)
val_count = max(1, int(len(valid_keys) * VAL_SPLIT))
val_keys = set(valid_keys[:val_count])
train_keys = set(valid_keys[val_count:])

train_ds = LipReadingDataset(FRAMES_DIR_NEW, LABELS_PATH_NEW, subset_keys=train_keys)
val_ds   = LipReadingDataset(FRAMES_DIR_NEW, LABELS_PATH_NEW, subset_keys=val_keys)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

print(f"New fine-tune dataset → train: {len(train_ds)} vids | val: {len(val_ds)} vids")

# ==============================
# Init model & load base weights
# ==============================
model = LipReadingModel(vocab_size=len(VOCAB) + 1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

start_epoch = 1
loaded = False
if os.path.exists(BASE_CHECKPOINT):
    print("🔄 Loading base checkpoint...")
    ckpt = torch.load(BASE_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    # fresh optimizer for fine-tune
    loaded = True
elif os.path.exists(BASE_MODEL_PATH):
    print("🔄 Loading base final model weights...")
    state = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    loaded = True

if not loaded:
    print("⚠️ Could not find base weights; training will start from scratch on new data.")

best_val_cer = 1e9

# Optionally freeze CNN for first few epochs
def set_cnn_frozen(flag: bool):
    for p in model.cnn.parameters():
        p.requires_grad = not flag

# ==============================
# Evaluation
# ==============================
def evaluate():
    model.eval()
    total_cer, total_wer, n = 0.0, 0.0, 0
    with torch.no_grad():
        for frames, labels, frame_lengths, label_lengths in val_loader:
            frames = frames.to(DEVICE)
            out = model(frames)              # [B,T,V]
            out = out.permute(1, 0, 2)       # [T,B,V] for CTC
            # decode first item
            pred_text = greedy_decode(out[:, 0, :])

            # build reference text
            ref_text = "".join(IDX2CHAR.get(i.item(), "") for i in labels[0] if i.item() != 0)

            total_cer += compute_cer(ref_text, pred_text)
            total_wer += compute_wer(ref_text, pred_text)
            n += 1
    return (total_cer / max(1, n), total_wer / max(1, n), n)

# ==============================
# Fine-tuning loop
# ==============================
for epoch in range(start_epoch, EPOCHS + 1):
    freeze_now = (epoch <= FREEZE_CNN_EPOCHS)
    set_cnn_frozen(freeze_now)
    if freeze_now:
        model.cnn.eval()  # keep BN/conv frozen behavior
    else:
        model.cnn.train()

    model.rnn.train()
    model.fc.train()

    running = 0.0
    for frames, labels, frame_lengths, label_lengths in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}/{EPOCHS}"):
        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(frames)        # [B,T,V]
        outputs = outputs.permute(1, 0, 2)  # [T,B,V]
        loss = criterion(outputs, labels, frame_lengths, label_lengths)
        loss.backward()

        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        running += loss.item()

    train_loss = running / max(1, len(train_loader))
    val_cer, val_wer, val_n = evaluate()

    print(f"Epoch {epoch}: TrainLoss {train_loss:.4f} | Val CER {val_cer:.3f} | Val WER {val_wer:.3f} on {val_n} samples")

    # Save best on CER
    if val_cer < best_val_cer:
        best_val_cer = val_cer
        torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
        print(f"💾 Saved best fine-tuned model → {OUTPUT_MODEL_PATH} (CER {best_val_cer:.3f})")

    # Save running checkpoint
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_cer": best_val_cer
    }, FINETUNE_CHECKPOINT)

print("✅ Fine-tuning complete.")
