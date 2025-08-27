import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
import cv2

# ==============================
# Config
# ==============================
DATA_DIR = r"D:\lip_reading_ai\frames"
LABELS_PATH = r"D:\lip_reading_ai\labels.json"
BATCH_SIZE = 4
START_EPOCH = 11
END_EPOCH = 20
LR = 1e-4
DEVICE = torch.device("cpu")
IMG_SIZE = (64, 64)

VOCAB = {i: chr(96 + i) for i in range(1, 27)}
VOCAB[27] = " "
CHAR2IDX = {v: k for k, v in VOCAB.items()}
IDX2CHAR = {k: v for v, k in CHAR2IDX.items()}

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "lipreading_checkpoint.pth")

# ==============================
# Dataset
# ==============================
class LipReadingDataset(Dataset):
    def __init__(self, frames_dir, labels_path):
        self.frames_dir = frames_dir
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        self.samples = []
        for speaker_id in os.listdir(frames_dir):
            speaker_path = os.path.join(frames_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            for video_id in os.listdir(speaker_path):
                video_path = os.path.join(speaker_path, video_id)
                if os.path.isdir(video_path):
                    key = f"{speaker_id}_{video_id}"
                    if key in self.labels:
                        self.samples.append((video_path, self.labels[key]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_path, label_text = self.samples[idx]
        frame_files = sorted(os.listdir(frames_path))
        frames = []
        for f in frame_files:
            img = cv2.imread(os.path.join(frames_path, f))
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

# ==============================
# Collate function
# ==============================
def collate_fn(batch):
    frames_list, labels_list, frame_lengths, label_lengths = zip(*batch)

    max_len = max(f.size(0) for f in frames_list)
    padded_frames = []
    for f in frames_list:
        pad_size = max_len - f.size(0)
        if pad_size > 0:
            pad_tensor = torch.zeros((pad_size, *f.shape[1:]), dtype=f.dtype)
            f = torch.cat([f, pad_tensor], dim=0)
        padded_frames.append(f)
    padded_frames = torch.stack(padded_frames)

    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=0)

    return padded_frames, padded_labels, torch.tensor(frame_lengths), torch.tensor(label_lengths)

# ==============================
# Model
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
# Metrics
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

def compute_cer(ref, hyp): return edit_distance(list(ref), list(hyp)) / len(ref)
def compute_wer(ref, hyp): return edit_distance(ref.split(), hyp.split()) / len(ref.split())

# ==============================
# Training
# ==============================
dataset = LipReadingDataset(DATA_DIR, LABELS_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)

model = LipReadingModel(vocab_size=len(VOCAB) + 1).to(DEVICE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

start_epoch = START_EPOCH
if os.path.exists(CHECKPOINT_PATH):
    print("🔄 Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"⏩ Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, END_EPOCH + 1):
    model.train()
    epoch_loss = 0.0
    for frames, labels, frame_lengths, label_lengths in tqdm(dataloader, desc=f"Epoch {epoch}/{END_EPOCH}"):
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(frames)
        outputs = outputs.permute(1, 0, 2)
        loss = criterion(outputs, labels, frame_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

    # quick evaluation (1 batch)
    model.eval()
    with torch.no_grad():
        frames, labels, frame_lengths, label_lengths = next(iter(dataloader))
        frames = frames.to(DEVICE)
        out = model(frames).permute(1,0,2)
        pred_text = greedy_decode(out[:,0,:])
        ref_text = "".join(IDX2CHAR.get(i.item(),"") for i in labels[0] if i.item()!=0)
        print(f"Sample REF: {ref_text}")
        print(f"Sample PRED: {pred_text}")
        print(f"CER: {compute_cer(ref_text,pred_text):.3f}, WER: {compute_wer(ref_text,pred_text):.3f}")

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, CHECKPOINT_PATH)
    print(f"💾 Checkpoint saved for epoch {epoch}")

torch.save(model.state_dict(), "lipreading_model_final.pth")
print("✅ Training complete. Final model saved.")
