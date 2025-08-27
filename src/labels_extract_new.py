import json
from pathlib import Path

# Paths
DATA_DIR = Path(r"D:\lip_reading_ai\data")    # apna dataset path
OUTPUT_JSON = Path(r"D:\lip_reading_ai\labels_new.json")  # sirf s21–s25 ke liye

def extract_labels_new(data_dir):
    labels = {}

    # 🔹 Sirf s21–s25 speakers select karna
    speakers = [d for d in data_dir.iterdir() if d.is_dir() and int(d.name[1:]) >= 21]

    for speaker in sorted(speakers, key=lambda x: int(x.name[1:])):
        align_folder = speaker / "align"
        align_files = sorted(align_folder.glob("*.align"))

        for af in align_files:
            vid_name = af.stem
            key = f"{speaker.name}_{vid_name}"

            with open(af, "r") as f:
                words = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:  # start, end, word
                        word = parts[2]
                        if word.lower() != "sil":  # Ignore silence tokens
                            words.append(word)

            labels[key] = " ".join(words)

    return labels


if __name__ == "__main__":
    labels_dict = extract_labels_new(DATA_DIR)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)

    print(f"✅ New labels saved to {OUTPUT_JSON} with {len(labels_dict)} entries")
