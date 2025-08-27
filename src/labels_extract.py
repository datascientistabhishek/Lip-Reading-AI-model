import json
from pathlib import Path

# Paths
DATA_DIR = Path(r"D:\lip_reading_ai\data")   # apna dataset path
OUTPUT_JSON = Path(r"D:\lip_reading_ai\labels.json")

def extract_labels(data_dir):
    labels = {}

    # 🔹 Numeric sorting of speaker folders (s1 → s20)
    speakers = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.name[1:])
    )

    for speaker in speakers:
        align_folder = speaker / "align"
        align_files = sorted(align_folder.glob("*.align"))

        for af in align_files:
            vid_name = af.stem  # filename without extension
            key = f"{speaker.name}_{vid_name}"

            with open(af, "r") as f:
                words = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:  # start, end, word
                        word = parts[2]
                        if word.lower() != "sil":  # Ignore silence tokens
                            words.append(word)

            # Save final transcript for this video
            labels[key] = " ".join(words)

    return labels


if __name__ == "__main__":
    labels_dict = extract_labels(DATA_DIR)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)

    print(f"✅ Labels saved to {OUTPUT_JSON} with {len(labels_dict)} entries")
