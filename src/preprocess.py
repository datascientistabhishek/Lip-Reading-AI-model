import cv2
import mediapipe as mp
import os
from pathlib import Path
from tqdm import tqdm

# Mediapipe FaceMesh init
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

# Lip landmark indices (468 total points, lips are around 61 points)
LIPS_IDX = list(set([
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 415, 310,
    311, 312, 13, 82, 81, 42, 183, 78,
    191, 80, 81, 82, 13, 312, 311, 310
]))

def extract_lip_frames(video_path, save_dir, img_size=96):
    """Extracts lip region frames from a video and saves them."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                xs = [int(face_landmarks.landmark[i].x * w) for i in LIPS_IDX]
                ys = [int(face_landmarks.landmark[i].y * h) for i in LIPS_IDX]

                x_min, x_max = max(min(xs) - 5, 0), min(max(xs) + 5, w)
                y_min, y_max = max(min(ys) - 5, 0), min(max(ys) + 5, h)

                lip_img = frame[y_min:y_max, x_min:x_max]
                lip_img = cv2.resize(lip_img, (img_size, img_size))

                save_path = save_dir / f"{frame_count:04d}.jpg"
                cv2.imwrite(str(save_path), lip_img)
                frame_count += 1
        else:
            # अगर face detect नहीं हुआ तो original frame skip कर देंगे
            pass

    cap.release()


def process_all_videos(data_dir, frames_dir):
    """Process all speakers' videos into lip frames."""
    data_dir = Path(data_dir)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    speakers = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for speaker in speakers:
        video_folder = speaker / "video"
        video_files = sorted(video_folder.glob("*.mpg"))

        for video_file in tqdm(video_files, desc=f"Processing {speaker.name}"):
            vid_name = video_file.stem
            save_path = frames_dir / speaker.name / vid_name
            save_path.mkdir(parents=True, exist_ok=True)
            extract_lip_frames(video_file, save_path)


if __name__ == "__main__":
    DATA_DIR = r"D:\lip_reading_ai\data"    # dataset ka absolute path
    FRAMES_DIR = r"D:\lip_reading_ai\frames"

    process_all_videos(DATA_DIR, FRAMES_DIR)
    print("✅ Lip frames extraction complete!")
