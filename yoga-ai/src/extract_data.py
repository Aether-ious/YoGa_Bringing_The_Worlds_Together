import cv2
import mediapipe as mp
import numpy as np
import os

# --- CONFIG ---
DATA_PATH = "training_data"
OUTPUT_FOLDER = "model_data"
SEQUENCE_LENGTH = 30 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def extract_keypoints(results):
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Flatten x, y, z, visibility
        pose_row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in lm]).flatten())
    else:
        pose_row = [0] * 132
    return np.array(pose_row)

def get_video_files():
    """Finds all video files and extracts labels, handling different separators."""
    video_files = []
    all_labels = set()

    # Walk through folder
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                
                # FIX: Check for both standard pipe (|) and fullwidth pipe (｜)
                separator = None
                if "|" in file:
                    separator = "|"
                elif "｜" in file:  # <--- The special character from your filenames
                    separator = "｜"
                
                if separator:
                    label = file.split(separator)[0].strip()
                    full_path = os.path.join(root, file)
                    
                    video_files.append((full_path, label))
                    all_labels.add(label)
                else:
                    print(f"⚠️ Skipping '{file}': Separator not found.")
    
    return video_files, sorted(list(all_labels))

def process_dataset():
    sequences, labels = [], []
    
    # 1. Find Files and Classes
    files, unique_labels = get_video_files()
    
    if not files:
        print(f"❌ No valid videos found in '{DATA_PATH}'.")
        return

    print(f"📂 Found {len(files)} videos across {len(unique_labels)} classes: {unique_labels}")
    
    # Map text label to number
    label_map = {label: num for num, label in enumerate(unique_labels)}

    # 2. Process Each Video
    for file_path, label_text in files:
        cap = cv2.VideoCapture(file_path)
        frames = []
        
        # Get Label ID
        label_id = label_map[label_text]
        
        print(f"   🎬 Processing: {label_text}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # MediaPipe Processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            # Extract
            keypoints = extract_keypoints(results)
            frames.append(keypoints)
            
            # Create Sequence
            if len(frames) == SEQUENCE_LENGTH:
                sequences.append(np.array(frames))
                labels.append(label_id)
                
                # Slide window
                frames = frames[5:] 
        
        cap.release()

    # 3. Save Data
    if len(sequences) == 0:
        print("❌ No sequences extracted. Videos might be too short or MediaPipe failed to detect bodies.")
        return

    X = np.array(sequences)
    y = np.array(labels)
    
    np.save(f'{OUTPUT_FOLDER}/X.npy', X)
    np.save(f'{OUTPUT_FOLDER}/y.npy', y)
    np.save(f'{OUTPUT_FOLDER}/labels.npy', unique_labels)
    
    print(f"\n🎉 Done! Extracted {len(X)} sequences.")
    print(f"💾 Saved to '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    process_dataset()