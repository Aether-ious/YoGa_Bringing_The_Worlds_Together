import cv2
import mediapipe as mp
import numpy as np
import torch
import os
import sys

# Import Model Architecture
sys.path.append(os.path.dirname(__file__))
from model import YogaLSTM

class YogaPoseAnalyzer:
    def __init__(self, mode="geometry"):
        self.mode = mode
        
        # MediaPipe Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # AI Model Setup
        self.sequence = [] # Buffer to hold 30 frames
        self.model = None
        self.labels = []
        
        if self.mode == "ai":
            self.load_ai_model()

    def load_ai_model(self):
        """Loads the trained PyTorch LSTM."""
        model_path = "models/yoga_lstm.pth"
        labels_path = "model_data/labels.npy"
        
        if os.path.exists(model_path) and os.path.exists(labels_path):
            try:
                self.labels = np.load(labels_path)
                self.model = YogaLSTM(num_classes=len(self.labels))
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"✅ AI Loaded. Classes: {self.labels}")
            except Exception as e:
                print(f"❌ Error loading AI: {e}")
        else:
            print("⚠️ AI Model not found. Run 'train_model.py' first.")

    def calculate_angle(self, a, b, c):
        """Geometry Math Helper"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return angle

    def process_frame(self, image):
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image.flags.writeable = True
        
        pose_name = "Analyzing..."
        feedback = "Hold steady"
        color = (200, 200, 200)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Draw Skeleton
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            # --------------------------
            # MODE 1: GEOMETRY (RULES)
            # --------------------------
            if self.mode == "geometry":
                def get_p(name): 
                    p = lm[self.mp_pose.PoseLandmark[name].value]
                    return [p.x, p.y]

                try:
                    # Extract needed points
                    l_hip, l_knee, l_ankle = get_p('LEFT_HIP'), get_p('LEFT_KNEE'), get_p('LEFT_ANKLE')
                    r_hip, r_knee, r_ankle = get_p('RIGHT_HIP'), get_p('RIGHT_KNEE'), get_p('RIGHT_ANKLE')
                    l_shldr, l_elbow, l_wrist = get_p('LEFT_SHOULDER'), get_p('LEFT_ELBOW'), get_p('LEFT_WRIST')
                    
                    # Calculate basic angles
                    knee_ang = self.calculate_angle(l_hip, l_knee, l_ankle)
                    arm_ang = self.calculate_angle(l_shldr, l_elbow, l_wrist)

                    # Simple Example Logic
                    if knee_ang < 160:
                        pose_name = "Warrior II (Rule)"
                        if arm_ang < 160:
                            feedback = "Straighten Arm!"
                            color = (0, 0, 255)
                        else:
                            feedback = "Good Form"
                            color = (0, 255, 0)
                    else:
                        pose_name = "Standing"
                except:
                    pass

            # --------------------------
            # MODE 2: AI (LSTM)
            # --------------------------
            elif self.mode == "ai" and self.model:
                # 1. Extract Keypoints
                keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in lm]).flatten()
                self.sequence.append(keypoints)
                
                # Keep only last 30 frames
                self.sequence = self.sequence[-30:]
                
                # Predict if we have enough data
                if len(self.sequence) == 30:
                    with torch.no_grad():
                        input_tensor = torch.tensor([self.sequence], dtype=torch.float32)
                        prediction = self.model(input_tensor)
                        res = torch.softmax(prediction, dim=1)[0]
                        
                        # Get best class
                        best_idx = torch.argmax(res).item()
                        confidence = res[best_idx].item()
                        
                        if confidence > 0.7:
                            pose_name = f"{self.labels[best_idx]} (AI)"
                            feedback = f"Conf: {confidence:.2f}"
                            color = (0, 255, 0)
                        else:
                            pose_name = "Unsure..."

        # --- DRAW HUD ---
        cv2.rectangle(image, (0,0), (400, 80), (245, 117, 16), -1)
        cv2.putText(image, f"MODE: {self.mode.upper()}", (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, pose_name, (15,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, feedback, (250,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return image