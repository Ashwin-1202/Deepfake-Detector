import cv2
import numpy as np
import json
import time
from datetime import datetime

class DetectionLogger:
    def __init__(self, log_file="detection_log.json"):
        self.log_file = log_file
        self.logs = []
    
    def log_detection(self, video_path, scores, label, confidence):
        """Log detection with JSON-safe types"""
        # Convert numpy types to Python natives
        safe_scores = [float(s) for s in scores]
        safe_confidence = float(confidence)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "video_path": str(video_path),
            "rppg_score": safe_scores[0],
            "illumination_score": safe_scores[1],
            "fft_score": safe_scores[2],
            "final_score": safe_confidence,
            "label": str(label)
        }
        self.logs.append(log_entry)
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"⚠️ Log save failed: {e}")

def draw_results(frame, scores, label, confidence, fps):
    """Draw detection results on frame"""
    h, w = frame.shape[:2]
    
    # Background overlay
    overlay = frame.copy()
    color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
    
    # Title
    cv2.rectangle(overlay, (10, 10), (w-10, 80), color, -1)
    cv2.putText(overlay, f"{label} ({confidence:.2f})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Scores
    y_offset = 100
    score_names = ["rPPG", "Light", "FFT"]
    for i, (name, score) in enumerate(zip(score_names, scores)):
        cv2.putText(overlay, f"{name}: {float(score):.2f}", (20, y_offset + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(overlay, f"FPS: {fps:.1f}", (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend overlay
    cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
    return frame

def preprocess_face_roi(roi, size=(64, 64)):
    """Preprocess face ROI for analysis"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    return gray.astype(np.float32) / 255.0