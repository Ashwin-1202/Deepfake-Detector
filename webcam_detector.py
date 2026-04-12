import cv2
import numpy as np
import time
from collections import Counter
from rppg_module import RPPGDetector
from fft_module import FFTDetector
from illumination_module import IlluminationDetector
from model import MultiModalClassifier
from utils import draw_results, DetectionLogger

class WebcamDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.rppg = RPPGDetector()
        self.fft = FFTDetector()
        self.illum = IlluminationDetector()
        self.classifier = MultiModalClassifier()
        self.logger = DetectionLogger()
        self.fps_counter = 0
        self.start_time = time.time()
        self.score_history = []
        self.label_history = []  # 1=REAL, 0=FAKE
        
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def run(self):
        print("🖥️  Deepfake Detector - REAL or FAKE only!")
        print("📊 Face forward → Watch for REAL (green 0.7+)")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            faces = self.detect_faces(frame)
            scores = [0.45, 0.50, 0.55]
            label = "FAKE"
            confidence = 0.40
            
            if len(faces) > 0:
                # Best face
                best_idx = np.argmax(faces[:, 2] * faces[:, 3])
                best_face = tuple(faces[best_idx])
                x, y, w, h = best_face
                face_roi = frame[y:y+h, x:x+w]
                
                # Scores
                scores[0] = self.rppg.update(face_roi)
                scores[1] = self.illum.update(face_roi)
                scores[2] = self.fft.update(face_roi)
                
                # Classify + SMOOTH
                raw_conf, raw_label = self.classifier.simple_classify(*scores)
                
                self.score_history.append(raw_conf)
                self.label_history.append(1 if raw_label == "REAL" else 0)
                
                if len(self.score_history) > 10:
                    self.score_history.pop(0)
                    self.label_history.pop(0)
                
                confidence = np.mean(self.score_history)
                real_votes = np.mean(self.label_history)
                label = "REAL" if real_votes > 0.5 else "FAKE"
                
                # Draw
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                for face in faces:
                    if not np.array_equal(face, best_face):
                        fx, fy, fw, fh = face
                        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            
            # FPS
            self.fps_counter += 1
            if time.time() - self.start_time > 1.0:
                fps = self.fps_counter / (time.time() - self.start_time)
                self.fps_counter = 0
                self.start_time = time.time()
            else:
                fps = 28.0
            
            frame = draw_results(frame, scores, label, confidence, fps)
            cv2.imshow('🛡️ Deepfake Detector - REAL or FAKE', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()