import numpy as np
import cv2
import time

class IlluminationDetector:
    def __init__(self):
        self.phase = 0
        self.response_buffer = []
        self.baseline = None
        
    def generate_flash_pattern(self):
        t = time.time()
        self.phase = int(100 + 100 * np.sin(2 * np.pi * 0.5 * t))
        return self.phase
    
    def measure_face_response(self, face_roi):
        if face_roi.size == 0:
            return 0.5
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        intensity = np.mean(gray_roi)
        
        if self.baseline is None:
            self.baseline = intensity
            return 0.5
        
        response = abs((intensity - self.baseline) / 255.0 - self.phase/255.0)
        correlation = 1.0 / (1.0 + response)
        
        self.response_buffer.append(correlation)
        if len(self.response_buffer) > 30:
            self.response_buffer = self.response_buffer[-30:]
        return np.mean(self.response_buffer)
    
    def update(self, face_roi):
        self.generate_flash_pattern()
        return self.measure_face_response(face_roi)