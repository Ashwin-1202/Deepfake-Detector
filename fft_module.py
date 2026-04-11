import numpy as np
import cv2
from scipy.fft import fft2, fftshift

class FFTDetector:
    def __init__(self, window_size=32):
        self.window_size = window_size
        self.fft_buffer = []
    
    def compute_fft_features(self, gray_frame):
        if gray_frame.shape[0] < 8:
            return 0.6
        
        h, w = gray_frame.shape
        crop_size = min(h, w, self.window_size, 64)
        
        start_h, end_h = h//2-crop_size//2, h//2+crop_size//2
        start_w, end_w = w//2-crop_size//2, w//2+crop_size//2
        
        cropped = gray_frame[start_h:end_h, start_w:end_w]
        if cropped.size < 16:
            return 0.6
        
        ch, cw = cropped.shape
        pad_size = max(ch, cw)
        padded = np.zeros((pad_size, pad_size))
        padded[:ch, :cw] = cropped
        
        fft_img = fft2(padded)
        magnitude = np.log(np.abs(fftshift(fft_img)) + 1)
        
        center = magnitude.shape[0] // 2
        high_freq_mask = np.zeros_like(magnitude, dtype=bool)
        high_freq_mask[center//2:, center//2:] = True
        
        high_energy = np.mean(magnitude[high_freq_mask])
        low_energy = np.mean(magnitude[:center//2, :center//2])
        
        ratio = high_energy / (low_energy + 1e-8)
        periodicity = np.std(np.max(np.abs(fft_img), axis=1)) / np.mean(np.abs(fft_img))
        
        fake_score = np.tanh(ratio * 0.3 + periodicity * 0.2)
        return 1.0 - min(1.0, fake_score)
    
    def update(self, face_roi):
        if face_roi.size == 0:
            return 0.6
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        score = self.compute_fft_features(gray)
        self.fft_buffer.append(score)
        if len(self.fft_buffer) > 30:
            return np.mean(self.fft_buffer[-30:])
        return score