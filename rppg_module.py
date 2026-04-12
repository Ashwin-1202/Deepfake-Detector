import numpy as np
from scipy import signal
import cv2

class RPPGDetector:
    def __init__(self, fps=30, window_size=150):
        self.fps = fps
        self.window_size = window_size
        self.buffer = []
        
    def extract_frontal_roi(self, face_roi):
        """Extract forehead + cheeks (frontal only)"""
        if face_roi.shape[0] < 40 or face_roi.shape[1] < 40:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        
        h, w = face_roi.shape[:2]
        roi_w, roi_h = min(20, w//4), min(15, h//5)
        
        # Safe bounds
        fh = min(roi_h, h)
        fw = min(w//2, w)
        lh = min(roi_h, h-roi_h)
        lw = min(roi_w, w)
        rw = min(roi_w, w)
        
        forehead = face_roi[:fh, (w-fw)//2:w-(w-fw)//2]
        left_cheek = face_roi[roi_h:roi_h+lh, :lw]
        right_cheek = face_roi[roi_h:roi_h+lh, w-rw:w]
        
        # Resize to match
        target = (max(lw, 16), max(lh, 12))
        forehead = cv2.resize(forehead, target)
        left_cheek = cv2.resize(left_cheek, target)
        right_cheek = cv2.resize(right_cheek, target)
        
        # Symmetry check
        diff = np.mean(np.abs(left_cheek.astype(np.float32) - right_cheek.astype(np.float32)))
        if diff > 50:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        
        combined = np.vstack([forehead, left_cheek, right_cheek])
        return cv2.resize(combined, (48, 48))
    
    def bandpass_filter(self, signal_data, low=0.7, high=4.0, fs=30):
        """Safe bandpass filter"""
        if len(signal_data) < 30:  # FIXED: Minimum length
            return signal_data * 0  # Zero signal
        
        nyquist = fs / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Safe filter order
        order = min(4, len(signal_data) // 10)
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        
        # Pad if needed
        padlen = max(len(b)-1, 3*order)
        if len(signal_data) <= padlen:
            return np.zeros_like(signal_data)
        
        try:
            return signal.filtfilt(b, a, signal_data)
        except:
            return signal_data * 0
    
    def extract_pulse_signal(self, face_roi):
        roi = self.extract_frontal_roi(face_roi)
        return float(np.mean(roi[:, :, 1])) if roi.size > 0 else 128.0
    
    def compute_heart_signal_features(self):
        """STABLE scoring - real humans get 0.7+"""
        if len(self.buffer) < 15:
            return 0.45  # Fast warmup
        
        # Smooth recent signal
        recent = np.array(self.buffer[-60:])
        trend = np.polyfit(range(len(recent)), recent, 1)[0]  # Remove trend
        detrended = recent - trend * np.arange(len(recent))
        
        # Always attempt filter
        filtered = self.bandpass_filter(detrended)
        
        # Relaxed peak detection
        peaks, _ = signal.find_peaks(filtered, prominence=0.15, distance=5)
        
        base_score = 0.5  # Human baseline
        
        if len(peaks) >= 2:
            intervals = np.diff(peaks)
            if len(intervals) > 0:
                hrv = np.std(intervals) / np.mean(intervals)
                periodicity = max(0.6, 1.0 / (1.0 + hrv * 0.3))
                base_score = 0.75 + 0.2 * periodicity
        
        # Signal quality
        snr = np.std(filtered) / (np.std(detrended - filtered) + 1e-7)
        quality = min(0.3, np.tanh(snr * 0.4))
        
        score = base_score + quality
        return min(0.98, max(0.4, score))  # Stable 0.4-0.98 range
    
    def update(self, face_roi):
        pulse = self.extract_pulse_signal(face_roi)
        self.buffer.append(pulse)
        if len(self.buffer) > self.window_size * 2:
            self.buffer = self.buffer[-self.window_size*2:]
        return self.compute_heart_signal_features()