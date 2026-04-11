import numpy as np

class MultiModalClassifier:
    def __init__(self):
        pass
    
    def simple_classify(self, rppg_score, light_score, fft_score):
        """HUMAN FIRST - Real people = REAL"""
        
        # HUMAN BOOST (0.4+ rPPG = almost guaranteed REAL)
        rppg_boost = max(0, (rppg_score - 0.3) * 2.0)  # Heavy bias
        light_boost = max(0, (light_score - 0.3) * 1.5)
        fft_boost = max(0, (fft_score - 0.3) * 1.2)
        
        # Weighted (rPPG dominates)
        score = 0.6 * (rppg_score + rppg_boost) + \
                0.2 * (light_score + light_boost) + \
                0.2 * (fft_score + fft_boost)
        
        # HUMAN THRESHOLD - Very generous
        label = "REAL" if score > 0.35 else "FAKE"  # 0.35!
        confidence = min(0.99, score * 1.2)
        
        return confidence, label