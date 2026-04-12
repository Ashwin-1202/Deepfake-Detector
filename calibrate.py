from model import MultiModalClassifier
import numpy as np

classifier = MultiModalClassifier()

# Test your scores
rppg, light, fft = 0.65, 0.58, 0.62  # Replace with yours
score, label = classifier.simple_classify(rppg, light, fft)
print(f"Your scores → {label} ({score:.3f})")