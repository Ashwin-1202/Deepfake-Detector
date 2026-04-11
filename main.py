import argparse
import sys
from webcam_detector import WebcamDetector
from video_detector import VideoDetector  # ADD THIS LINE
from utils import DetectionLogger

def main():
    parser = argparse.ArgumentParser(description='🛡️ Real-Time Deepfake Detector')
    parser.add_argument('--webcam', action='store_true', help='Run webcam detection')
    parser.add_argument('--video', type=str, help='Analyze video file')
    parser.add_argument('--test', action='store_true', help='Run demo')
    
    args = parser.parse_args()
    
    if args.webcam:
        detector = WebcamDetector()
        detector.run()
    
    elif args.video:
        detector = VideoDetector()  # FIXED
        detector.analyze_video(args.video)
    
    elif args.test:
        logger = DetectionLogger()
        print("✅ Test OK - Check detection_log.json")
    
    else:
        print("🤖 Usage:")
        print("  python main.py --webcam")
        print("  python main.py --video myvideo.mp4")
        print("  python main.py --test")

if __name__ == "__main__":
    main()