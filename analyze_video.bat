@echo off
echo 🛡️ Drag video file here and press ENTER
set /p video_path="Video path: "
python video_detector.py "%video_path%"
pause