# Hand Gesture Controller

A computer vision application that translates hand gestures into keyboard and mouse commands using MediaPipe and OpenCV.

## 🚀 Features

- Real-time hand gesture recognition
- Multi-threaded processing for smooth performance
- Configurable gesture sensitivity
- Visual feedback and debugging UI
- 7+ supported gestures including swipes and poses

## 🎮 Supported Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| Open Palm | Jump (Space) | All five fingers extended |
| Fist | Duck (Down) | Closed hand |
| Swipe Left/Right | Left/Right Arrow | Horizontal hand movement |
| Swipe Up/Down | Up/Down Arrow | Vertical hand movement |
| Point | Mouse Click | Index finger only |
| Victory | Double Click | Index and middle fingers |
| Thumbs Up | Enter | Thumb only |

## 📋 Requirements

```
opencv-python>=4.7.0
mediapipe>=0.10.0
pyautogui>=0.9.53
numpy>=1.22.0
```

## 💻 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/hand-gesture-controller.git
cd hand-gesture-controller

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 🔧 Configuration

Edit these variables in `main.py` to customize:

```python
# Performance settings
self.frame_width = 640      # Camera resolution width
self.frame_height = 480     # Camera resolution height
self.process_fps = 15       # Frames processed per second
self.debounce_time = 0.3    # Time between gesture recognition
self.swipe_threshold = 40   # Pixels required for swipe detection
```

## 🔍 How It Works

1. **WebcamCaptureThread** continuously captures frames from the webcam
2. **GestureController** processes frames to detect hand landmarks using MediaPipe
3. Hand positions are analyzed to recognize specific gestures
4. Recognized gestures trigger keyboard/mouse commands via PyAutoGUI

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Webcam Feed  │ ──► │  Hand Detect  │ ──► │   Gesture     │ ──► │  Keyboard/    │
│               │     │  (MediaPipe)  │     │  Recognition  │     │  Mouse Action │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```


