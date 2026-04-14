# VisionBooth

A contactless, gesture- and voice-controlled photo booth built with Python, Flask, MediaPipe, and Web Speech Recognition.

---

## Overview

VisionBooth is a fully touchless photo booth experience. Users control every step — from setting a countdown timer to selecting a template and filter — using only hand gestures and voice commands. No buttons, no touchscreens.

---

## Features

- **Gesture-based timer setting** — Hold up 1–5 fingers to set a countdown timer (1–5 seconds)
- **Voice command control** — Say "START" to begin a countdown; say "YES"/"NO", "CLASSIC"/"FUN"/"ELEGANT", "NORMAL"/"BLACK AND WHITE" at the appropriate steps
- **Automatic 4-photo strip capture** — Countdown repeats automatically between each of the 4 shots
- **Photo review / retake** — After capturing, users can choose to retake all photos via voice
- **Template selection** — Three photo strip designs: Classic, Fun, Elegant
- **Filter selection** — Normal (colour) or Black & White
- **Strip compositing** — Final photo strip is generated server-side using PIL, overlaying photos into template slots with branding
- **Downloadable output** — Final strip is saved as a high-resolution PNG and offered for download

---

## Project Structure

```
visionbooth/
├── app.py                  # Flask + SocketIO server, state machine, event handlers
├── gesture_detector.py     # MediaPipe hand tracking and gesture classification
├── speech_handler.py       # SpeechRecognition microphone listener and TTS
├── filter_processor.py     # PIL-based photo compositing and filter application
├── finger_counter.py       # Utility: finger counting from hand landmarks
├── functions.py            # Utility: distance, angle, and gesture helper functions
├── delete_blured.py        # Utility: blur detection and removal using Laplacian variance
├── main.py                 # Standalone OpenCV desktop version (non-web)
├── requirements.txt        # Python dependencies
├── templates/              # HTML pages (Jinja2)
│   ├── home.html
│   ├── guide.html
│   ├── privacy.html
│   ├── index.html              # Main photobooth camera page
│   ├── retake.html
│   ├── template_selection.html
│   ├── filter_selection.html
│   └── final_preview.html
├── static/
│   └── images/
│       ├── VB_1.png
│       ├── template_1.png      # Classic template
│       ├── template_2.png      # Fun template
│       └── template_3.png      # Elegant template
└── sessions/               # Auto-created; stores captured photos and final strips
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Node.js (optional, not required for running)
- A working webcam and microphone
- Internet connection (for Google Speech Recognition API)

### Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- `flask`, `flask-socketio` — web server and real-time communication
- `opencv-python` — webcam capture and image processing
- `mediapipe` — hand landmark detection
- `SpeechRecognition` — voice command recognition (uses Google Speech API)
- `pyttsx3` — text-to-speech feedback
- `Pillow` — photo strip compositing
- `numpy` — image array operations

### Run the app

```bash
python app.py
```

Then open your browser at: [http://localhost:5000](http://localhost:5000)

---

## How to Use

| Step | Action |
|------|--------|
| 1 | Hold up **1–5 fingers** for 5 consecutive frames to set the countdown timer |
| 2 | Say **"START"** (or show a sustained thumbs up) to begin the countdown |
| 3 | Smile and hold still — the booth captures **4 photos** automatically |
| 4 | Say **"YES"** to retake all photos, or **"NO"** to continue |
| 5 | Say **"CLASSIC"**, **"FUN"**, or **"ELEGANT"** to pick your strip design |
| 6 | Say **"NORMAL"** or **"BLACK AND WHITE"** to apply a filter |
| 7 | Download your finished photo strip |

> **Tip:** Say "START" instead of relying on the thumbs-up gesture — it is faster and more reliable.  
> **Tip:** A closed **fist** gesture resets the timer selection at any point during countdown setup.

---

## Architecture

### State Machine (Camera Page)

The camera page runs a server-side state machine with the following states:

```
TIMER_DETECT → TIMER_DETECTING → TIMER_SET → TIMER_READY → COUNTDOWN → CAPTURE
```

- Each video frame is sent from the browser to the server via Socket.IO
- MediaPipe detects hand gestures on a downscaled frame (320px wide) every 3 frames for performance
- The full-resolution frame is captured client-side when the state reaches `CAPTURE`

### Voice Listening

- A background thread (`_camera_voice_listener`) listens for "START" while the camera is in `TIMER_READY` state
- All other pages (retake, template, filter) use dedicated Socket.IO events to spawn listener threads
- Microphone access is serialised through `mic_lock` to prevent concurrent recording conflicts
- A monotonic listener ID (`_listener_id`) invalidates stale threads when a new page loads

### Photo Compositing

- `filter_processor.py` detects photo slot regions in a template image by scanning for rows with high colour saturation (the decorative borders between slots)
- Each photo is centre-cropped to the slot's aspect ratio, then pasted onto a white canvas
- The template PNG is overlaid on top (requires transparent areas in the template) to preserve borders and branding
- Branding text (`VisionBooth MM/DD/YY`) is drawn at the bottom of the strip

---

## Configuration

Key constants in `app.py`:

| Constant | Default | Description |
|---|---|---|
| `PHOTOS_PER_STRIP` | `4` | Number of photos per session |
| `CONSECUTIVE_REQUIRED` | `5` | Frames a gesture must be held to register |
| `GESTURE_EVERY_N_FRAMES` | `3` | Run gesture detection every N frames |
| `GESTURE_DETECT_WIDTH` | `320` | Downscale width for gesture detection |
| `STREAM_JPEG_QUALITY` | `60` | JPEG quality for annotated frame stream |

---

## Known Limitations

- Voice recognition requires an active internet connection (Google Speech API)
- Lighting conditions significantly affect gesture detection accuracy
- Only one user session is supported at a time (global server-side state)
- The microphone and camera must be on the same machine as the server
- Template slot detection depends on the template having clearly saturated colour borders

---

## License

This project was developed as an academic prototype. All rights reserved by the project authors.
