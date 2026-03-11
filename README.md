# 🏗️ AI-Augmented Smart Safety Helmet

A **real-time falling object detection system** for construction worker safety. Uses an ESP32-CAM mounted on a helmet to detect objects falling toward the wearer and trigger immediate audio/visual alerts.

> **No AI model needed.** Uses motion-based detection that works with **any** falling object — bricks, tools, debris, phones, anything.

---

## How It Works

```
ESP32-CAM Stream → Background Subtraction (MOG2) → Contour Detection → Centroid Tracking → Velocity Calculation → Fall Decision → Alert
```

### Detection Pipeline

| Stage | What it does |
|-------|-------------|
| **1. MJPEG Stream** | Custom `urllib` HTTP reader pulls frames from ESP32-CAM WiFi stream. Crash-proof (no ffmpeg dependency). |
| **2. Background Subtraction** | OpenCV MOG2 learns the static background. Any moving region becomes foreground. |
| **3. Contour Filtering** | Moving regions are filtered by area to ignore tiny noise and full-frame shifts. |
| **4. Centroid Tracking** | Each blob is tracked across frames by nearest-neighbor position matching. No class labels needed. |
| **5. Velocity Calculation** | Vertical displacement (ΔY) per frame is calculated for each tracked blob. |
| **6. Ego-Motion Compensation** | Sparse optical flow (Lucas-Kanade) estimates camera movement. Camera motion is **subtracted** from object velocity so head movement doesn't trigger false alerts. |
| **7. Fall Decision** | If an object's **adjusted** downward velocity exceeds the threshold for N consecutive frames → **DANGER!** |

### Ego-Motion Compensation

Since the camera is mounted on a **moving helmet**, plain background subtraction would trigger false alerts constantly. Two layers prevent this:

1. **Optical Flow** — Tracks ~200 feature points to compute global camera motion vector. This gets subtracted from every tracked object's velocity.
2. **Global Motion Ratio** — If >25% of the frame is foreground, the entire scene is shifting (camera pan). Fall detection pauses and MOG2 adapts 10x faster.

---

## Setup

### Hardware
- **ESP32-CAM** (AI-Thinker or similar) running the CameraWebServer example sketch
- PC/Laptop on the same WiFi network

### Software

```bash
# Clone the repo
git clone https://github.com/Cave-Man-yt/Smart-Helmet-MDP.git
cd Smart-Helmet-MDP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### ESP32-CAM Configuration
1. Flash the **CameraWebServer** example from Arduino IDE
2. Set resolution to **VGA (640×480)** in the ESP32 web UI
3. Note the IP address shown in Serial Monitor

### Update Stream URL
In `esp32_fall_detection.py`, update line 47:
```python
STREAM_URL = "http://<YOUR_ESP32_IP>:81/stream"
```

---

## Usage

```bash
python esp32_fall_detection.py
```

### Runtime Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Toggle debug mode (shows motion mask window) |
| `S` | Cycle sensitivity: **LOW** → **MEDIUM** → **HIGH** |

### Sensitivity Presets

| Preset | Min Blob Area | Fall Velocity | Consecutive Frames |
|--------|--------------|---------------|-------------------|
| **LOW** | 3000 px² | 25 px/frame | 3 |
| **MEDIUM** | 1500 px² | 15 px/frame | 2 |
| **HIGH** | 800 px² | 8 px/frame | 2 |

### Visual Feedback

- 🟢 **Green box** — Object tracked, normal movement
- 🟠 **Orange box** — Downward velocity detected, building fall count
- 🔴 **Red box** — Fall alert imminent
- 🔴 **Red overlay + "DANGER!"** — Fall confirmed, alert active
- 🟡 **"CAMERA MOVING"** — Ego-motion detected, fall detection paused

---

## Project Structure

```
Smart-Helmet-MDP/
├── esp32_fall_detection.py   # Main detection script
├── requirements.txt          # Python dependencies (opencv, numpy, pygame)
├── README.md                 # ← This file
├── .agent/context.md         # Agent context documentation
└── .gitignore
```

---

## Key Parameters

All tunable in the `CONFIGURATION` section at the top of `esp32_fall_detection.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FALL_VELOCITY_THRESHOLD` | 15.0 | Minimum downward px/frame to flag as falling |
| `MIN_CONSECUTIVE_FRAMES` | 2 | Frames above threshold before alerting |
| `MIN_CONTOUR_AREA` | 1500 | Ignore moving blobs smaller than this (noise) |
| `MAX_CONTOUR_AREA` | 200000 | Ignore blobs larger than this |
| `LEARNING_RATE` | 0.005 | How fast background model adapts |
| `MAX_TRACKING_DISTANCE` | 200 | Max px to match a blob between frames |
| `CAMERA_MOTION_RATIO_THRESHOLD` | 0.25 | Foreground % that triggers camera-motion mode |
| `ALERT_COOLDOWN` | 3.0 | Seconds between successive alerts |

---

## Future Roadmap

- [ ] **Accelerometer integration** — ESP32 IMU data for precise ego-motion subtraction
- [ ] **Buzzer/speaker output** — On-helmet alert instead of laptop-only
- [ ] **ESP32 board control** — Direct GPIO for hardware alerts
- [ ] **Multi-zone detection** — Only alert for objects in the "danger zone" above the wearer

---

## License

This project is part of a Multidisciplinary Project (MDP) for academic purposes.
