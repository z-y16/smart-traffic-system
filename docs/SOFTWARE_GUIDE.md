# Software Guide

This repository contains the software for a computer-vision smart traffic analytics system. The software accepts camera, video-file, or network-stream input; detects and tracks vehicles; estimates traffic conditions and vehicle speed; recognises emergency vehicles; records analytics; and exposes the results through a Streamlit dashboard.

## Architecture

```text
Camera / Video / Stream
        |
        v
Video Source
        |
        v
YOLO11m Detection
        |
        v
BoT-SORT Tracking
        |
        +----> Speed Estimation
        +----> Vehicle Counting
        +----> Congestion Analysis
        +----> Emergency Recognition
        |
        v
CV / API Node
        |
        +----> Logs + Session Data
        +----> Traffic-Signal State
        |
        v
Streamlit Dashboard
```

The project is split into two main software processes:

- `broadcast_server.py` — computer-vision processing, video-source control, telemetry, logging, and the local HTTP interface.
- `SmartTrafficSystem/` — Streamlit dashboard, analytics, system state, settings, simulation services, and UI components.

`start.py` launches both processes together.

## Core processing pipeline

### Video sources

`video_source.py` provides a common interface for webcams, uploaded/recorded videos, and network streams. Recorded videos use media time rather than wall-clock processing time so speed estimates are not changed by how quickly the computer decodes the file.

### Vehicle detection and tracking

The default detector is `yolo11m.pt`. Detection is restricted to traffic-relevant vehicle classes and is paired with BoT-SORT tracking so each vehicle can be followed across frames. Tracking enables stable vehicle counts, speed measurement, and per-vehicle analytics instead of treating every frame independently.

### Speed estimation

`traffic_vision.py` estimates speed from tracked motion. The preferred mode uses `speed_calibration.json`, created with `calibrate_speed.py`, to map image coordinates to real road distance. Speed is measured over time windows rather than a single frame-to-frame displacement to reduce bounding-box jitter.

Synthetic ground-truth checks in the project verify the speed mathematics at known test speeds. Field accuracy still depends on camera placement and calibration quality.

### Congestion analysis

Congestion combines traffic occupancy and traffic speed rather than relying on vehicle count alone. The logic includes safeguards for empty roads and isolated slow vehicles, plus hysteresis/stability rules to avoid rapid state changes near thresholds.

### Emergency recognition

Emergency recognition combines two software methods:

1. Vehicle appearance recognition identifies likely ambulance, police, and fire-service vehicles.
2. `siren_vision.py` analyses temporal red/blue beacon behaviour and looks for repeated switched flashing rather than steady lights or gradual reflections.

These signals are kept separate because identifying an emergency-style vehicle and detecting an actively flashing beacon are different questions. The software stores both the classification and priority state.

### Traffic-signal logic

`traffic_light.py` contains the software state machine for red/yellow/green behaviour. Transitions use a committed yellow phase and minimum dwell times so rapid changes in congestion do not create unstable signal states.

This repository keeps the software state machine and its tests. Physical-controller firmware and mechanical prototype files are intentionally excluded.

## Dashboard

`SmartTrafficSystem/` provides the user interface. Its main software areas are:

- Operations dashboard
- Live camera and detections
- Traffic analytics
- Emergency-control state
- System logs
- Settings
- About/project information

The dashboard can operate with simulated data when external devices or the CV node are unavailable. This makes the UI and analytics independently testable.

## Important modules

| File / folder | Purpose |
|---|---|
| `start.py` | Starts the complete software stack |
| `broadcast_server.py` | CV node and local API |
| `traffic_vision.py` | Detection, tracking, speed, congestion, logging |
| `video_source.py` | Cameras, files, stream URLs, media timing |
| `traffic_light.py` | Adaptive signal state machine |
| `emergency_vision.py` | Emergency-vehicle appearance recognition |
| `siren_vision.py` | Flashing-beacon analysis |
| `SmartTrafficSystem/` | Streamlit dashboard and backend services |
| `trackers/` | Tracker configuration |
| `assets/` | Software test assets |
| `test_*.py` | Automated software tests |

## Models

The repository keeps the two model files needed by the current software:

- `yolo11m.pt` — primary vehicle detector
- `emergency_cls.pt` — trained emergency-classification model used by the project

The emergency pipeline also supports CLIP-based zero-shot recognition in the Python implementation.

## Performance notes

Measurements recorded during development on an RTX 4050 Laptop GPU with FP16 showed:

- YOLO11m at 1280 px: about **36.6 FPS** for the detector/tracker benchmark under the original benchmark conditions.
- A later same-model rerun measured about **31.5 FPS**, illustrating normal machine-state variation.
- Live end-to-end telemetry on the development setup measured about **27 FPS** on representative daylight footage.

Resolution had a larger effect on distant-vehicle yield than increasing model size. These are development measurements, not universal hardware guarantees.

## Testing

The software includes dedicated checks for:

- Detection/tracking and traffic calculations
- Traffic-light state transitions
- Video sources and media timing
- API/server behaviour
- Dashboard rendering and interactions
- Traffic analytics

Typical commands:

```bash
python test_traffic_light.py
python test_traffic_system.py
python test_video_source.py
python test_analytics.py
```

For a quick syntax check across the repository:

```bash
python -m compileall -q .
```

## Running the software

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r SmartTrafficSystem/requirements.txt
```

Start the complete stack:

```bash
python start.py
```

Examples:

```bash
python start.py --source road.mp4
python start.py --source rtsp://...
python start.py --no-display
```

Run the two software processes separately if needed:

```bash
python broadcast_server.py
```

and:

```bash
cd SmartTrafficSystem
streamlit run app.py
```

The default dashboard port is `8501`; the CV/API node uses `8502`.

## Repository policy

The repository is intentionally software-focused. It excludes physical-prototype drawings, Arduino/ESP32 firmware, servo sketches, coursework presentation files, generated runtime databases, session outputs, large local videos, and machine-specific tool configuration.
