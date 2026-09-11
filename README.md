# Smart Traffic Management System

Computer Vision-Based Smart Traffic Congestion Analytics with Adaptive Signal Optimization.

This Group Design Project combines computer vision, traffic analytics, emergency-vehicle detection, adaptive signal control, microcontroller hardware and a Streamlit dashboard in one prototype.

## Main features

- **YOLO11m vehicle detection and tracking**
- **Vehicle counting and traffic-density estimation**
- **Real km/h speed estimation** using road calibration
- **Adaptive traffic-light control** with safe red/yellow/green transitions
- **Emergency-vehicle priority support** using flashing-beacon detection and vehicle recognition
- **Camera, video-file and network-stream inputs**
- **Arduino / ESP32 hardware integration**
- **Streamlit monitoring dashboard**
- **CSV/XLSX traffic logging and analytics**
- **Simulation mode** for testing without physical hardware

## System overview

```text
Camera / Video / Stream
        |
        v
Computer Vision Node
YOLO11m -> Tracking -> Speed -> Congestion -> Emergency Detection
        |
        +----> Traffic Signal / Hardware Control
        |
        +----> Logs and Analytics
        |
        v
Streamlit Dashboard
```

The CV node performs the real-time processing. The dashboard connects to it over the local network and displays traffic conditions, detections, signal status, emergency state, hardware health and analytics.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r SmartTrafficSystem/requirements.txt
```

For NVIDIA GPU acceleration, install the correct CUDA-enabled PyTorch build before installing the remaining packages.

### 2. Start the complete system

On Windows, double-click:

```text
START HERE.bat
```

or run:

```bash
python start.py
```

This starts both the CV node and the Streamlit dashboard.

Useful examples:

```bash
python start.py --source road.mp4
python start.py --source rtsp://...
python start.py --no-arduino
python start.py --no-display
```

Use `Ctrl+C` to shut the system down cleanly.

## Running the components separately

CV node:

```bash
python broadcast_server.py
```

Dashboard:

```bash
cd SmartTrafficSystem
streamlit run app.py
```

Default services:

- Dashboard: **port 8501**
- CV node / telemetry stream: **port 8502**

The source can also be changed while the system is running from the dashboard.

## Traffic-signal behaviour

The controller uses a state machine with committed yellow transitions and minimum phase timing to prevent unsafe or unstable rapid switching.

The project supports two signal interpretations:

- **Merge mode** — controls side-road traffic entering a main road according to congestion.
- **Junction mode** — controls the measured approach directly.

Emergency detection does not force an unsafe instant signal change. The normal transition logic remains active while the system provides emergency-priority indication through the hardware outputs.

## Project structure

```text
smart-traffic-system/
├── start.py                     # Starts the complete system
├── broadcast_server.py          # CV node and HTTP interface
├── traffic_vision.py            # Detection, tracking, speed and congestion
├── traffic_light.py             # Signal state machine
├── video_source.py              # Camera, file and stream handling
├── siren_vision.py              # Flashing-beacon detection
├── emergency_vision.py          # Emergency-vehicle recognition
├── calibrate_speed.py           # Camera speed calibration
├── yolo11m.pt                   # Main YOLO model
├── emergency_cls.pt             # Emergency classification model
├── SmartTrafficSystem/          # Streamlit dashboard and backend
├── test_*.py                    # Automated software checks
├── METRICS.md                   # Measured performance and methodology
└── requirements.txt             # CV-node dependencies
```

## Testing

The repository includes automated checks for the traffic pipeline, dashboard, analytics, video sources, server integration and traffic-light state machine.

Examples:

```bash
python test_traffic_light.py
python test_traffic_system.py
python test_analytics.py
```

## Models and generated data

The repository keeps the two project model files:

- `yolo11m.pt`
- `emergency_cls.pt`

Large video footage, runtime databases, local settings, session logs and generated outputs are intentionally excluded from Git.

## Documentation

For more detail:

- `METRICS.md` — measured results and how they were obtained
- `HOW_THE_WHOLE_SYSTEM_WORKS.txt` — complete system explanation
- `HOW_EMERGENCY_DETECTION_WORKS.txt` — emergency-detection explanation
- `SmartTrafficSystem/PROJECT_SPEC.md` — dashboard/system specification

## Project

**Programme:** Group Design Project  
**Theme:** Smart City & AI Traffic Optimization  
**Project:** Computer Vision-Based Smart Traffic Congestion Analytics with Adaptive Signal Optimization
