You are a senior Software Architect, Senior Python Developer, Streamlit Expert, UI/UX Designer, Computer Vision Integration Engineer, and Systems Integration Engineer.

Your task is to build a complete, production-quality Final Year Project called:

SMART TRAFFIC MANAGEMENT SYSTEM

AI-Powered Intelligent Traffic Monitoring and Emergency Vehicle Priority System

The application must be modular, scalable, maintainable, and follow software engineering best practices.

=========================

GENERAL REQUIREMENTS

=========================

Language:

Python 3.12+

Framework:

Streamlit

Architecture:

Modular

Coding Style:

PEP8

Documentation:

Every function must include docstrings and type hints.

The UI must have a modern dark theme with professional colors, cards, icons, charts, and responsive layouts.

Do not create one large file. Organize the project into logical modules.

=========================

PROJECT STRUCTURE

=========================

Create folders such as:

[app.py](http://app.py)

pages/

components/

services/

communication/

database/

models/

assets/

utils/

config/

logs/

tests/

=========================

PAGES

=========================

Create:

1. Dashboard

2. Live Camera

3. Traffic Analytics

4. Emergency Control

5. Hardware Monitor

6. System Logs

7. Settings

8. About

=========================

DASHBOARD

=========================

Display:

Current Time

Current Date

System Status

Camera Status

YOLO Status

ESP32 Status

Serial Status

Database Status

Traffic Density

Vehicle Count

Emergency Status

Current Lane

Average Speed

FPS

CPU Usage

RAM Usage

Use beautiful KPI cards.

=========================

LIVE CAMERA

=========================

Use a placeholder video.

Later this will connect to OpenCV.

Overlay fake detections.

Draw bounding boxes.

Draw vehicle IDs.

Draw lane markings.

Draw emergency vehicle in red.

=========================

TRAFFIC ANALYTICS

=========================

Generate realistic dummy data.

Show:

Vehicle Count Timeline

Traffic Density

Pie Chart of Vehicle Types

Congestion Trend

Average Waiting Time

Heatmap placeholder

Export CSV button

=========================

EMERGENCY CONTROL

=========================

Show

Emergency Vehicle Detected

Confidence

Detected Lane

Estimated Arrival

Current Signal

Current Speed Bump

Lane Divider Status

LED Status

LCD Status

Buttons

Activate Emergency

Deactivate

Open Lane

Close Lane

Raise Speed Bump

Lower Speed Bump

=========================

HARDWARE PAGE

=========================

Display

ESP32 Connected

Arduino Connected

Servo

LED Strip

LCD

WiFi

Voltage

Current

Heartbeat

Firmware Version

=========================

SYSTEM LOGS

=========================

Generate fake logs.

Allow filtering.

Allow searching.

Allow exporting.

=========================

SETTINGS

=========================

Camera

Serial Port

COM Port

Baud Rate

Detection Threshold

Theme

Save Settings

=========================

DATABASE

=========================

SQLite

Store

Detection History

Commands

System Logs

Traffic Statistics

Emergency Events

=========================

COMMUNICATION

=========================

Create interfaces for:

Serial

UART

MQTT

Initially use simulated responses.

=========================

MEMBER INTEGRATION

=========================

Member 1

Provide interfaces to receive:

Vehicle Count

Vehicle Classes

Tracking IDs

Traffic Density

Emergency Detection

YOLO Confidence

Member 3

Provide interfaces to send:

AMBULANCE_LANE_A

OPEN_GATE

CLOSE_GATE

RAISE_SPEED_BUMP

LOWER_SPEED_BUMP

LED_RED

LED_GREEN

LCD_MESSAGE

Simulate acknowledgements.

=========================

SIMULATION MODE

=========================

Since hardware is unavailable, implement a simulation mode with realistic random values and timers. Make it easy to switch to real hardware later.

=========================

UI

=========================

Use Streamlit's multipage navigation.

Use Plotly.

Use icons.

Use metric cards.

Use expanders.

Use tabs.

Use progress bars.

Use toast notifications.

Use session state.

Make it look like software used by a real traffic control center.

=========================

GOAL

=========================

The project should be demonstration-ready even without the AI model or hardware. Every external dependency must have a simulated implementation that can later be replaced with the real integrations without changing the UI.