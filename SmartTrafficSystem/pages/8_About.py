"""About page for the Smart Traffic Management System."""

import streamlit as st

from components.navigation import render_sidebar_nav
from components.page_header import render_page_header
from config import project
from config.settings import get_settings
from utils.session import init_session_state
from utils.theme import apply_dark_theme

settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_title} | About",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_dark_theme()
init_session_state()

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.divider()
    render_sidebar_nav()

render_page_header(
    "ℹ️ About",
    "AI-Powered Intelligent Traffic Monitoring and Emergency Vehicle Priority System",
)

st.markdown(
    f"""
    ### Project Overview

    **{project.PROJECT_TITLE}** is a {project.PROGRAMME} that combines computer
    vision, IoT hardware, and a real-time control dashboard to monitor traffic
    density and prioritize emergency vehicles.

    > **Aim.** {project.PROJECT_AIM}
    >
    > **Group objective.** {project.GROUP_OBJECTIVE}

    ### Key Features

    - Real-time traffic monitoring with YOLO11-m object detection and BoT-SORT tracking
    - Vehicle speed measured in **km/h**, not pixels — via road-plane calibration
    - **Priority lane control triggered by a working siren light** — a beacon
      that genuinely switches on and off, with its colour naming the vehicle
      (blue → police, red → ambulance or fire engine)
    - CLIP zero-shot livery recognition, used to name the vehicle type rather
      than to trigger — a marked vehicle with its lights off gets no priority
    - ESP32 hub driving two Arduino Unos over its own UARTs — one USB cable
      from the PC reaches the signal heads, the lane changer and the LED strip
    - Crash-safe CSV + styled Excel telemetry logging
    - Simulation mode for demonstration without hardware

    ### Technology Stack

    | Component | Technology |
    |-----------|------------|
    | Frontend | Streamlit |
    | Backend | Python 3.12+ |
    | Database | SQLite |
    | Charts | Plotly |
    | AI Detection | YOLO11-m (Ultralytics) @ 1280 + BoT-SORT, FP16 on CUDA |
    | Emergency Trigger | Temporal siren-beacon detection (colour → vehicle type) |
    | Emergency Type Naming | CLIP ViT-B/16 zero-shot livery recognition |
    | Speed Measurement | Homography road-plane calibration → km/h |
    | Hardware | ESP32, Arduino, Servo, LED, LCD |

    ### The Group

    """
    # One roster, read from config/project.py, so a name corrected for the
    # front page is corrected here at the same time rather than drifting.
    + "\n".join(
        f"    - **{member.name}** — {member.role}: {member.contribution}"
        for member in project.TEAM
    )
    + """

    ---
    *Built with modular architecture for easy replacement of simulated components.*
    """
)

if not project.roster_is_complete():
    st.info(
        f"{project.missing_names()} name(s) still to fill in — edit `TEAM` in "
        "`SmartTrafficSystem/config/project.py`."
    )
