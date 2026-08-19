"""Shared session state initialization helpers."""

import streamlit as st

from backend.services.analytics_service import AnalyticsService
from backend.services.camera_service import CameraService
from backend.services.recording_service import RecordingService
from backend.services.system_status_service import SystemStatusService
from backend.services.video_source_service import VideoSourceService
from config.settings import get_settings


def init_session_state() -> None:
    """Initialize shared session state keys used across pages."""
    if "settings" not in st.session_state:
        st.session_state.settings = get_settings()
    if "status_service" not in st.session_state:
        st.session_state.status_service = SystemStatusService(st.session_state.settings)
    if "camera_service" not in st.session_state:
        st.session_state.camera_service = CameraService(st.session_state.settings)
    if "analytics_service" not in st.session_state:
        st.session_state.analytics_service = AnalyticsService(st.session_state.settings)
    if "recording_service" not in st.session_state:
        st.session_state.recording_service = RecordingService()
    if "video_source_service" not in st.session_state:
        st.session_state.video_source_service = VideoSourceService()
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
