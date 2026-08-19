"""Shared sidebar navigation.

Streamlit's built-in ``pages/`` navigation labels the entry point after its
filename ("app") and cannot carry icons, so it is hidden in ``utils.theme``
and replaced by this explicit list, rendered identically on every page.
"""

import streamlit as st

PAGES: tuple[tuple[str, str, str], ...] = (
    ("app.py", "Home", "🏠"),
    ("pages/1_Operations_Dashboard.py", "Operations Dashboard", "📊"),
    ("pages/2_Live_Camera.py", "Live Camera", "📹"),
    ("pages/3_Traffic_Analytics.py", "Traffic Analytics", "📈"),
    ("pages/4_Emergency_Control.py", "Emergency Control", "🚨"),
    ("pages/5_Hardware_Monitor.py", "Hardware Monitor", "🔧"),
    ("pages/6_System_Logs.py", "System Logs", "📋"),
    ("pages/7_Settings.py", "Settings", "⚙️"),
    ("pages/8_About.py", "About", "ℹ️"),
)


def render_sidebar_nav() -> None:
    """Render the full page list inside the sidebar."""
    st.markdown("**Navigation**")
    for path, label, icon in PAGES:
        st.page_link(path, label=label, icon=icon)
