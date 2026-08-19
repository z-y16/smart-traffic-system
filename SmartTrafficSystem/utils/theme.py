"""UI theme and styling utilities."""

import streamlit as st


def apply_dark_theme() -> None:
    """Inject custom CSS for a professional traffic control center theme."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }

            .dashboard-header {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 16px;
                padding: 1.5rem 2rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
            }

            .dashboard-header h1 {
                color: #f8fafc;
                font-size: 1.75rem;
                font-weight: 700;
                margin: 0 0 0.25rem 0;
            }

            .dashboard-header p {
                color: #cbd5e1;
                margin: 0;
                font-size: 0.95rem;
            }

            /* ── Front page ─────────────────────────────────────────────
               The title page is the only place with no live figures on it,
               so it is allowed more room than a dashboard header: it is read
               once, from further away, by people deciding what they are
               looking at. */

            .title-hero {
                background:
                    radial-gradient(120% 140% at 85% 0%,
                                    rgba(45, 212, 191, 0.22) 0%,
                                    rgba(15, 23, 42, 0) 55%),
                    linear-gradient(135deg, #0b1220 0%, #111c2e 45%, #0f2e2b 100%);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 20px;
                padding: 2.5rem 2.5rem 2.25rem 2.5rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 18px 48px rgba(2, 6, 23, 0.55);
            }

            .hero-eyebrow {
                color: #2dd4bf;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                margin-bottom: 0.9rem;
            }

            .hero-title {
                color: #f8fafc;
                font-size: 2.35rem;
                font-weight: 700;
                line-height: 1.18;
                letter-spacing: -0.02em;
                margin: 0 0 0.85rem 0;
                max-width: 26ch;
            }

            .hero-tagline {
                color: #cbd5e1;
                font-size: 1.02rem;
                line-height: 1.6;
                margin: 0 0 1.4rem 0;
                max-width: 62ch;
            }

            .hero-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }

            .hero-chip {
                display: inline-block;
                background: rgba(148, 163, 184, 0.12);
                border: 1px solid rgba(148, 163, 184, 0.28);
                border-radius: 999px;
                color: #e2e8f0;
                font-size: 0.78rem;
                font-weight: 500;
                padding: 0.3rem 0.85rem;
            }

            .hero-chip.is-live {
                background: rgba(52, 211, 153, 0.14);
                border-color: rgba(52, 211, 153, 0.4);
                color: #34d399;
            }

            .hero-chip.is-offline {
                background: rgba(148, 163, 184, 0.1);
                border-color: rgba(148, 163, 184, 0.3);
                color: #94a3b8;
            }

            /* Five members do not fit five-across on a laptop, so the row
               wraps itself rather than being pinned to a column count that
               only looks right on one screen. */
            .team-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
                gap: 0.9rem;
            }

            .team-card {
                background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 14px;
                padding: 1.15rem 1.2rem;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }

            .team-card:hover {
                transform: translateY(-2px);
                border-color: rgba(45, 212, 191, 0.4);
            }

            .team-card.is-unfilled {
                border-style: dashed;
                border-color: rgba(251, 191, 36, 0.45);
            }

            .team-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, #0f766e 0%, #2dd4bf 100%);
                color: #04211f;
                font-size: 0.9rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 0.75rem;
            }

            .team-card.is-unfilled .team-avatar {
                background: rgba(251, 191, 36, 0.18);
                color: #fbbf24;
            }

            .team-name {
                color: #f1f5f9;
                font-size: 1.02rem;
                font-weight: 600;
                line-height: 1.3;
                margin-bottom: 0.2rem;
            }

            .team-card.is-unfilled .team-name {
                color: #fbbf24;
            }

            .team-role {
                color: #2dd4bf;
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.55rem;
            }

            .team-contribution {
                color: #94a3b8;
                font-size: 0.83rem;
                line-height: 1.5;
            }

            .result-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 0.9rem;
            }

            .result-card {
                background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 14px;
                padding: 1.15rem 1.2rem;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
            }

            .result-figure {
                color: #2dd4bf;
                font-size: 1.9rem;
                font-weight: 700;
                line-height: 1.15;
                margin-bottom: 0.3rem;
            }

            .result-label {
                color: #e2e8f0;
                font-size: 0.86rem;
                font-weight: 600;
                margin-bottom: 0.25rem;
            }

            /* The qualifier is the point of the card, not decoration: "0 false
               alarms" means nothing without "on night footage". */
            .result-note {
                color: #94a3b8;
                font-size: 0.76rem;
                line-height: 1.45;
            }

            .aim-card {
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-left: 3px solid #2dd4bf;
                border-radius: 12px;
                padding: 1.1rem 1.35rem;
                height: 100%;
            }

            .aim-label {
                color: #2dd4bf;
                font-size: 0.72rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 0.5rem;
            }

            .aim-body {
                color: #cbd5e1;
                font-size: 0.9rem;
                line-height: 1.6;
            }

            .kpi-card {
                background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 14px;
                padding: 1.1rem 1.25rem;
                min-height: 120px;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }

            .kpi-card:hover {
                transform: translateY(-2px);
                border-color: rgba(45, 212, 191, 0.4);
            }

            .kpi-icon {
                font-size: 1.4rem;
                margin-bottom: 0.35rem;
            }

            .kpi-label {
                color: #94a3b8;
                font-size: 0.78rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.35rem;
            }

            .kpi-value {
                color: #f1f5f9;
                font-size: 1.65rem;
                font-weight: 700;
                line-height: 1.2;
                margin-bottom: 0.25rem;
            }

            .kpi-delta {
                font-size: 0.8rem;
                font-weight: 500;
            }

            .status-online { color: #34d399; }
            .status-simulated { color: #38bdf8; }
            .status-degraded { color: #fbbf24; }
            .status-offline { color: #f87171; }
            .status-neutral { color: #94a3b8; }

            .density-low { color: #34d399; }
            .density-moderate { color: #38bdf8; }
            .density-high { color: #fbbf24; }
            .density-critical { color: #f87171; }

            .section-title {
                color: #e2e8f0;
                font-size: 1.05rem;
                font-weight: 600;
                margin: 1.5rem 0 0.75rem 0;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(148, 163, 184, 0.15);
            }

            .sim-badge {
                display: inline-block;
                background: rgba(56, 189, 248, 0.15);
                color: #38bdf8;
                border: 1px solid rgba(56, 189, 248, 0.35);
                border-radius: 999px;
                padding: 0.25rem 0.75rem;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 14px;
                padding: 1rem;
            }

            div[data-testid="stMetric"] label {
                color: #94a3b8 !important;
            }

            /* Streamlit sizes metric values for short numbers and ellipsises
               anything longer, which clipped the date and timestamp cards. */
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
                color: #f1f5f9 !important;
                font-size: 1.4rem;
                line-height: 1.3;
                white-space: normal;
                overflow-wrap: anywhere;
            }

            div[data-testid="stMetric"] div[data-testid="stMetricValue"] > div {
                overflow: visible;
                text-overflow: clip;
                white-space: normal;
            }

            /* The built-in pages/ nav labels the entry point "app" and cannot
               show icons — components.navigation renders the list instead. */
            div[data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_class(status: str) -> str:
    """Return CSS class name for a connection or system status label."""
    mapping = {
        "Online": "status-online",
        "Simulated": "status-simulated",
        "Degraded": "status-degraded",
        "Offline": "status-offline",
    }
    return mapping.get(status, "status-neutral")


def density_class(density: str) -> str:
    """Return CSS class name for traffic density level."""
    mapping = {
        "Low": "density-low",
        "Moderate": "density-moderate",
        "High": "density-high",
        "Critical": "density-critical",
    }
    return mapping.get(density, "status-neutral")
