"""Who made this and what it is called.

Everything the front page says about the *project* rather than about the
traffic lives here: the title, the aim, the group, the measured results. It is
separated from :mod:`config.settings` because none of it is a setting — nobody
changes it between runs, and it must not end up in ``settings.json`` where a
saved preference could quietly overwrite a member's name.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Identity ────────────────────────────────────────────────────────────────

PROGRAMME = "Group Design Project"
STAGE = "Final Submission"

PROJECT_TITLE = (
    "Computer Vision-Based Smart Traffic Congestion Analytics "
    "with Adaptive Signal Optimization"
)

PROJECT_TAGLINE = (
    "A camera, a GPU and embedded controllers that monitor traffic, measure "
    "vehicle movement, and adapt the prototype traffic-control system."
)

PROJECT_AIM = (
    "To develop a Computer Vision-Based Smart Traffic Congestion Analytics "
    "System with Adaptive Signal Optimization that monitors real-time traffic "
    "conditions and improves urban traffic flow efficiency through intelligent "
    "signal control."
)

GROUP_OBJECTIVE = (
    "To develop a fully functional smart traffic congestion analytics "
    "prototype capable of reducing simulated traffic waiting time by at least "
    "20% before final project evaluation."
)

DURATION = "14 weeks"
THEME = "Smart City & AI Traffic Optimization"


# ── The group ───────────────────────────────────────────────────────────────

PLACEHOLDER = "Add name"


@dataclass(frozen=True)
class Member:
    """One member of the group, as shown on the front page."""

    name: str
    role: str
    contribution: str

    @property
    def filled(self) -> bool:
        """Whether a real name has been put in this slot yet."""
        return bool(self.name.strip()) and self.name.strip() != PLACEHOLDER

    @property
    def initials(self) -> str:
        """Up to two initials for the avatar, or a question mark if unfilled."""
        if not self.filled:
            return "?"
        parts = [word for word in self.name.split() if word]
        return "".join(word[0].upper() for word in parts[:2]) or "?"


TEAM: tuple[Member, ...] = (
    Member(
        name="Zeyad Khairy",
        role="AI & Computer Vision Lead",
        contribution="Vehicle detection and tracking, traffic analytics, speed "
                     "measurement, emergency-vehicle vision and CSV output.",
    ),
    Member(
        name="Abdulrahman Osama",
        role="Dashboard & System Integration",
        contribution="Dashboard integration, live system status, hardware links "
                     "and presentation of traffic analytics.",
    ),
    Member(
        name="Naif Mohammed",
        role="Embedded Firmware",
        contribution="Microcontroller firmware, traffic-light control logic and "
                     "servo/lane-control integration.",
    ),
    Member(
        name="Anfaz Mohammed",
        role="Power & Electronics",
        contribution="LED driving, prototype wiring, power distribution and "
                     "electrical integration of the road model.",
    ),
    Member(
        name="Adam Becheikh",
        role="Mechanical Prototype",
        contribution="Road-model construction, lane layout and mechanical "
                     "integration of the physical prototype.",
    ),
)


def roster_is_complete() -> bool:
    """Whether every slot in :data:`TEAM` has a real name in it."""
    return all(member.filled for member in TEAM)


def missing_names() -> int:
    """How many roster slots are still placeholders."""
    return sum(1 for member in TEAM if not member.filled)


# ── Measured results ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Result:
    """One headline figure, with the qualifier that keeps it honest."""

    figure: str
    label: str
    note: str


HEADLINE_RESULTS: tuple[Result, ...] = (
    Result("36.6 fps", "Detector throughput", "YOLO11m @ 1280 px, FP16, RTX 4050"),
    Result("±5%", "Speed accuracy", "calibrated — exact against ground truth"),
    Result("0", "False alarms, night footage", "down from 29% of all traffic"),
    Result("833", "Automated checks", "across 7 suites"),
)

MEASURED_ON = "9 August 2026 · RTX 4050 laptop GPU, FP16"
