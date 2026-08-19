"""Checks for the congestion-driven traffic light state machine.

Everything here runs against a synthetic clock rather than ``time.time()``,
so the 5-second yellow interval is asserted exactly instead of being slept
through. The whole file finishes in milliseconds.

    python test_traffic_light.py
"""

from __future__ import annotations

import sys

from traffic_light import (
    DEFAULT_GREEN_LEVELS,
    LightState,
    Phase,
    TrafficLightConfig,
    TrafficLightController,
    lcd_lines,
)

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    """Print a section banner."""
    print(f"\n[{title}]")


def drive(controller: TrafficLightController, level: str,
          start: float, seconds: float, step: float = 0.1) -> LightState:
    """Run the controller over a span of simulated time, 10 Hz by default."""
    state = controller.update(level, start)
    elapsed = 0.0
    while elapsed < seconds:
        elapsed += step
        state = controller.update(level, start + elapsed)
    return state


# ══════════════════════════════════════════════════════════════════════════
section("A] Start-up and steady states")

light = TrafficLightController()
state = light.update("FREE", 0.0)
check("starts on RED", state.phase is Phase.RED, state.phase.value)
check("start-up is settled, not transitioning", not state.transitioning)

state = drive(light, "FREE", 0.0, 60.0)
check("an empty road stays RED indefinitely", state.phase is Phase.RED,
      f"after 60s -> {state.phase.value}")

check("HEAVY maps to GREEN", light.target_for("HEAVY") is Phase.GREEN)
check("MODERATE maps to GREEN", light.target_for("MODERATE") is Phase.GREEN)
check("FREE maps to RED", light.target_for("FREE") is Phase.RED)
check("an unknown level falls back to RED", light.target_for("WHATEVER") is Phase.RED)


# ══════════════════════════════════════════════════════════════════════════
section("B] RED -> GREEN passes through a 5s YELLOW")

light = TrafficLightController()
light.update("FREE", 0.0)

state = light.update("HEAVY", 10.0)
check("heavy traffic does not jump straight to GREEN", state.phase is not Phase.GREEN,
      state.phase.value)
check("it enters YELLOW first", state.phase is Phase.YELLOW)
check("YELLOW announces GREEN as its target", state.target is Phase.GREEN)
check("countdown starts at the full interval", abs(state.remaining - 5.0) < 1e-6,
      f"{state.remaining}s")

state = light.update("HEAVY", 12.5)
check("still YELLOW half way through", state.phase is Phase.YELLOW)
check("countdown has halved", abs(state.remaining - 2.5) < 1e-6, f"{state.remaining}s")

state = light.update("HEAVY", 14.9)
check("still YELLOW at 4.9s", state.phase is Phase.YELLOW, f"{state.remaining:.1f}s left")

state = light.update("HEAVY", 15.0)
check("turns GREEN at exactly 5.0s", state.phase is Phase.GREEN)
check("no countdown once settled", state.remaining == 0.0)
check("target equals phase once settled", state.target is Phase.GREEN)

state = drive(light, "HEAVY", 15.0, 30.0)
check("GREEN holds while traffic stays heavy", state.phase is Phase.GREEN)


# ══════════════════════════════════════════════════════════════════════════
section("C] GREEN -> RED passes through a 5s YELLOW")

state = light.update("FREE", 100.0)
check("emptying road does not snap to RED", state.phase is Phase.YELLOW, state.phase.value)
check("YELLOW announces RED as its target", state.target is Phase.RED)

state = light.update("FREE", 104.9)
check("still YELLOW at 4.9s", state.phase is Phase.YELLOW)

state = light.update("FREE", 105.0)
check("turns RED at exactly 5.0s", state.phase is Phase.RED)

state = drive(light, "MODERATE", 105.0, 20.0)
check("MODERATE traffic brings GREEN back", state.phase is Phase.GREEN,
      "FREE -> MODERATE -> GREEN")


# ══════════════════════════════════════════════════════════════════════════
section("D] A transition, once started, is always completed")

light = TrafficLightController()
light.update("FREE", 0.0)
light.update("HEAVY", 10.0)                      # yellow -> green begins

state = light.update("FREE", 12.0)               # traffic vanishes mid-yellow
check("mid-yellow level change does not abort the transition",
      state.phase is Phase.YELLOW and state.target is Phase.GREEN,
      f"{state.phase.value}->{state.target.value}")

state = light.update("FREE", 15.0)
check("it still lands on the committed phase", state.phase is Phase.GREEN,
      "reached GREEN even though traffic had cleared")

# min_phase_seconds now holds GREEN briefly before the next transition starts.
state = light.update("FREE", 15.5)
check("minimum green time blocks an instant re-transition",
      state.phase is Phase.GREEN, f"{state.phase.value} at +0.5s")

state = light.update("FREE", 18.0)
check("after the minimum dwell the next transition starts",
      state.phase is Phase.YELLOW and state.target is Phase.RED,
      f"{state.phase.value}->{state.target.value}")


# ══════════════════════════════════════════════════════════════════════════
section("E] No single-frame phases (the reason min_phase_seconds exists)")

light = TrafficLightController()
now = 0.0
light.update("FREE", now)
phases: list[tuple[float, Phase]] = []
# Alternate the level every 0.5s for two minutes - the worst case for chatter.
for step in range(240):
    now = step * 0.5
    level = "HEAVY" if (step // 1) % 2 == 0 else "FREE"
    phases.append((now, light.update(level, now).phase))

runs: list[tuple[Phase, float]] = []
for stamp, phase in phases:
    if runs and runs[-1][0] is phase:
        continue
    runs.append((phase, stamp))
durations = [runs[i + 1][1] - runs[i][1] for i in range(len(runs) - 1)]
shortest = min(durations) if durations else 0.0
check("even with the level flipping every 0.5s, no phase is momentary",
      shortest >= 3.0, f"shortest phase held {shortest:.1f}s over {len(runs)} changes")
yellow_runs = [durations[i] for i, (phase, _) in enumerate(runs[:-1]) if phase is Phase.YELLOW]
check("every yellow lasted the full 5s",
      all(abs(d - 5.0) < 1e-6 for d in yellow_runs),
      f"{len(yellow_runs)} yellows, all {set(round(d, 1) for d in yellow_runs)}s")
check("green and red never appear back to back without a yellow",
      all(not (runs[i][0] is not Phase.YELLOW and runs[i + 1][0] is not Phase.YELLOW)
          for i in range(len(runs) - 1)),
      "R->Y->G->Y->R ordering held")


# ══════════════════════════════════════════════════════════════════════════
section("F] Configuration")

fast = TrafficLightController(TrafficLightConfig(yellow_seconds=2.0, min_phase_seconds=0.0))
fast.update("FREE", 0.0)
fast.update("HEAVY", 1.0)
check("a shorter yellow is honoured", fast.update("HEAVY", 3.0).phase is Phase.GREEN,
      "2s yellow completed at 2s")

strict = TrafficLightController(TrafficLightConfig(green_levels=("HEAVY",)))
check("green_levels can exclude MODERATE", strict.target_for("MODERATE") is Phase.RED)
check("HEAVY still goes green under the strict mapping",
      strict.target_for("HEAVY") is Phase.GREEN)

check("default mapping serves MODERATE and HEAVY",
      DEFAULT_GREEN_LEVELS == ("MODERATE", "HEAVY"), str(DEFAULT_GREEN_LEVELS))
check("config round-trips to plain JSON values",
      TrafficLightConfig().to_dict()["yellow_seconds"] == 5.0)
check("default yellow is the requested 5 seconds",
      TrafficLightConfig().yellow_seconds == 5.0)

green_start = TrafficLightController(TrafficLightConfig(initial_phase=Phase.GREEN))
check("initial phase is configurable",
      green_start.update("MODERATE", 0.0).phase is Phase.GREEN)


# ══════════════════════════════════════════════════════════════════════════
section("G] Manual override")

light = TrafficLightController()
light.update("FREE", 0.0)
light.request_phase(Phase.GREEN)
state = light.update("FREE", 5.0)
check("a forced GREEN still goes through YELLOW", state.phase is Phase.YELLOW,
      "an override cannot skip the amber")
state = light.update("FREE", 10.0)
check("the forced phase is reached", state.phase is Phase.GREEN)
state = drive(light, "FREE", 10.0, 30.0)
check("the override holds against the congestion level", state.phase is Phase.GREEN,
      "road is empty but GREEN is forced")
check("the override is reported", light.forced is Phase.GREEN)

light.request_phase(None)
check("releasing the override clears it", light.forced is None)
state = drive(light, "FREE", 40.0, 20.0)
check("automatic control resumes after release", state.phase is Phase.RED)


# ══════════════════════════════════════════════════════════════════════════
section("H] LCD rendering")

light = TrafficLightController()
light.update("FREE", 0.0)
settled = light.update("FREE", 4.0)
line1, line2 = lcd_lines(settled, "FREE", 0.12)
check("settled line 1 names the phase", "RED" in line1, repr(line1))
check("line 1 is exactly 16 columns", len(line1) == 16, str(len(line1)))
check("line 2 is exactly 16 columns", len(line2) == 16, str(len(line2)))
check("line 2 carries level and congestion index",
      "FREE" in line2 and "0.12" in line2, repr(line2))

light.update("HEAVY", 10.0)
mid = light.update("HEAVY", 12.0)
line1, line2 = lcd_lines(mid, "HEAVY", 0.81)
check("yellow line 1 shows the countdown and destination",
      "YELLOW" in line1 and "GREEN" in line1 and "3s" in line1, repr(line1))
check("yellow line 1 still fits 16 columns", len(line1) == 16, str(len(line1)))

line1, line2 = lcd_lines(mid, "HEAVY", 0.81, emergency_text="!! AMBULANCE !!")
check("an emergency takes over line 2", "AMBULANCE" in line2, repr(line2))
check("the emergency banner still fits", len(line2) == 16, str(len(line2)))
check("the signal line is unaffected by the emergency", "YELLOW" in line1, repr(line1))

long_text = lcd_lines(settled, "FREE", 0.0, emergency_text="X" * 40)[1]
check("over-long text is truncated, never wrapped", len(long_text) == 16, str(len(long_text)))


# ══════════════════════════════════════════════════════════════════════════
section("I] Telemetry shape")

payload = mid.to_dict()
check("state serialises to JSON-friendly values",
      all(isinstance(v, (str, int, float, bool)) for v in payload.values()),
      str(payload))
check("phase serialises as a plain string", payload["phase"] == "YELLOW", payload["phase"])
check("Phase compares equal to its string form", Phase.GREEN == "GREEN")


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f" RESULT: {passed} passed, {failed} failed")
print("=" * 70)
sys.exit(1 if failed else 0)
