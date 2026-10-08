"""Render the README's explanatory workflow assets with Pillow.

Run from any directory: python assets/presentation/render_workflow.py
The animation explains data flow; it is not synchronized to recorded footage.
"""

from dataclasses import dataclass
from html import escape
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 720, 480
SCALE = 3
FPS = 10
SECONDS = 12
BACKGROUND = "#101417"
TEXT = "#F3F6F7"
MUTED = "#DFE6E9"
GREEN = "#68D5AF"
AMBER = "#F3BE65"
TEAL = "#7FD4DF"
VIOLET = "#BBABFA"

FILLS = {
    "input": "#203740",
    "tracking": "#342C4D",
    "traffic": "#203D32",
    "emergency": "#42351F",
    "signal": "#1A3028",
    "priority": "#362B1A",
    "api": "#203A43",
    "dashboard": "#30283F",
    "analytics": "#30283F",
}


@dataclass(frozen=True)
class Node:
    key: str
    box: tuple[int, int, int, int]
    title: str
    detail: str
    color: str
    group: int
    title_size: int = 22
    detail_size: int = 20


NODES = [
    Node("input", (196, 57, 524, 105), "Camera / Video / Stream", "", TEAL, 0),
    Node("tracking", (142, 127, 578, 183), "YOLO11m + BoT-SORT", "Vehicle detection + stable tracking", VIOLET, 1, 24, 20),
    Node("traffic", (28, 220, 338, 286), "Traffic measurements", "Speed / counts / congestion", GREEN, 2),
    Node("emergency", (382, 220, 692, 286), "Emergency recognition", "Livery + flashing-beacon check", AMBER, 2, 21, 20),
    Node("signal", (50, 315, 316, 363), "Signal state", "Congestion-driven phases", GREEN, 3, 21, 20),
    Node("priority", (404, 315, 670, 363), "Priority state", "Beacon required by default", AMBER, 3, 21, 20),
    Node("api", (238, 402, 482, 460), "CV / API node", "Video + telemetry", TEAL, 4, 22, 20),
    Node("dashboard", (28, 410, 214, 458), "Dashboard", "Streamlit", VIOLET, 5, 21, 20),
    Node("analytics", (506, 410, 692, 458), "CSV / Excel", "Session analytics", VIOLET, 5, 21, 20),
]

# Direct telemetry routes sit outside the two decision branches.
EDGES = [
    ([(360, 105), (360, 127)], VIOLET, 1),
    ([(360, 183), (360, 202), (183, 202), (183, 220)], GREEN, 2),
    ([(360, 183), (360, 202), (537, 202), (537, 220)], AMBER, 2),
    ([(183, 286), (183, 315)], GREEN, 3),
    ([(537, 286), (537, 315)], AMBER, 3),
    ([(28, 253), (16, 253), (16, 384), (238, 384), (238, 415)], GREEN, 4),
    ([(692, 253), (704, 253), (704, 384), (482, 384), (482, 415)], AMBER, 4),
    ([(183, 363), (183, 377), (300, 377), (300, 402)], GREEN, 4),
    ([(537, 363), (537, 377), (420, 377), (420, 402)], AMBER, 4),
    ([(238, 438), (214, 438)], VIOLET, 5),
    ([(482, 438), (506, 438)], VIOLET, 5),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size * SCALE)
    raise RuntimeError("Install Segoe UI or DejaVu Sans to render this asset.")


FONTS = {(size, bold): font(size, bold) for size in (20, 21, 22, 24, 28) for bold in (False, True)}


def points(values):
    return [(x * SCALE, y * SCALE) for x, y in values]


def mix(first, second, amount):
    a = tuple(int(first[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(second[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(a, b))


def arrow(draw, path, color, width=2):
    draw.line(points(path), fill=color, width=width * SCALE, joint="curve")
    (x0, y0), (x1, y1) = path[-2:]
    angle = math.atan2(y1 - y0, x1 - x0)
    wings = [(x1, y1)] + [
        (x1 - 6 * math.cos(angle + side * 0.55), y1 - 6 * math.sin(angle + side * 0.55))
        for side in (-1, 1)
    ]
    draw.polygon(points(wings), fill=color)


def path_position(path, fraction):
    lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
    distance = fraction * sum(lengths)
    for (a, b), length in zip(zip(path, path[1:]), lengths):
        if distance <= length:
            ratio = distance / length
            return a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio
        distance -= length
    return path[-1]


def draw_node(draw, node, active, progress=None):
    x0, y0, x1, y1 = node.box
    outline = node.color if active else mix(FILLS[node.key], node.color, 0.42)
    fill = mix(FILLS[node.key], node.color, 0.12) if active else FILLS[node.key]
    draw.rounded_rectangle((x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE),
                           radius=7 * SCALE, fill=fill, outline=outline,
                           width=(3 if active else 1) * SCALE)
    draw.line(((x0 + 12) * SCALE, (y0 + 1) * SCALE,
               (x1 - 12) * SCALE, (y0 + 1) * SCALE), fill=node.color, width=2 * SCALE)
    if active and progress is not None:
        head = x0 + 12 + (x1 - x0 - 24) * progress
        draw.line(((x0 + 12) * SCALE, (y0 + 1) * SCALE,
                   head * SCALE, (y0 + 1) * SCALE), fill=TEXT, width=3 * SCALE)
    middle = (x0 + x1) / 2
    center = (y0 + y1) / 2
    title_y = center - 11 if node.detail else center
    draw.text((middle * SCALE, title_y * SCALE), node.title,
              font=FONTS[node.title_size, True], anchor="mm", fill=TEXT)
    if node.detail:
        draw.text((middle * SCALE, (center + 13) * SCALE), node.detail,
                  font=FONTS[node.detail_size, False], anchor="mm", fill=MUTED)


def render(frame_number=None):
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.line((18 * SCALE, 17 * SCALE, 18 * SCALE, 40 * SCALE), fill=TEAL, width=4 * SCALE)
    draw.text((30 * SCALE, 27 * SCALE), "Smart Traffic", anchor="lm",
              fill=TEXT, font=FONTS[28, True])
    draw.text((692 * SCALE, 28 * SCALE), "Software Workflow", anchor="rm",
              fill=MUTED, font=FONTS[20, False])
    group = None if frame_number is None else min(5, frame_number // 20)
    for path, color, edge_group in EDGES:
        arrow(draw, path, color if group == edge_group else mix(BACKGROUND, color, 0.54),
              3 if group == edge_group else 2)
        if group == edge_group:
            x, y = path_position(path, (frame_number % 20) / 19)
            draw.ellipse(((x - 3.5) * SCALE, (y - 3.5) * SCALE,
                          (x + 3.5) * SCALE, (y + 3.5) * SCALE), fill=TEXT)
    for node in NODES:
        draw_node(draw, node, node.group == group,
                  None if frame_number is None else (frame_number % 20) / 19)
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def svg():
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480" viewBox="0 0 720 480" role="img" aria-labelledby="title desc">',
        '<title id="title">Smart Traffic software workflow</title>',
        '<desc id="desc">Video enters YOLO11m detection and BoT-SORT tracking. Traffic measurements feed congestion-driven signal states. Emergency recognition feeds a separate priority state, which requires beacon confirmation by default and supports manual override. Both measurements and emergency telemetry reach the CV API node, which exposes a Streamlit dashboard and CSV or Excel session analytics.</desc>',
        f'<rect width="720" height="480" fill="{BACKGROUND}"/>',
        '<g font-family="Segoe UI, DejaVu Sans, Arial, sans-serif" text-anchor="middle">',
        f'<path d="M18,17 V40" stroke="{TEAL}" stroke-width="4"/>',
        f'<text x="30" y="37" text-anchor="start" font-size="28" font-weight="700" fill="{TEXT}">Smart Traffic</text>',
        f'<text x="692" y="35" text-anchor="end" font-size="20" fill="{MUTED}">Software Workflow</text>',
    ]
    for path, color, _ in EDGES:
        edge_color = mix(BACKGROUND, color, 0.54)
        coords = " ".join(f"{x},{y}" for x, y in path)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{edge_color}" stroke-width="2" stroke-linejoin="round"/>')
        (x0, y0), (x1, y1) = path[-2:]
        angle = math.atan2(y1 - y0, x1 - x0)
        tip = [(x1, y1)] + [(x1 - 6 * math.cos(angle + side * 0.55), y1 - 6 * math.sin(angle + side * 0.55)) for side in (-1, 1)]
        parts.append(f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in tip)}" fill="{edge_color}"/>')
    for node in NODES:
        x0, y0, x1, y1 = node.box
        center = (y0 + y1) / 2
        middle = (x0 + x1) / 2
        parts.extend([
            f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="7" fill="{FILLS[node.key]}" stroke="{mix(FILLS[node.key], node.color, 0.42)}"/>',
            f'<path d="M{x0+12},{y0+1} H{x1-12}" stroke="{node.color}" stroke-width="2"/>',
            f'<text x="{middle}" y="{center-3 if node.detail else center+8}" font-size="{node.title_size}" font-weight="700" fill="{TEXT}">{escape(node.title)}</text>',
        ])
        if node.detail:
            parts.append(f'<text x="{middle}" y="{center+19}" font-size="{node.detail_size}" fill="{MUTED}">{escape(node.detail)}</text>')
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    render().save(OUT / "workflow.png", optimize=True)
    (OUT / "workflow.svg").write_text(svg(), encoding="utf-8")
    samples = [render(10 + group * 20) for group in range(6)]
    palette_sheet = Image.new("RGB", (WIDTH * 6, HEIGHT))
    for group, sample in enumerate(samples):
        palette_sheet.paste(sample, (WIDTH * group, 0))
    palette = palette_sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [render(frame).quantize(palette=palette, dither=Image.Dither.NONE)
              for frame in range(FPS * SECONDS)]
    frames[0].save(OUT / "workflow.gif", save_all=True, append_images=frames[1:],
                   duration=1000 // FPS, loop=0, disposal=1, optimize=False)
    print(f"Rendered workflow.gif: {WIDTH}x{HEIGHT}, {SECONDS}s, {FPS}fps")


if __name__ == "__main__":
    main()
