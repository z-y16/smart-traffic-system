#!/usr/bin/env python3
"""Build the printable four-way intersection mat.

Trim size 50.0 x 76.0 cm, laid out on an A1 portrait sheet so it can be
printed at 100% and cut down.  Every dimension below is in centimetres:

                        50.0 cm across
        |<--12.5-->|6.25|6.25|6.25|6.25|<--12.5-->|
        +----------+----+----+----+----+----------+  ---
        |          |                   |          |
        |  green   |    north road     |  green   |   25.5
        |          |                   |          |
        +----------+----+----+----+----+----------+  ---   6.25
        |                                         |       6.25
        |  west road    junction    east road     |   25  6.25   76.0
        |                25 x 25                  |       6.25    down
        +----------+----+----+----+----+----------+  ---
        |          |                   |          |
        |  green   |    south road     |  green   |   25.5
        |          |                   |          |
        +----------+----+----+----+----+----------+  ---

The sheet is fixed at 50 x 76, so road width and approach length trade off
directly against each other:

    east/west approach = (50 - north-south road width) / 2
    north/south approach = (76 - east-west road width) / 2

Both roads carry four 6.25 cm lanes -- two in each direction -- so each of the
four approaches is a two-lane road 25 cm wide.  That keeps the junction down to
25 x 25 and leaves 12.5 cm of approach east and west, 25.5 cm north and south.
Nothing else is drawn: road surface, lane markings, green corners.

Colours, marking proportions and the kerb radius are taken from the original
circular version so the two look like the same drawing.

    python make_intersection_pdf.py
"""

import zlib

# --- units -----------------------------------------------------------------
CM = 72.0 / 2.54                     # PDF points per centimetre
MM = CM / 10.0

# --- sheet -----------------------------------------------------------------
SHEET_W = 50.0                       # trim width
SHEET_H = 76.0                       # trim height

A1_W = 59.4                          # A1 portrait, for the printable sheet
A1_H = 84.1

# --- road layout -----------------------------------------------------------
# Widen a lane and the junction grows while the approach roads shrink; narrow
# it and the opposite happens.  Everything below follows from these two.
NS_LANE = 6.25                       # north-south lane width   4 x 6.25 = 25
EW_LANE = 6.25                       # east-west lane width     4 x 6.25 = 25

NS_W = 4 * NS_LANE                   # 25.0  width of the vertical road
EW_W = 4 * EW_LANE                   # 25.0  width of the horizontal road

GREEN_X = (SHEET_W - NS_W) / 2       # 12.5  green margin = east/west approach
GREEN_Y = (SHEET_H - EW_W) / 2       # 25.5  green margin = north/south approach

VX0, VX1 = GREEN_X, GREEN_X + NS_W   # 12.5 .. 37.5   vertical road edges
HY0, HY1 = GREEN_Y, GREEN_Y + EW_W   # 25.5 .. 50.5   horizontal road edges

VXC = (VX0 + VX1) / 2                # 25.0  centreline of the vertical road
HYC = (HY0 + HY1) / 2                # 38.0  centreline of the horizontal road

# --- drawing detail --------------------------------------------------------
# Kept in proportion to the lane width, so the drawing looks the same whatever
# the lanes are set to.
CORNER_R = 0.48 * NS_LANE            # 3.00  kerb radius at the inner corners
MARK_W = 0.056 * NS_LANE             # 0.35  painted lane marking
EDGE_W = 0.064 * NS_LANE             # 0.40  road edge line
CENTRE_OFF = MARK_W                  # 0.35  half-gap of the double centre line
DASH_ON = 0.24 * NS_LANE             # 1.50  dashed lane divider
DASH_OFF = 0.192 * NS_LANE           # 1.20
BORDER_W = 0.2                       # outline / cut line

ROAD = (0.270588, 0.278431, 0.294118)
GREEN = (0.470588, 0.678431, 0.270588)
WHITE = (1.0, 1.0, 1.0)
INK = (0.0666667, 0.0666667, 0.0666667)

KAPPA = 0.5522847498307936           # circle -> cubic bezier constant


def n(v):
    """Shortest exact-enough PDF number."""
    s = "%.4f" % v
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def rgb(c, stroke=False):
    op = "RG" if stroke else "rg"
    return "%s %s %s %s" % (n(c[0]), n(c[1]), n(c[2]), op)


# --- the four kerb fillets --------------------------------------------------
# Each arc turns the road edge out into a green corner.  Given as
# (start, control1, control2, end) so they can be reused by both the road fill
# and the white edge line.
def _arc(sx, sy, c1x, c1y, c2x, c2y, ex, ey):
    return ((sx, sy), (c1x, c1y), (c2x, c2y), (ex, ey))


K = CORNER_R * KAPPA
R = CORNER_R

ARC_BL = _arc(VX0, HY0 - R, VX0, HY0 - R + K, VX0 - R + K, HY0, VX0 - R, HY0)
ARC_TL = _arc(VX0 - R, HY1, VX0 - R + K, HY1, VX0, HY1 + R - K, VX0, HY1 + R)
ARC_TR = _arc(VX1, HY1 + R, VX1, HY1 + R - K, VX1 + R - K, HY1, VX1 + R, HY1)
ARC_BR = _arc(VX1 + R, HY0, VX1 + R - K, HY0, VX1, HY0 - R + K, VX1, HY0 - R)


def curve(a):
    _, c1, c2, e = a
    return "%s %s %s %s %s %s c" % (n(c1[0]), n(c1[1]), n(c2[0]), n(c2[1]),
                                    n(e[0]), n(e[1]))


def road_outline():
    """The filleted cross, traced as one closed subpath."""
    p = ["%s 0 m" % n(VX0)]
    p.append("%s %s l" % (n(VX0), n(HY0 - R)))          # up the south-west edge
    p.append(curve(ARC_BL))
    p.append("0 %s l" % n(HY0))                          # out along the west arm
    p.append("0 %s l" % n(HY1))
    p.append("%s %s l" % (n(VX0 - R), n(HY1)))
    p.append(curve(ARC_TL))
    p.append("%s %s l" % (n(VX0), n(SHEET_H)))           # up the north arm
    p.append("%s %s l" % (n(VX1), n(SHEET_H)))
    p.append("%s %s l" % (n(VX1), n(HY1 + R)))
    p.append(curve(ARC_TR))
    p.append("%s %s l" % (n(SHEET_W), n(HY1)))           # out along the east arm
    p.append("%s %s l" % (n(SHEET_W), n(HY0)))
    p.append("%s %s l" % (n(VX1 + R), n(HY0)))
    p.append(curve(ARC_BR))
    p.append("%s 0 l" % n(VX1))                          # down the south arm
    p.append("h")
    return "\n".join(p)


def edge_lines():
    """White road edge line, only where the road meets green."""
    out = []
    for arc, before, after in (
        (ARC_BL, (VX0, 0.0), (0.0, HY0)),
        (ARC_TL, (0.0, HY1), (VX0, SHEET_H)),
        (ARC_TR, (VX1, SHEET_H), (SHEET_W, HY1)),
        (ARC_BR, (SHEET_W, HY0), (VX1, 0.0)),
    ):
        s = arc[0]
        out.append("%s %s m" % (n(before[0]), n(before[1])))
        out.append("%s %s l" % (n(s[0]), n(s[1])))
        out.append(curve(arc))
        out.append("%s %s l" % (n(after[0]), n(after[1])))
        out.append("S")
    return "\n".join(out)


def _seg(x1, y1, x2, y2):
    return "%s %s m %s %s l S" % (n(x1), n(y1), n(x2), n(y2))


def lane_lines(dashed):
    """Lane markings on the four approaches.  Nothing is drawn inside the box.

    Each line is emitted starting at the intersection and running outwards, so
    every approach opens with a full dash.
    """
    out = []
    if dashed:
        xs = (VX0 + NS_LANE, VX1 - NS_LANE)          # 18.75, 31.25
        ys = (HY0 + EW_LANE, HY1 - EW_LANE)          # 31.75, 44.25
    else:
        xs = (VXC - CENTRE_OFF, VXC + CENTRE_OFF)    # 24.65, 25.35
        ys = (HYC - CENTRE_OFF, HYC + CENTRE_OFF)    # 37.65, 38.35

    for x in xs:
        out.append(_seg(x, HY0, x, 0.0))             # south approach
        out.append(_seg(x, HY1, x, SHEET_H))         # north approach
    for y in ys:
        out.append(_seg(VX0, y, 0.0, y))             # west approach
        out.append(_seg(VX1, y, SHEET_W, y))         # east approach
    return "\n".join(out)


def artwork(ox, oy):
    """The whole drawing, placed with its bottom-left corner at (ox, oy) cm."""
    b = BORDER_W / 2
    return "\n".join([
        "q",
        "%s 0 0 %s %s %s cm" % (n(CM), n(CM), n(ox * CM), n(oy * CM)),
        "0 0 %s %s re W n" % (n(SHEET_W), n(SHEET_H)),

        rgb(GREEN),
        "0 0 %s %s re f" % (n(SHEET_W), n(SHEET_H)),

        rgb(ROAD),
        road_outline(),
        "f",

        rgb(WHITE, stroke=True),
        "0 J 1 j",
        "%s w" % n(EDGE_W),
        edge_lines(),

        "%s w [] 0 d" % n(MARK_W),
        lane_lines(dashed=False),

        "[%s %s] 0 d" % (n(DASH_ON), n(DASH_OFF)),
        lane_lines(dashed=True),

        "[] 0 d",
        rgb(INK, stroke=True),
        "%s w 0 j" % n(BORDER_W),
        "%s %s %s %s re S" % (n(b), n(b), n(SHEET_W - BORDER_W),
                              n(SHEET_H - BORDER_W)),
        "Q",
    ])


def crop_marks(ox, oy):
    """Hairline trim marks in the A1 margin, clear of the artwork."""
    gap, ln = 0.3 * CM, 1.0 * CM
    x0, y0 = ox * CM, oy * CM
    x1, y1 = x0 + SHEET_W * CM, y0 + SHEET_H * CM
    out = ["0 G", "0.25 w", "[] 0 d", "0 J"]
    for x in (x0, x1):
        for y in (y0, y1):
            sx = -1 if x == x0 else 1
            sy = -1 if y == y0 else 1
            out.append("%s %s m %s %s l S" % (n(x + sx * gap), n(y),
                                              n(x + sx * (gap + ln)), n(y)))
            out.append("%s %s m %s %s l S" % (n(x), n(y + sy * gap),
                                              n(x), n(y + sy * (gap + ln))))
    label = ("trim 50.0 x 76.0 cm  -  print at 100% (actual size), "
             "do not scale to fit")
    out.append("BT /F1 9 Tf 0.35 g")
    out.append("%s %s Td (%s) Tj" % (n(x0), n(y0 - 2.0 * CM), label))
    out.append("ET")
    return "\n".join(out)


def build_pdf(path, page_w, page_h, ox, oy, marks):
    """Write a one-page PDF.  Sizes in cm, artwork placed at (ox, oy)."""
    content = artwork(ox, oy)
    if marks:
        content += "\n" + crop_marks(ox, oy)
    stream = zlib.compress(content.encode("latin-1"), 9)

    trim = "[%s %s %s %s]" % (n(ox * CM), n(oy * CM),
                             n((ox + SHEET_W) * CM), n((oy + SHEET_H) * CM))
    res = "<< /Font << /F1 5 0 R >> >>" if marks else "<< >>"

    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %s %s] /TrimBox %s "
         "/ArtBox %s /Resources %s /Contents 4 0 R >>"
         % (n(page_w * CM), n(page_h * CM), trim, trim, res)),
        None,                                            # the content stream
    ]
    if marks:
        objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        if body is None:
            out += ("%d 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n"
                    % (i, len(stream))).encode("latin-1")
            out += stream
            out += b"\nendstream\nendobj\n"
        else:
            out += ("%d 0 obj\n%s\nendobj\n" % (i, body)).encode("latin-1")

    start = len(out)
    out += ("xref\n0 %d\n" % (len(objs) + 1)).encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += ("%010d 00000 n \n" % off).encode("latin-1")
    out += ("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, start)).encode("latin-1")

    with open(path, "wb") as fh:
        fh.write(bytes(out))
    return len(out)


if __name__ == "__main__":
    a1 = build_pdf("intersection_50x76_A1_print_ready.pdf", A1_W, A1_H,
                   (A1_W - SHEET_W) / 2, (A1_H - SHEET_H) / 2, marks=True)
    exact = build_pdf("intersection_50x76_exact.pdf", SHEET_W, SHEET_H,
                      0.0, 0.0, marks=False)
    print("intersection_50x76_A1_print_ready.pdf  %6d bytes  (A1 sheet, "
          "artwork centred, trim marks)" % a1)
    print("intersection_50x76_exact.pdf           %6d bytes  (page is exactly "
          "50 x 76 cm)" % exact)
