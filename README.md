# Smart Traffic Management System — CV Node

AI-powered traffic monitoring with **YOLO11-m**, real **km/h** speed
measurement, emergency-vehicle priority, Arduino hardware control and a
Streamlit control-centre dashboard.

It watches whatever you point it at: the camera on the desk, a video you
upload from the dashboard, or a roadside camera's stream URL — the same
pipeline, live, on all three.

```
GDP/
├── START HERE.bat            ← double-click this: runs everything
├── start.py                  ← the same thing from a terminal: python start.py
├── broadcast_server.py       ← the CV node: source → detection → hardware → logs → HTTP
├── traffic_vision.py         ← the engine: detector, km/h speeds, congestion, logging
├── video_source.py           ← where frames come from: cameras, videos, stream URLs
├── traffic_light.py          ← the signal state machine (green/yellow/red timing)
├── siren_vision.py           ← the trigger: is a beacon actually flashing, and what colour
├── emergency_vision.py       ← ambulance / police / fire-truck livery recognition
├── make_siren_test_video.py  ← paint a beacon onto real footage, to test detection
├── check_siren_detection.py  ← run the detector over a clip and explain every verdict
├── demo_emergency_clip.py    ← emergency recognition alone on any clip, no ROI, no speed
├── benchmark.py              ← measure fps, per-stage cost and vehicles found
├── train_emergency_classifier.py ← rebuild emergency_cls.pt from new footage
├── emergency_cls.pt          ← the trained livery model (ambulance/police/fire/ordinary)
├── calibrate_speed.py        ← run once per camera position for accurate km/h
├── traffic_controller_arduino/  ← the Arduino sketch (flash this)
├── HOW_THE_WHOLE_SYSTEM_WORKS.txt      ← plain-English tour of the whole pipeline
├── HOW_EMERGENCY_DETECTION_WORKS.txt   ← plain-English tour of CLIP + the beacon detector
├── METRICS.md                ← every measured number, and how it was measured
├── test_traffic_system.py    ← 217 checks on the vision pipeline
├── test_server.py            ← 165 checks on hardware mapping, the ESP32 relay, HTTP
├── test_video_source.py      ← 113 checks on uploads, stream URLs and playback
├── test_dashboard_interactive.py ← 109 checks that the dashboard widgets work
├── test_analytics.py         ← 90 checks on the dashboard's analytics
├── test_dashboard.py         ← 83 checks that all 9 dashboard pages render
├── test_traffic_light.py     ← 56 checks on the signal timing
├── trackers/traffic_botsort.yaml
├── yolo11m.pt                ← the model now in use
├── uploads/                  ← videos uploaded from the dashboard
├── vedios/                   ← footage copied on by hand; searched the same way
├── traffic_history.{csv,xlsx}       ← the current session only
├── traffic_all_sessions.{csv,xlsx}  ← every session ever recorded
├── sessions/                 ← a standalone copy of each finished session
└── SmartTrafficSystem/       ← the Streamlit dashboard
```

---

## Quick start

**Double-click `START HERE.bat`**, or from a terminal in this folder:

```bash
python start.py
```

That is the whole thing. It starts the CV node, starts the dashboard, tells the
dashboard where the node is, raises the upload limit, and opens the browser.
**One Ctrl-C in that window stops everything cleanly** — the workbooks are
finished, the vehicles still in frame are logged and the serial port is
released, none of which happens if the window is simply closed.

First time only:

```bash
pip install -r requirements.txt
pip install -r SmartTrafficSystem/requirements.txt
```

Anything `start.py` does not recognise is passed straight to the CV node, so
every option below still works:

```bash
python start.py --source road.mp4        # start on a recorded video
python start.py --source rtsp://...      # start on a roadside camera
python start.py --no-arduino             # no hardware connected
python start.py --no-display             # no preview window on the CV node
```

You do not have to get the source right at start-up: it can be changed from the
dashboard at any time, and a missing camera no longer stops the system from
starting.

### Running the two parts on separate machines

The point of the split is that the CV node needs the camera and the GPU while
the dashboard only needs a browser. On the machine with the camera:

```bash
python start.py --no-dashboard
```

and on the other machine, with the node's IPv4 address from `ipconfig`:

```bash
python start.py --no-server --node 192.168.1.25
```

Both machines must be on the same network, and Windows Firewall must allow
Python on port 8502.

### Starting the parts by hand

Still supported, and what `start.py` does for you:

```bash
# 1. CV node (the PC with the camera)
python broadcast_server.py

# 2. Dashboard (same PC or another one on the WiFi)
cd SmartTrafficSystem
streamlit run app.py
```

Started this way the dashboard has to be told where the node is: set
`TRAFFIC_STREAM_HOST=127.0.0.1` before launching Streamlit, or fill it in on
the Settings page. Forgetting this is the single most common way to end up
looking at an "offline" dashboard next to a perfectly healthy node.

### Useful options

```bash
python broadcast_server.py --list-cameras        # name every camera, say which works, exit
python broadcast_server.py --source droidcam     # pick a camera by name (survives reindexing)
python broadcast_server.py --source 1            # or by index
python broadcast_server.py --source road.mp4     # a recorded video instead of the camera
python broadcast_server.py --source rtsp://...   # a roadside camera
python broadcast_server.py --imgsz 960           # ~28 fps instead of ~20, misses distant cars
python broadcast_server.py --model yolo12n.pt    # fall back to the old nano model
python broadcast_server.py --no-display          # headless
python broadcast_server.py --no-arduino          # skip the serial port
python broadcast_server.py --help                # everything else
```

The source is only a *starting* point — it can be changed from the dashboard
while the system is running, and does not have to work at start-up. If the
camera is unplugged the node comes up empty and waits, rather than refusing to
start.

### Keys while the video window is focused

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `q` | quit | `x` | write the workbooks now |
| `e` | toggle emergency mode | `c` | reload the calibration file |
| `g` `y` `r` | force green / yellow / red | `p` | save a snapshot PNG |
| `a` | back to automatic mode | | |

Video files only:

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `space` | pause / resume | `l` | toggle looping |
| `[` `]` | jump back / forward 5 s | `0` | restart from the beginning |

A forced phase is still reached **through the normal 5-second yellow** — an
override cannot make the signal jump straight from green to red.

---

## The traffic signal

The light is **demand-responsive**, and what it means depends on where it
stands. `--signal-mode` picks which road it governs.

**`merge` (default)** — the layout in `MERGE_4_*.pdf`: two main carriageways
running opposite ways, each fed by a side road that merges into it. The signals
are on the **side roads**, so they hold merging traffic back exactly when the
carriageway is busiest and let it in once there is room. The main roads run
free and are never signalled.

| Congestion on the main road | Side-road signal |
|------------|--------|
| `HEAVY` | **RED** — close the merge, do not add to the jam |
| `MODERATE` | **GREEN** — there is room, let them in |
| `FREE` (empty road) | **GREEN** |
| any change between them | **YELLOW** for 5 seconds |

**`junction`** — the older behaviour, for a signal on the measured approach
itself: green while there is traffic to serve, red once the road is empty, so
an empty approach never holds a green nobody is using.

| Congestion | Signal |
|------------|--------|
| `HEAVY` | **GREEN** — let the queue discharge |
| `MODERATE` | **GREEN** |
| `FREE` (empty road) | **RED** — nothing to serve |
| any change between them | **YELLOW** for 5 seconds |

Only *which levels call for green* differs between the two — the state machine,
the committed yellow and the minimum dwell below are identical.

Two rules keep it stable:

* **Transitions are committed.** Once yellow starts it always runs the full five
  seconds and always lands on the phase it was aiming at, exactly like a real
  signal, which never aborts halfway through an amber.
* **Minimum dwell** (`--min-phase-seconds`, default 3 s). Green and red are held
  briefly before another change may begin. Without it, traffic that flips during
  the yellow interval could produce a green lasting a single frame.

`test_traffic_light.py` drives the machine against a synthetic clock and proves
the 5-second interval exactly, including the worst case: with the congestion
level flipping every 0.5 s for two minutes, no phase is ever momentary and the
`R→Y→G→Y→R` ordering never breaks.

```bash
python broadcast_server.py --yellow-seconds 3     # shorter amber
python broadcast_server.py --min-phase-seconds 0  # disable the minimum dwell
```

To make **moderate** traffic wait at red instead, set `green_levels=("HEAVY",)`
in `TrafficLightConfig` (`traffic_light.py`) — it is a one-line change.

### Emergency vehicles

Emergency mode is raised by a **working siren light** — a beacon that actually
switches on and off — and not by a vehicle merely looking like an ambulance.
An ambulance parked outside a hospital is still an ambulance and does not need
the junction held for it; see
[Emergency vehicle detection](#emergency-vehicle-detection--ambulance-police-fire-truck)
for how that is measured and why it replaced livery recognition as the trigger.

An emergency vehicle **does not seize the signal**. Emergency traffic proceeds
through any indication, so taking the light over would only disrupt everyone
else. Instead a **dedicated LED strip output (Arduino pin 6)** lights to tell
drivers to leave space, while the signal carries on serving normal traffic. The
LCD names the vehicle type read from the beacon colour.

### Extending to a full intersection

One `TrafficLightController` drives one approach. A four-way junction is four
controllers plus a policy object deciding which approach may hold green; the
per-approach timing rules do not change. `request_phase()` exists for exactly
that — a coordinator can drive each approach directly instead of deriving the
phase from congestion. **Not built yet**, by request.

---

## Where the frames come from

The detector, the speed estimator, the congestion index, the emergency
recogniser, the signal, the Arduino and both workbooks are identical whatever
is being watched. So the source is a **choice made while running**, not
something fixed when the process started.

| Kind | What to give it | Example |
|------|-----------------|---------|
| **camera** | a name or an index | `droidcam`, `2` |
| **file** | a path, or the bare name of a video in `uploads/` or `vedios/` | `road.mp4`, `nightroad.mp4` |
| **stream** | an RTSP / MJPEG / HLS URL, or a page yt-dlp can resolve | `rtsp://user:pass@192.168.1.40:554/stream1` |

A bare filename is looked for in `uploads/` (what the dashboard sent) and then
`vedios/` (footage copied onto the machine by hand), so `nightroad.mp4` names
the same file from the CLI, the URL tab and the library list without anyone
having to type the folder. Both directories are listed in the picker; only
`uploads/` entries can be deleted from the dashboard, since the rest is source
material it did not put there.

All three are assumed to be **fixed cameras watching a road** — mounted on a
gantry, a pole or a bridge — which is what makes the measurements mean
anything: the ground plane stays put, so a calibration stays valid and a
vehicle's motion across the frame is the vehicle moving rather than the camera.
Footage from a moving vehicle would break that assumption, and every speed
with it.

### From the dashboard

Live Camera → **Change source**, which offers four tabs: upload a video, paste
a camera or stream URL, pick a connected camera, or re-use something uploaded
earlier. A video gets transport controls under the picture — pause, ±5 s,
restart, loop — and a scrub bar and speed control in the panel.

Two details worth knowing:

* **The live refresh stops while the panel is open.** A page that reruns twice
  a second cannot be typed into. The video feed keeps streaming regardless,
  because it is an MJPEG `<img>` and not part of the rerun.
* **Streamlit accepts 200 MB uploads by default.** Raise it with
  `streamlit run app.py --server.maxUploadSize 2000`, or for a very large file
  copy it next to `broadcast_server.py` and type its name on the URL tab —
  that path never crosses the network at all.

### Why a video needs its own clock

Speed is distance over **time**, and for a recording the only honest clock is
the video's own. Measuring against the wall clock makes the answer a property
of the computer rather than of the traffic: a PC that manages 20 fps on a
30 fps file watches every vehicle take half again as long to cross the frame
and reports speeds a third too low, and one that races through the file
unthrottled reports them absurdly high — or, as it happens here, fails to
measure them at all, because the whole video passes inside the 0.8-second
window a speed reading needs.

So `video_source.py` publishes a **media clock** that advances one frame
interval per frame consumed. `test_video_source.py` drives a vehicle scripted
to travel at exactly 16.2 km/h through a real MP4 at four playback rates —
real time, 2×, half speed and unthrottled — and requires all four answers to
land on 16.2 km/h. They agree to 0.00 km/h.

The clock is expressed as a real timestamp, so the log is stamped sensibly, and
it is monotonic across loops and seeks so no duration ever comes out negative.

### Real-time playback

A file plays at its native frame rate and **drops frames to stay on schedule**
when detection cannot keep up — which is exactly what a live camera does, where
a frame not read in time is simply gone. Skipped frames still advance the media
clock, so dropping them costs nothing but overlay smoothness.

```bash
python broadcast_server.py --source road.mp4 --playback-rate 0   # analyse it as fast as possible
python broadcast_server.py --source road.mp4 --no-drop-frames    # decode every frame, slower than life
python broadcast_server.py --source road.mp4 --loop-video        # start again at the end
python broadcast_server.py --source road.mp4 --exit-on-end       # stop when it finishes
```

By default a finished video **holds on its last frame** and the node stays up,
because the point of all this is to load another one without restarting.

### What happens at the moment of a switch

The new source is opened **before** the old one is dropped, so a mistyped URL
reports what was wrong and leaves the running feed untouched. Once the switch
lands, everything belonging to the old scene is retired: tracks, speed history,
class votes, emergency evidence, the congestion dwell timer and the signal's
dwell clock. Vehicles still in frame are written to the per-vehicle log first,
so none are lost, and the tracker's ids restart — which is why the
*unique vehicles* count starts again too. Anything else would invent journeys
between two unrelated roads.

The same reset happens when a looping file wraps round and when someone scrubs
the timeline, for the same reason.

### Cameras on the road

Most public "live traffic camera" links are web pages rather than streams.
`pip install yt-dlp` on the CV node lets those be used directly; without it,
such a link is refused with a message saying so rather than an opaque decoder
error. Direct RTSP, MJPEG and HLS addresses need nothing extra.

RTSP is forced over TCP — over UDP it drops packets on any congested network
and produces the smeared, half-decoded frames that ruin detection. A stream
that dies is reconnected automatically (three attempts, backing off), and each
reconnection is treated as a discontinuity. An address that cannot be reached
is rejected in about three seconds, by a plain socket check, instead of the
thirty OpenCV would otherwise spend before saying the same thing.

---

## Speed in km/h

The old pipeline reported **pixels per second**, which is not a speed — a car
30 m away moves a tenth as many pixels as the same car up close at the same
km/h. Two calibration modes now convert pixels into real metres.

### Automatic (works immediately, ±25%)

No setup. Each vehicle's bounding-box **height** is compared with the known
typical height of its class (a car is ~1.50 m, a bus ~3.20 m), which gives a
metres-per-pixel scale **at that vehicle's exact position in the frame**.
Because every vehicle carries its own local scale, perspective is handled
implicitly.

This assumes the camera can see how *tall* a vehicle is, which a roadside or
gantry camera looking along the road can. A camera pointing **straight down**
cannot: what fills the box vertically is then the vehicle's four-metre length,
not its one-and-a-half-metre height, and every reading comes out roughly three
times too slow. For a near-vertical view, calibrate with
`python calibrate_speed.py --mode scale` — two clicks and one measurement —
rather than relying on the automatic mode.

### Calibrated (recommended, ±5%)

```bash
python calibrate_speed.py
```

Click 4 points on the road surface that form a rectangle in the real world,
then type its real width and length in metres. Press `v` to preview a 1-metre
grid — if the grid lines follow the road markings, the calibration is good.
Press `s` to save.

Things that are easy to measure: a standard lane is ~3.5 m wide, the gap
between two lamp posts, a stretch of kerb with a tape measure.

The result is written to `speed_calibration.json` and picked up on the next
start (or press `c` to reload live). Re-run it whenever the camera moves.

**A calibration belongs to one camera in one position**, so it does not follow
the source when you switch. Point the node at a second road camera and the
calibration on disk is still the first one's — geometrically meaningless for
the new view. Two ways round it:

* calibrate against the new view (`python calibrate_speed.py --source rtsp://...`,
  or `--source clip.mp4` for an uploaded video), then press `c` on the CV node
  to reload without restarting; or
* keep a file per camera and pass it with
  `python start.py --calibration junction_a.json`.

Comparing speeds *between* two cameras is only meaningful once both are
calibrated. Comparing counts and congestion between them needs no calibration
at all.

You can also draw a **region of interest** with `r` during calibration —
detections outside it are ignored, which removes a pavement or the opposite
carriageway from the counts.

### How a speed is actually computed

1. The **bottom-centre** of each box is used, since that is where the vehicle
   touches the road and therefore the only point whose motion is planar.
2. Displacement is measured over a **0.8 s window**, not between consecutive
   frames — box jitter would otherwise be reported as motion.
3. Readings above 200 km/h are discarded as tracker ID switches.
4. An EMA smooths the per-vehicle result.

Verified in `test_traffic_system.py`: a synthetic vehicle scripted to travel
at exactly 18 / 36 / 72 km/h is measured at 18.0 / 36.0 / 72.0.

---

## Detection accuracy upgrades

| # | Upgrade | Why it matters |
|---|---------|----------------|
| 1 | **yolo11m** instead of yolo12n | ~+12 mAP; distant vehicles the nano model missed are now found. yolo11m rather than yolo12m: measured below, YOLO12's attention blocks cost real time on an Ada laptop GPU and buy nothing at this resolution |
| 2 | **FP16 on CUDA** | ~2× throughput — this is what pays for the bigger model |
| 3 | **imgsz 1280** (was 640) | the biggest single lever on small/far vehicles, and the only one that helps at night: measured over 30 night frames, 960 finds 15.6 vehicles/frame and 1280 finds 19.6 |
| 4 | **Class-filtered inference** | people and traffic lights can never become "vehicles" |
| 5 | **Class-agnostic NMS** | kills the duplicate box where a van is both `car` and `truck` |
| 6 | **BoT-SORT + motion compensation** | far fewer ID switches than the old ByteTrack config; every ID switch corrupts a speed reading |
| 7 | **Track confirmation** (3 frames) | one-frame ghost detections never reach the counts or the servo |
| 8 | **Sticky class voting** | a vehicle stops flickering between `car` and `truck` in the log |
| 9 | **Congestion hysteresis** | the LEDs and servo stop chattering when the index sits on a threshold |
| 10 | **Siren-triggered emergency mode** | the priority lane opens for a *working beacon*, not for anything that resembles a police car: 88 false alarms over the night footage became 0 |
| 11 | **Presence-weighted congestion** | one slow driver on an empty road is no longer reported as a traffic jam |
| 12 | **Memory pruning** | track/colour history no longer grows unbounded over a long session |

Measured with `python benchmark.py --grid` on an RTX 4050 Laptop GPU, over 60
consecutive 1080p frames of the night footage in `vedios/`, FP16:

| model @ imgsz | ms/frame | FPS | vehicles/frame |
|---------------|----------|-----|----------------|
| yolo11s @ 640 | 14.2 | 70.6 | 18.4 |
| yolo11s @ 960 | 16.0 | 62.5 | 20.8 |
| yolo11s @ 1280 | 18.4 | 54.3 | 22.8 |
| yolo11m @ 640 | 17.2 | 58.0 | 17.0 |
| yolo11m @ 960 | 19.0 | 52.8 | 19.3 |
| **yolo11m @ 1280 (default)** | **27.3** | **36.6** | **23.0** |
| yolo11l @ 1280 | 32.8 | 30.5 | 22.5 |
| yolo12s @ 1280 | 24.0 | 41.7 | 23.6 |
| yolo12m @ 1280 | 38.7 | 25.8 | 22.1 |

Two things fall out of that table, and the second one is uncomfortable.

**Resolution is what buys detections; model size is not.** Every model gains
about five vehicles per frame going from 640 to 1280, and almost nothing going
from `s` to `l` at a fixed resolution — yolo11l at 1280 finds *fewer* than
yolo11m (22.5 against 23.0) while costing 20% more time. The limit is how many
pixels a distant car occupies, not how much capacity is spent on it. That is
also why yolo12m is the slowest row here and not the best: YOLO12's attention
blocks cost real time on this GPU and buy nothing at this resolution.

**yolo11s at 1280 finds 22.8 vehicles per frame at 54 fps, against yolo11m's
23.0 at 37 fps** — within 1% of the yield for 48% more throughput. That is a
real result and it is not yet acted on, because `vehicles/frame` is a *count*,
not a correctness score: with no labelled ground truth for this footage, a
model that draws more boxes cannot be distinguished from a model that draws
better ones. Anyone wanting to switch the default should label a few hundred
frames first. See `METRICS.md`.

CLAHE contrast enhancement was also measured and made things slightly *worse*
(18.7), because at night it amplifies sensor noise along with the cars.

### What each stage costs

`python benchmark.py --stages`, same GPU, yolo11m at 1280:

| stage | night (23 veh/frame) | day (25.5 veh/frame) |
|-------|---------------------|----------------------|
| detection + tracking (YOLO) | 29.6 ms (66%) | 28.8 ms (82%) |
| livery recognition (CLIP) | 5.4 ms (12%) | 1.1 ms (3%) |
| beacon detection (siren) | 9.8 ms (22%) | 5.2 ms (15%) |
| **end to end** | **44.8 ms — 22.3 fps** | **35.1 ms — 28.5 fps** |

Detection dominates, so it is the only stage worth optimising. Both recognisers
together cost less than a quarter of the frame, and they cost *more at night* —
CLIP because dark crops fail the readability gate and get retried rather than
backed off, the beacon detector because there are actually beacons to measure.

Tuning knobs live in `VisionConfig` at the top of `traffic_vision.py`, and the
common ones are exposed as command-line flags.

---

## Logging

Two records are kept side by side, because they answer different questions.

### The current session — `traffic_history.csv` / `.xlsx`

Only the run happening right now. This is what you look at during a demo, and
what the dashboard's **This Session** export hands you. It resets on every
start; the previous run is moved into `sessions/` first, so nothing is lost.

| Sheet | Contents |
|-------|----------|
| Live Status | the most recent sample |
| Traffic Log | every sample **in this session**, colour-coded by level, with autofilter |
| Vehicle Speeds | one row per vehicle that passed through: class, duration, distance, average and top speed |
| Session Summary | totals, means, peaks, time at each congestion level **and each signal phase** |

### Every session ever — `traffic_all_sessions.csv` / `.xlsx`

Appended to continuously and never reset, so no past run is ever lost. Each row
carries the session id that produced it.

| Sheet | Contents |
|-------|----------|
| All Sessions | one summary row per session: start, end, duration, peak vehicles, mean speed, mean/peak congestion, time at each level |
| All Records | every telemetry row from every session |
| Lifetime Summary | totals across the whole archive |

A standalone copy of each finished session is also kept in `sessions/`.

### How it stays safe

Every row is appended to **both** CSVs immediately and flushed — a plain append
that cannot corrupt earlier rows. The CSVs are the durable record: they are
always complete up to the last second, however the process ends.

The styled workbooks are derived from them and are built **on request** — when
one is downloaded, on **Save Session Now**, when a recording is stopped, and on
exit — then written to a temp file and atomically renamed, so killing the
process can never leave a half-written workbook. If a workbook is open in Excel
when a write is due, the write is skipped with a warning and retried — the CSVs
keep flowing regardless.

They used to be rebuilt on a background thread every 20 s instead. A rebuild is
a full regeneration rather than an append, and openpyxl is pure Python, so it
held the GIL against the detection loop sharing the process: measured on an
8,600-row archive, 6.3 s per master rebuild, which works out at roughly a
quarter of a core an hour into a run and approaches half after three. Building
on request costs the same work only when something actually wants a workbook.

Columns: session id and name, date, time, congestion index, level, **light
phase**, total vehicles, per-class counts, **average and max speed in km/h**,
moving/stopped split, weighted density (passenger-car equivalents), servo angle,
emergency flag and count, unique vehicles, FPS and the calibration mode.

Any file whose columns no longer match the schema is archived as
`*_legacy_<timestamp>.*` rather than being appended to with mismatched columns.

---

## HTTP endpoints

Served by the CV node on port 8502 (`--stream-port`):

| Endpoint | Purpose |
|----------|---------|
| `/video_feed` | annotated MJPEG stream |
| `/telemetry` | latest snapshot, including per-vehicle detections and the signal phase |
| `/history` | rolling ~1 hour of 1 Hz telemetry |
| `/export` · `/export.csv` | download **this session's** workbook / CSV |
| `/export/all` · `/export/all.csv` | download the **all-sessions** workbook / CSV |
| `/sessions` | one summary record per session ever logged |
| `/records` | telemetry rows across every session (`?session=ID`, `?limit=N`) |
| `/record/status` | is a named recording running, and how much has it captured |
| `/record/start` · `/record/stop` | begin / finish a named recording |
| `/record/save` | force both workbooks to be written now |
| `/light` | read the signal phase; `POST {"phase": "GREEN"}` to force one, `"AUTO"` to release |
| `/source` | what is being watched and where it has got to; `POST {"source": "..."}` to change it |
| `/sources` | the cameras this node can see, and the videos already uploaded |
| `/source/upload` | `POST` a video file (field `file`); it is checked, stored and played |
| `/source/upload/<name>` | `DELETE` an uploaded video |
| `/source/control` | `POST {"action": "pause"\|"resume"\|"toggle"\|"restart"\|"seek"\|"loop"\|"rate"}` |
| `/command` | `POST {"command": "...", "payload": {...}}` — manual hardware control, forwarded to the Arduino |
| `/health` | liveness, FPS, calibration mode, signal phase, recording state, source |

`/telemetry` and `/health` both carry the current source, and `/telemetry`
reads it live rather than from the last frame — a paused video produces no
frames, so a paused flag carried on a frame could never arrive.

### Hardware commands

The dashboard has no serial port of its own. This node owns the Arduino — on
Windows a second process cannot open the same COM port at all — so the manual
controls on the Emergency Control page are POSTed to `/command` and forwarded
down the same link. That is also what lets the dashboard run on a different
machine from the camera.

| Command | Payload | Sent to the board |
|---------|---------|-------------------|
| `MOVE_LANE_DIVIDER` | `{"direction": "LEFT"\|"RIGHT"}` | the neighbouring position's `LANE_*` word |
| `SET_LANE_ALLOCATION` | `{"allocation": "BALANCED"\|"FORWARD_4"\|"OPPOSITE_4"}` | `LANE_BALANCED` / `LANE_FORWARD_4` / `LANE_OPPOSITE_4` |
| `LED_MODE` | `{"mode": "POLICE"\|"AMBULANCE"\|"SPEED"\|"OFF"}` | `POLICE` / `AMBULANCE` / `SPEED` / `NORMAL` |
| `RAISE_SPEED_BUMP` · `LOWER_SPEED_BUMP` | — | `BUMP_UP` / `BUMP_DOWN` |
| `LCD_MESSAGE` | `{"message": "..."}` | `LCD:<message>` |
| `ACTIVATE_EMERGENCY` · `DEACTIVATE_EMERGENCY` | `{"lane": "..."}` | `AMBULANCE` / `NORMAL` |

These are bare words, not the seven-field telemetry packet, and
`traffic_controller_arduino.ino` drops any line without seven comma-separated
fields — so the signal controller ignores them and only the lane-changer/LED
board (`SmartTrafficSystem/Traffic_light_gdp/`) acts on
them. One port carries both.

The reply says whether the command actually reached the wire, so a missing
board is visible rather than silently successful:

```bash
curl -X POST localhost:8502/command -H "Content-Type: application/json" \
     -d '{"command": "SET_LANE_ALLOCATION", "payload": {"allocation": "FORWARD_4"}}'
# {"success": true, "sent": "LANE_FORWARD_4", "delivered": true,
#  "lane_allocation": "FORWARD_4_OPPOSITE_2", "arduino_connected": true, ...}
```

The Emergency Control page drives the divider with two arrows rather than three
position buttons, so `MOVE_LANE_DIVIDER` is what it actually sends: the node
holds the current position, steps one place along `OPPOSITE_4 → BALANCED →
FORWARD_4`, and forwards the resulting `SET_LANE_ALLOCATION` — the boards never
learn a second lane command. An arrow pressed at the outermost lane sends
nothing and comes back `"at_limit": true`, rather than replaying the servo's
move-and-blink routine to change nothing.

The commanded state is echoed back in `/telemetry`'s `hardware` block, so the
dashboard shows the position the barrier was actually moved to rather than one
it assumes. The divider is deliberately *not* derived from `servo_angle`: that
servo is the signal's phase gate and swings on every light change, whereas the
divider only moves when someone moves it.

```bash
curl -X POST localhost:8502/source -H "Content-Type: application/json" \
     -d '{"source": "rtsp://192.168.1.40:554/stream1"}'
curl -F file=@road.mp4 localhost:8502/source/upload
curl -X POST localhost:8502/source/control -H "Content-Type: application/json" \
     -d '{"action": "seek", "fraction": 0.5}'
```

---

## Dashboard

### Recording a session

The CV node **always** logs to the all-sessions archive, so nothing is ever
lost. Recording marks a *named slice* of it, which is what makes a particular
run findable afterwards — so when analysing an uploaded video, name the
recording after it. The log's columns are deliberately unchanged by the source
feature, so every past session stays readable alongside the new ones.

* **⏺️ Record Session** — sidebar of the main dashboard, or the Analytics page
  where you can also give it a name first.
* **⏹️ Stop Recording** — closes the run, writes its workbook, appends it to the
  archive and starts a fresh session so logging never pauses.
* **💾 Save Session Now** — forces both workbooks to be written immediately,
  without changing what is being recorded.

### Analytics

Two bodies of data, and the difference matters: **Live Session** charts come
from the rolling in-memory buffer (~1 hour), while **Archive Insights** read
every session ever recorded. The **Data scope** selector switches between this
session, all sessions, or one past session.

| Chart | The question it answers |
|-------|-------------------------|
| Demand by Hour | When is this road busiest? Which hours to avoid scheduling work |
| Speed vs Congestion | What does congestion actually cost in journey time? Where the road starts failing |
| Signal Behaviour | How is the light spending its time, and how congested was the road really? |
| Distribution | Is congestion a steady problem or occasional spikes? Two cases an average cannot tell apart |
| Session Comparison | Is it getting worse over time? |
| Session Shape | The narrative of one run: when traffic built, how long it stayed, whether speed fell |

Each chart carries a **How to read this / Why it is useful** note, and the
**What the data says** panel turns the archive into plain-language findings —
the busiest hour, whether congestion is persistent or intermittent, and how
much speed is lost under load.

---

## Emergency vehicle detection — ambulance, police, fire truck

COCO has no ambulance or police-car class: to YOLO an ambulance is a `truck`
and a police car is a `car`. Two cues are therefore combined — but not as
equals. **The siren light decides; the livery only names the type.**

### Why it works this way

An earlier version asked CLIP "does this look like a police car?" and let a
yes raise the priority lane on its own. Measured over the six night clips in
`vedios/`, that flagged **88 of 299 vehicles — 29% of all traffic** — as
emergency vehicles, almost all of them "police".

Two things were wrong with it, and both matter:

**A dark car at night matches the description.** "A police vehicle with sirens
and blue lights" is a sentence that fits any saloon with its headlamps on, seen
at 50 pixels across in the dark. Appearance is a *description*, and ordinary
traffic fits it.

**The arithmetic was against us.** The score summed the probability of *all*
emergency prompts. With 14 emergency prompts against 15 ordinary ones, a CLIP
that has learned nothing at all from a crop scores 14/29 = 0.48 — just over the
0.45 threshold, which had been calibrated on daylight photographs where CLIP
actually was informative. Every ambiguous vehicle came out "police" by
construction.

And underneath both: **livery is the wrong question.** An ambulance parked
outside a hospital is still an ambulance. What earns a cleared lane is a
vehicle *responding to a call*, and the visible evidence of that is its beacon
working.

### 1. The siren light — what the vehicle is *doing*. This is the trigger.

`siren_vision.py`. For every tracked vehicle, the roofline is measured each
frame for pure red and pure blue light, and the resulting signal must **switch**
rather than merely brighten:

| Test | Requirement | What it rejects |
|------|-------------|-----------------|
| Colour purity | `g ≤ 0.55·r` for red, `r ≤ 0.70·b` for blue | amber indicators and tow beacons, white headlights |
| Brightness | dominant channel ≥ 165 | bluish bodywork and reflections (measured at 130–155 on real cars) |
| Position | red sampled only in the roof band | tail lights, brake lights, US red indicators |
| Modulation depth | light nearly vanishes between flashes | steady tail lights, red paint |
| **Bimodality** | **≤15% of samples caught mid-transition** | **a car drifting through a pool of street light** |
| Repetition | ≥3 flashes at 1–6 Hz within 3.4 s | a driver tapping the brakes once or twice; a red bus drifting through street light at 0.9 Hz |
| Persistence | evidence sustained ≥1.2 s | a high-mounted centre brake light in traffic |

The bimodality test is the one that does the heavy lifting on real footage. A
lamp is a *switch* — fully on or fully off, with essentially no sample caught
in between. Bodywork passing under a street light is a *ramp*, and spends most
of its time part-lit. Measured over the night clips, the real beacon scored
0.00 on this and every false positive scored 0.18–0.43.

The beacon's **colour names the vehicle**, which is what makes this a
classifier and not just an alarm:

| Beacon | Vehicle | Note |
|--------|---------|------|
| blue | police | decisive — no civilian vehicle anywhere carries a flashing blue lamp |
| red | ambulance | fire engine if the vehicle is a truck or bus |
| alternating red + blue | police | configurable |

Conventions differ by country, so the mapping is settable:

```bash
python broadcast_server.py --siren-blue police --siren-red ambulance
```

### 2. The livery — what the vehicle *is*. This names, it does not trigger.

Every tracked vehicle is cropped and classified into ambulance / police / fire
truck / ordinary. Two recognisers are available behind one interface, and
`emergency_backend=auto` (the default) picks the trained one when
`emergency_cls.pt` is present and falls back to zero-shot CLIP when it is not —
see [Training a livery model](#training-a-livery-model-for-this-camera) for the
head-to-head.

CLIP's half needed three fixes before it was usable at all:

* the police prompts describe **paintwork and lettering**, not sirens and blue
  lights — flashing lights are now measured, so CLIP is no longer asked to
  guess at them;
* nine night-specific negatives were added ("a car at night with its headlights
  on", "the red tail lights of a car in the dark", …), because those images
  previously had nowhere to go but into an emergency class;
* the score is now a two-way posterior, `P(best emergency type) / (P(best
  emergency type) + P(ordinary))`, so an uninformative crop scores 0.25 rather
  than 0.48, and the larger ordinary prompt set acts as the prior it should
  always have been. Crops too dark or too flat to read are not shown to CLIP at
  all.

CLIP is used for two things: telling an **ambulance from a fire engine** when
both show a red beacon, and raising confidence when both cues agree.

A vehicle CLIP recognises whose beacon is *not* running is reported as
**suspected** — drawn in amber with a `?`, listed in the dashboard, written to
the log — and given **no priority**. That is a decision, not an oversight.

```bash
# Let livery trigger the lane on its own. Right for a camera watching a
# hospital or station approach; wrong for a public road.
python broadcast_server.py --livery-triggers-emergency
```

### Measured result

Same six night clips, same pipeline, before and after:

| | vehicles flagged | of tracks |
|---|---|---|
| Livery-triggered (before) | **88** | 299 — 29.4% |
| Siren-triggered (now) | **0** | 370 — 0.0% |

CLIP's own emergency score over the same footage fell from a mean of 0.38
(37% of vehicles over the old threshold) to a mean of 0.018, peaking at 0.16
against a 0.70 threshold.

Detection was verified in the other direction too, because a system that
flags nothing passes a false-positive test perfectly. `make_siren_test_video.py`
composites a physically plausible light bar — correct colour, flash rate, bloom
and halo — onto a real vehicle in real footage, so one vehicle in the clip is an
emergency vehicle and every other is genuine traffic:

```bash
python make_siren_test_video.py --source "vedios/Test1_dark _long_vedio.mp4" \
    --colour redblue --rate 2.0
python make_siren_test_video.py --source clip.mp4 --colour blue \
    --strength 0.45 --size 0.05     # a dim, distant beacon
```

| Planted beacon | Detected | Type | Confidence |
|---|---|---|---|
| red + blue, 2 Hz, close | 88 frames | police ✓ | 1.00 |
| blue, 1.5 Hz, dim and distant | 470 frames | police ✓ | 0.77 |

Getting to zero took four discriminators that footage alone would not have
suggested, each added after looking at what actually tripped: bodywork brightness
(dull lavender panels at B≈140 reading as blue), bimodality (cars under street
lights), the roofline band (a car's high-mounted centre brake light), and the
minimum flash rate (a red double-decker bus at 0.9 Hz).

Two more came from writing tests rather than from the videos. **Sunlight sweeping
across red bodywork** — through trees, past buildings — passed everything,
because the hard brightness cut-off was *converting* a smooth ramp into a
switched signal by clipping everything below it to zero, destroying the very
evidence the bimodality test needs. The floor now fades in over 45 levels. And a
**scarlet car in direct sun** is as saturated and as bright as a beacon at
midnight; only the temporal test separates them. Both are in the suite as
`[8e] Siren detection in daylight`, which matters because every measurement
above was taken at night.

### Cost and control

The siren detector needs no GPU, no weights and no download; it costs
**0.5–8 ms per frame** depending on how many vehicles are in shot, and runs on
machines where CLIP is switched off. CLIP classification is batched once per
frame, each vehicle re-checked every 5 frames, with vehicles that look clearly
ordinary backing off exponentially.

```bash
python broadcast_server.py --no-emergency-ai      # siren detection only, no CLIP
python broadcast_server.py --clip-model ViT-B/32  # faster, slightly less accurate
python broadcast_server.py --emergency-interval 8 # re-check livery less often
```

### One thing to watch: frame rate

A beacon flashes 1–4 times a second, so the pipeline must *see* each vehicle at
roughly 8 fps or faster for the flashing to survive sampling. Below that a
siren can pass undetected. Rather than report a clear road, the system says so:
the preview overlay, the telemetry (`siren_undersampled`) and the dashboard all
warn when sampling drops too low. If that appears, lower `--imgsz` to 960 or
switch to `yolo12n.pt`.

### Training a livery model for this camera

CLIP is the default because it needs no data and knows any country's vehicles.
What it does not know is *this* camera. `train_emergency_classifier.py` builds
a model that does:

```bash
python train_emergency_classifier.py          # fetch, mine, build, train
python broadcast_server.py --emergency-backend trained
```

Four stages, and the third is the one that matters:

**fetch** — pulls the vehicle classes of ImageNet-1k from Hugging Face:
ambulance, fire engine and police van as positives, and fourteen deliberately
confusable ordinary classes as negatives — white moving vans (an ambulance's
body), taxis (a roof sign), tow trucks (an amber beacon), school buses (a large
vehicle covered in markings). Public, ungated, no account, ~1 GB.

**mine** — runs this project's own detector over the clips in `vedios/` and
saves real vehicle crops as *ordinary*. These are the most valuable training
images available: they are exactly what the old system was getting wrong, and
no public dataset contains anything like them. Only clips known to contain no
emergency vehicle are mined (`EMERGENCY_FREE_CLIPS`) — mining a clip with a
real ambulance in it would teach the model the opposite of the truth.

**build** — **the stage that decides whether any of this is worth doing.**
ImageNet's ambulances are sharp daylight photographs taken from the pavement.
What the camera hands the classifier is a 60-pixel dark shape at 3 a.m.,
smeared by motion blur and chewed by H.264. Training on the first and deploying
on the second is precisely the mistake that made the old system call 29% of
night traffic "police". So every source photograph is baked into degraded
variants that imitate the camera — resolution thrown away, exposure pulled
down, motion blur, sodium or LED colour cast, sensor noise, JPEG artefacts —
applied in the order a real camera applies them.

Two details in that stage are easy to get wrong and both were:

* *Degraded variants are checked for readability* against the same thresholds
  the pipeline uses at run time, and dropped if they fail. A variant so dark
  the pipeline would refuse to classify it must not sit in the training set
  labelled "ambulance" — that is training the model to answer "ambulance" from
  noise, which is the exact failure being corrected.
* *The validation split is degraded too.* Validating on clean photographs
  reports an accuracy that has nothing to do with the camera.

**train** — fine-tunes `yolo11s-cls` and writes `emergency_cls.pt`.
The augmentation is deliberately restrained, because **for this task colour is
the feature**: a fire engine is red, an ambulance is white and yellow, a patrol
car is blue and white. Ultralytics defaults to `auto_augment="randaugment"`
with `hsv_s=0.7`, which recolours vehicles freely — inspecting the first run's
training batches showed a cyan police car and a green ambulance — and no amount
of training recovers a feature that has been augmented away. RandAugment is off,
hue is left almost alone, and the realistic variation comes from the build
stage instead.

#### Measured, head to head

Held-out crops neither model saw, degraded to match the camera:

| over the 0.70 threshold | CLIP | Trained |
|---|---|---|
| ambulance recognised | 53.4% | **71.1%** |
| police car recognised | 41.1% | **71.4%** |
| fire engine recognised | 39.7% | **70.3%** |
| false alarms on **this camera's** night crops | 0.0% | **0.0%** |
| false alarms on generic ImageNet vans/taxis | **0.3%** | 5.3% |
| cost per crop | 4.3 ms | **0.7 ms** |

The trained model recognises far more of what it should, seven times faster, and
neither produces a single false alarm on the footage this camera actually sees.
Its one weakness — generic vans and taxis — is also the cheapest mistake
available here, because livery alone cannot open the priority lane. It shows a
"?" and nothing else happens.

So `emergency_backend` defaults to **`auto`**: the trained model when
`emergency_cls.pt` is present, CLIP when it is not. A fresh clone works either
way.

| | CLIP | Trained |
|---|---|---|
| Setup | 335 MB download + the `clip` package | 11 MB file, ultralytics only |
| Knows | any country's vehicles | what it was shown |
| Knows *this* camera | no | yes — part of its training data came from it |

Neither changes **when** the priority lane opens: that is the siren detector's
decision in both cases.

#### Retraining after new footage

```bash
# add clips to vedios/, list the emergency-free ones in EMERGENCY_FREE_CLIPS
python train_emergency_classifier.py --stage mine --stage build --stage train
```

Daytime footage is worth adding: the mined crops so far are all night, so the
model has seen this camera only in the dark.

---

## Tests

```bash
python test_traffic_system.py          # 217 checks, no GPU needed (~5 s)
python test_traffic_system.py --yolo   # + real yolo11m and real CLIP on real photos
python test_server.py                  # 124 checks: hardware mapping, serial format, HTTP, launcher
python test_video_source.py            # 113 checks: uploads, stream URLs, playback, km/h
python test_dashboard_interactive.py   # 100 checks: filters, exports and controls work
python test_analytics.py               # 90 checks: every analytics dataset and chart
python test_dashboard.py               # 69 checks: every page renders, offline and live
python test_traffic_light.py           # 56 checks: signal timing against a synthetic clock
```

**769 checks in total, all passing** (last run 2026-08-09; see `METRICS.md`).

On Windows, run them with `set PYTHONIOENCODING=utf-8` first, or the console
dies printing an emoji from a button label. This is not a formality: without
it, `test_dashboard.py` reports a spurious failure and
`test_dashboard_interactive.py` crashes in the middle of its run and never
prints a total, so the suite looks broken when nothing is wrong with it.

`test_traffic_light.py` drives the signal against a synthetic clock, so the
5-second yellow is asserted exactly rather than slept through.
`test_analytics.py` builds a two-session archive with a deliberately known
shape — a quiet 08:00 and a busy 17:00 — and checks that each chart recovers
the properties that were built into it. `test_server.py` verifies the serial
line field by field against the format the Arduino sketch parses, and drives
every HTTP endpoint through Flask's test client.

The vision tests use a scripted fake detector so the exact ground-truth speed
of every synthetic vehicle is known — that is the only way to prove the km/h
conversion is right, since a real camera gives nothing to check the answer
against. `test_video_source.py` extends that idea to recorded video: it
generates real MP4s, plays them at four different speeds and requires the same
km/h from all four, then drives every source endpoint against a real Flask
test client — including the failures, since a mistyped address must leave the
running feed alone and say what was wrong. The `--yolo` run additionally scores the livery recogniser against
the real photographs in `assets/emergency_test` and reports recall/precision.

Siren detection is tested from both ends, which matters because a detector
that flags nothing passes a false-alarm test perfectly. `test_traffic_system.py`
drives synthetic beacons of known colour, rate and duty past the detector
alongside the things that imitate them — steady lamps, red paint, brake taps,
amber indicators — and asserts the temporal analysis on signals of known shape.
For real footage, `make_siren_test_video.py` paints a physically plausible
light bar onto a vehicle in a clip you already have, giving a video where
exactly one vehicle is an emergency vehicle and the rest is genuine traffic.
The dashboard tests run all 9 pages headlessly with the broadcast server both
stubbed-live and offline. `test_dashboard_interactive.py` goes a step further
and drives the widgets — log filters, the analytics interval, every hardware
command, the confidence slider and a settings save round trip — because a page
can render perfectly while its controls do nothing.

---

## Troubleshooting

**Speeds look 20–30% off.** You are in auto mode. Run `calibrate_speed.py`.
For a quick trim instead, set `auto_scale_correction` in `VisionConfig`
(0.8 makes every reading 20% slower).

**FPS is very low (~2).** Torch is CPU-only. Check with
`python -c "import torch; print(torch.cuda.is_available())"` and reinstall the
CUDA build (see `requirements.txt`). Or drop to `--imgsz 640 --model yolo12n.pt`.

**Dashboard says the stream is offline.** The host is wrong. Set it on the
Settings page or with `TRAFFIC_STREAM_HOST`, check `ipconfig` on the CV node,
and confirm both machines are on the same network and Windows Firewall allows
Python on port 8502.

**Vehicles are counted twice / IDs jump.** Raise `new_track_thresh` and
`match_thresh` in `trackers/traffic_botsort.yaml`.

**Distant vehicles are missed.** Raise `--imgsz` to 1280 and lower `--conf` to
0.25.

**Controller not connecting.** The system starts on **COM6** with the ESP32
sketch (`--board esp32`), which is the one board the PC is wired to. Check the
port in Device Manager and, if it differs, change it in `START HERE.bat` or pass
`--arduino-port COM7`; run with `--no-arduino` when no board is attached. The
Arduino IDE's Serial Monitor holds the port exclusively — close it before
starting the server.

**The Uno's LCD is stuck on `Traffic System / Waiting...`.** That is the sketch's
start-up message: the board is alive and its LCD is wired correctly, it has just
never received a packet it could read. With the default wiring the line reaches
it through the ESP32, so check that relay first — **ESP32 GPIO33 → Uno pin 7,
and the two grounds tied together**. A missing common ground is the usual cause
and looks exactly like a missing wire. Confirm the hub end with
`{"command":"GET_STATUS"}` in the Serial Monitor: `signal_relayed` counts the
lines it has passed on, and it should climb by one a second while the node runs.

The other cause is `--board`. It tells the node which sketch is on the other end
— the ESP32 is sent JSON, a directly-attached Uno the seven-field line — and if
it names the wrong one, nothing the board understands is ever sent. The port
still opens and every write still succeeds, which is what makes it look like
dead hardware. The serial speed follows the board automatically, so
`--arduino-baud` is only needed if you changed `Serial.begin()` yourself.

After eight seconds the sketch stops saying `Waiting...` and names the fault
itself:

| LCD | Meaning |
|---|---|
| `No data from PC` / `Check D7 + GND` | nothing is arriving at all — the relay wire or its ground, the wrong port, or the node is not running |
| `Serial garbled` / `Need baud 9600` | bytes are arriving but never form a packet — the speed does not match `Serial.begin(9600)` |
| `NO SIGNAL` / `Check PC cable` | it *was* receiving and the packets stopped |

**LEDs show the wrong colour, or the LCD shows numbers.** The sketch has not
been reflashed since the signal change. The serial protocol now has **seven**
fields (`PHASE,LEVEL,CI,VEHICLES,EMG,LINE1,LINE2`) and the phase is sent
explicitly instead of being derived from the congestion level on the Arduino.
Re-upload `traffic_controller_arduino/traffic_controller_arduino.ino`.

**ESP32 wiring.** LED strip data → GPIO5, LCD SDA/SCL → GPIO22/21. Lane-changer
Uno: GPIO26 → its D2, its D3 → 1k/2k divider → GPIO25. Signal-head Uno: GPIO33 →
its pin 7, plus a shared ground. The ESP32's 3.3 V output clears the Uno's 3.0 V
threshold for a HIGH, so that direction needs no level shifting; the return
direction would, which is why the signal link is one wire and the Uno answers on
its own USB port instead.

**Arduino wiring.** Ramp 1 head (side road merging into the N→S carriageway):
Green → pin 9, Yellow → pin 10, Red → pin 11. Ramp 2 head (side road merging
into the S→N carriageway, optional): Green → pin 2, Yellow → pin 3, Red → pin 4.
Blue (emergency indicator) → pin 8, **emergency LED strip → pin 6**, LCD SDA/SCL
→ A4/A5. Each LED needs a 330 Ω resistor. Anything longer than a few LEDs on pin
6 draws more than the 40 mA a pin can source — drive it through a
transistor/MOSFET or relay with its own supply. Every output cycles once at boot
as a wiring self-test, in the order listed above, so an LED that lights out of
turn names the pin that is wrong.

**The second traffic light.** Both heads show the same phase, because congestion
is measured as one reading for the whole scene. Nothing crosses here — two ramps
merging into two separate carriageways — so both green together is correct, and
there is no conflicting movement to protect against. A lost serial link drops
both to red. Wiring only the first head leaves the sketch behaving exactly as it
did with one signal: pins 2, 3 and 4 simply drive nothing.

**One cable, three boards.** The PC's only USB cable goes to the ESP32 on COM6,
and the ESP32 is the hub: it drives the LED strip and its own LCD, reaches the
lane changer on UART2 (GPIO25/26) and the signal-head Uno on UART1 (GPIO33 →
Uno pin 7). Both formats travel that one cable and are told apart by shape —
JSON, or a single bare word, is a command for the hub; seven comma-separated
fields is signal telemetry, which it passes down UART1 untouched. Nothing else
needs plugging into the PC:

```bash
python broadcast_server.py --board esp32 --arduino-port COM6
```

**Giving the signal Uno its own USB cable instead.** Still supported, and the
node prefers a direct link over the relay whenever one is open — the line then
reaches the Uno with no second board in the path. Name both ports:

```bash
python broadcast_server.py --board esp32 --arduino-port COM6 --second-port COM4
```

The Uno sketch reads USB and the ESP32 wire every loop and does not care which
one delivered the line, so no reflash is needed to move between the two.

**Emergency reaches every board by itself.** A detected emergency vehicle used
to light the Uno's strip while the ESP32 sat dark until somebody pressed the
dashboard button. Now the same detection that drives the signal heads also sends
`ACTIVATE_EMERGENCY` to the ESP32 and `DEACTIVATE_EMERGENCY` when the road
clears. The board already knew how to show an emergency; it was just never told
when one was happening.

**The two LCDs say different things, on purpose.** The ESP32's reports its LED
strip and nothing else — `Ambulance passing` when the strip is red, `Police
passing` when blue, `Safe travels` otherwise — so it is readable at a glance
from across the room. It is driven from the strip state in one place, so the
two can never disagree. Everything with detail on it — the signal phase, the
yellow countdown, the congestion level and index, and the named vehicle type
(`!! AMBULANCE !!`, `!! POLICE !!`, `!! FIRE TRUCK !!`) — is on the Uno's LCD,
which already receives all of it in the seven-field line. Only changes are sent, so the link
is not spent repeating a state that has not moved, and because the node records
what it sent, the dashboard shows the board's real state rather than the last
thing a human clicked. The ramps keep metering normally throughout: an emergency
vehicle proceeds through any indication, so seizing the signal would only
disturb traffic that is already giving way.

**Wrong camera, or a black picture.** Run `python broadcast_server.py
--list-cameras`, which names every camera and says which one is delivering a
picture. Camera indices are assigned by Windows at runtime and shift when a
device is replugged or a virtual-camera app (OBS, Iriun, DroidCam) starts, so
prefer `--source <name>` over `--source <number>`.

**Using a phone as the camera (DroidCam / Iriun).** Connect the phone in the
desktop client first and confirm the video is visible in that window, then start
the server with `--source droidcam`. The virtual camera is registered with
Windows whether or not the phone is connected, so it appears in `--list-cameras`
either way and hands out black frames until the client has a live connection.

DroidCam's virtual camera cannot be opened through DirectShow at all — it needs
the Media Foundation backend, which `open_capture` falls back to automatically.
A `--source 2` that fails with *"backend is generally available but can't be
used to capture by index"* is this, not a missing camera.

DroidCam defaults to 640x480; the server requests 1920x1080 and gets it. Check
the resolution in `--list-cameras`, because at 640x480 distant vehicles are
missed no matter what `--imgsz` is set to.

**Emergency recognition says "off".** The `clip` package is missing. Install it
with `pip install git+https://github.com/ultralytics/CLIP.git`. Siren detection
is unaffected — it needs no weights at all — so emergency mode keeps working;
what is lost is the ability to tell an ambulance from a fire engine when both
show a red beacon.

**An emergency vehicle was missed.** Record the clip and ask:

```bash
python check_siren_detection.py --source clip.mp4 --save-crops out/
```

It prints what it flagged and, for every vehicle it did not, *why* — `too-dim`,
`steady-light`, `gradual-not-switched`, `not-repeating`, `red-below-roofline`
and so on. `too-dim` or `shallow-modulation` on a real beacon in daylight means
the sun is washing it out rather than that anything is broken; raise the
exposure, or run with `--livery-triggers-emergency` so the markings can carry
the decision. The tool also warns if the frame rate was too low for a beacon's
flashes to survive sampling at all.

**An ordinary vehicle was flagged.** Note its beacon colour and rate from the
box label, then loosen the matching gate in `SirenConfig` (`siren_vision.py`):
`min_channel_value` for something not bright enough to be a lamp,
`max_midband_fraction` for something that ramps rather than switches,
`min_active_seconds` for something brief, `roof_fraction` for a light too low
on the body to be a beacon.

**Too many / too few livery guesses.** Adjust `on_threshold` in
`EmergencyConfig` (`emergency_vision.py`): raise it towards 0.85 for fewer false
alarms, lower it towards 0.35 to catch more unusual vehicles.

**"Nothing is answering at host:port."** The address was reachable enough to
try but nothing accepted a connection. Check the camera is powered and on this
network, that the port is right (RTSP is usually 554, an IP camera's MJPEG
endpoint usually 8080), that any username and password are in the URL, and
that a firewall is not in the way. `ping` the host, then try the URL in VLC —
if VLC cannot play it, neither can OpenCV.

**A YouTube link is refused.** Those are web pages, not streams. Run
`pip install yt-dlp` on the CV node and try again.

**The uploaded video plays but the speeds look wrong.** Speed is only as good
as the scale. Auto mode infers metres per pixel from vehicle heights and is
±25%; for a fixed camera position, `calibrate_speed.py` gets it to ±5%. A
video from a *different* camera needs its own calibration — the one on disk
belongs to whichever camera it was drawn for.

**The video is analysed slower than real time.** It is not: frames are dropped
to keep up and the media clock counts them, so the km/h are right and only the
overlay is choppier. `--no-drop-frames` decodes every frame instead, and
`--playback-rate 0` runs flat out. If you want more frames actually looked at,
drop `--imgsz` to 640 or use `yolo12n.pt`.

**The upload button says the file is too large.** Streamlit's own limit is
200 MB: `streamlit run app.py --server.maxUploadSize 2000`. The node's limit is
separate and much higher (`--max-upload-mb`). For a really large file, copy it
next to `broadcast_server.py` and type its name on the URL tab instead.

**The vehicle count restarted when I changed source.** It is meant to. Track
ids begin again on a new source, so keeping the old set would merge two roads'
vehicles into one undercounted total. Every vehicle already seen has been
written to the log before the reset.
