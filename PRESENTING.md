# Presenting — run sheet

Everything needed to run the demo, in the order you'll need it.

---

## 0. Do this first (once, before the room fills)

**Restart the system.** It is running right now on the old code, so it does not
yet have the speed fix. Close the black console window (or press `q` in the
preview), then start it again as below.

```
Ctrl-C in the console, or press q in the preview window
```

Then re-launch and let it warm up — the model takes a few seconds to load.

---

## 1. Starting it

**Double-click `START HERE.bat`.** That is the whole thing. It opens both
halves and the browser.

| What | Where |
|------|-------|
| Dashboard | <http://localhost:8501> |
| CV node (video + telemetry) | <http://localhost:8502> |
| Live video feed direct | <http://localhost:8502/video_feed> |

One USB cable goes to the **ESP32 on COM6** — already set inside
`START HERE.bat`. Everything else hangs off it: the lane changer on GPIO25/26
and the signal-head Uno on GPIO33 → its pin 7 (grounds tied together). The Unos
need power, not a data cable. **Close the Arduino IDE's Serial Monitor first**;
Windows only lets one program hold a COM port, and whoever gets there first
wins.

Useful variants, if you need them:

```
python start.py --no-arduino          # no board connected
python start.py --no-display          # hide the node's preview window
python start.py --source nightroad.mp4  # start straight on a video
```

Wait for this line before you start talking:

```
[YOLO] Loading yolo11m.pt on CUDA (FP16) @ imgsz=1280
```

---

## 2. Your three videos

All three load **by name** — no folder, no upload needed.

| Video | Length | What it shows |
|-------|--------|---------------|
| `Dayroad.mp4` | 2:10 | Daytime traffic, clean detection and speeds |
| `nightroad.mp4` | 1:56 | Night — the harder case, worth showing on purpose |
| `emergencyvedio.mp4` | 2:36 | Emergency vehicle with a running beacon |

**These three were re-cut on 11 August 2026 and now open on picture.** Every one
of them used to begin with a stretch of pure black — 13.6 s on `Dayroad.mp4`,
29.5 s on `nightroad.mp4` and **53.6 s on `emergencyvedio.mp4`** — during which
the feed is black and the vehicle count sits at zero. With Loop on that black
came back round every cycle. The lead-in is gone, so pressing ▶ now shows
traffic and detections in the first second. The originals are kept untouched in
`vedios/_originals/` (that folder is deliberately not listed by the picker).

**On `emergencyvedio.mp4`, the timings worth knowing:** a vehicle is recognised
as an emergency vehicle from **0:01**, but the red **beacon-confirmed** banner —
the one that means priority was actually granted — first fires at **0:43**, and
then repeatedly between **2:00 and 2:25**, which is the richest stretch. If you
want the banner on screen while you talk about it, run from about 1:58. Beacon
colours read blue, red and alternating red+blue at 1.2–4.1 Hz.

**To switch during the demo:**
Live Camera → **Change source** toggle → **🗂️ Uploaded videos** tab → press ▶
next to the one you want.

Or paste the bare name on the **🌐 Camera / stream URL** tab and press Connect.

Under the picture you get **⏸ Pause · ⏪ 5s · 5s ⏩ · ⏮ Restart · 🔁 Loop**, plus
a scrub bar and playback speed in the panel. **Turn Loop on** so a video does
not run out mid-sentence.

---

## 3. The Excel / data files

All in the project folder, next to `START HERE.bat`:

| File | What it is |
|------|-----------|
| `traffic_history.csv` / `.xlsx` | **This session only** — what you're recording now |
| `traffic_all_sessions.csv` / `.xlsx` | **Every session ever** — 51 sessions, ~11,000 samples |
| `sessions/session_<id>.*` | A standalone copy of each finished session |

The **CSVs are written every second** and are always complete — they are the
safe copy. The **workbooks are built when you ask for one**, so they are always
current when you click download.

**To get a workbook during the demo:**

- Operations Dashboard sidebar → **Prepare Excel Log for Download** → **⬇️ This Session (Excel)**
- Operations Dashboard sidebar → **Prepare All-Sessions Workbook** → **⬇️ All Sessions (Excel)**
- Or press **`x`** in the node's preview window to write it to disk immediately

**To record a named run:** Analytics page sidebar → type a name → **⏺️ Start
Recording** → **⏹️ Stop Recording**. It lands in `sessions/` and in the
all-sessions workbook.

The sheets inside the workbook: **Live Status**, **Traffic Log**, **Vehicle
Speeds**, **Session Summary**. The all-sessions one adds **All Sessions**,
**All Records** and **Lifetime Summary**.

---

## 4. A running order that shows it off

1. **Home** — what the browser opens on. The project title, the whole group and
   what each of you did, the aim, and the four measured results. Leave it up
   while you introduce yourselves, then move on.
2. **Operations Dashboard** — first item in the sidebar. System health, live
   vehicle count, density, speed, the signal phase counting down, CPU/RAM.
3. **Live Camera on `Dayroad.mp4`** — the picture with boxes and tracking IDs,
   the live speeds in km/h, the Active Detections table underneath.
4. Expand **"How speed is measured in km/h"** — this is the strong technical
   point. Speeds are real metres, not pixels.
5. **Switch to `emergencyvedio.mp4`** — the red banner fires, the vehicle is
   named (ambulance / police / fire), the beacon colour and flash rate are
   shown, and the emergency strip lights on the board.
6. Expand **"How ambulances, police cars and fire trucks are detected"** — the
   two-cue argument, and *why livery alone is refused*. That's the bit that
   sounds like engineering judgement rather than a tutorial.
7. **Switch to `nightroad.mp4`** — showing the hard case deliberately is
   stronger than hiding it.
8. **Traffic Analytics** — set scope to **All sessions** for the trend charts
   across every run, then download the workbook live.
9. **Emergency Control / Hardware Monitor** — the board, the LEDs, the servo.

---

## 5. Numbers you can quote

| | |
|---|---|
| Detection | YOLO11-m, FP16, 1280 px inference, on the RTX 4050 |
| Speed | **~27 fps** measured on `Dayroad.mp4` |
| Inference | ~27 ms/frame |
| Logging | 1 sample/second to CSV, ~11,000 samples across 51 sessions |
| Emergency trigger | Beacon must *flash* — 0.7–6 Hz, 3+ flashes, sustained >1 s |
| Tests | **833 automated checks** across 7 suites, 831 passing |

The speed figures were re-measured on 11 August 2026 against the running
system: the node reported a **27.1 fps** mean with **27.8 ms** inference over a
20-second sample of `Dayroad.mp4`, so the two numbers above are what it
actually does, not a best case.

On the tests: 831 of the 833 pass in one run. The two that do not are the pair
that POST to the CV node, and they pass when it is up — but four checks in the
Live Camera section are written against the *simulated* feed and fail while it
is up, so no single run can show all 833. Quote 831; it is the honest number
and it is reproducible.

If asked why a beacon and not appearance: on the night footage, appearance
alone flagged **29% of ordinary traffic** as emergency vehicles. Requiring a
working beacon brought that to **zero** while still catching every planted one.

---

## 6. If something goes wrong

| Symptom | Fix |
|---------|-----|
| Dashboard says **"Live broadcast stream offline"** | The node isn't up. Check the console window. Restart `START HERE.bat`. |
| **Signal-head Uno dead / LCD stuck on start-up message** | The relay wire. ESP32 GPIO33 → Uno pin 7, and the grounds must be tied together. After 8 s the LCD says `Check D7 + GND` itself. |
| **No board at all / everything dark** | Wrong `--board`, or the Arduino Serial Monitor is holding COM6. Close it and restart. |
| **Video won't load** | Type just the filename (`nightroad.mp4`) — it is found automatically. |
| **Video ran out** | Press **⏮ Restart**, or turn **🔁 Loop** on. The node keeps running either way. |
| **Numbers frozen** | The feed ended. The dashboard says so rather than showing stale values as live. Reload the source. |
| **Download button does nothing** | The node builds the workbook on request; give it a few seconds on a long session. |
| **Everything feels slow** | Close other GPU apps. As a last resort `python start.py --imgsz 960` — faster, slightly weaker at distance. |
| **Need to stop cleanly** | Ctrl-C in the console, or `q` in the preview. Both write the workbooks and release COM6. |

---

## 7. Preview-window keys

Only work when the node's own preview window has focus.

| Key | Does |
|-----|------|
| `q` | Quit cleanly |
| `e` | Toggle emergency manually |
| `g` / `y` / `r` | Force green / yellow / red |
| `a` | Back to automatic |
| `x` | Write the Excel workbook now |
| `p` | Save a snapshot PNG |
| `space` | Pause / resume the video |
| `[` / `]` | Back / forward 5 s |
| `l` | Toggle loop |
| `0` | Restart the video |

---

## 8. One thing to know about the data

The old duplicate archive under `logs/` and the per-session copies in
`sessions/` were cleared out. **The live archive — `traffic_all_sessions.csv`
and `.xlsx`, 51 sessions and ~11,000 samples — is untouched**, so the
Analytics page's "All sessions" view still has its full history to show.
