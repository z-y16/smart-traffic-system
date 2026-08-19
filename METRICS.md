# Metrics

Every number this project claims, what was measured to get it, and — where it
matters — what it does **not** prove.

Measured **2026-08-09** on the development machine:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop (6 GB) |
| Precision | FP16 |
| Python / PyTorch | 3.13.14 / 2.6.0+cu124 |
| Detector | yolo11m.pt @ imgsz 1280 |
| Livery recogniser | CLIP ViT-B/16 (zero-shot) |
| Footage | `vedios/Test1_day_fotage.mp4`, `vedios/Test2_dark_fotage.mp4`, 1920×1080 |

Reproduce with `python benchmark.py --source <clip> --grid --stages`.

---

## 1. Tests — 833 checks

Counted 11 August 2026.

| Suite | Checks | Covers |
|-------|-------:|--------|
| `test_traffic_system.py` | 217 | the vision pipeline: speed, congestion, ROI, emergency fusion, logging |
| `test_server.py` | 165 | hardware command mapping, serial format, the ESP32 relay, HTTP endpoints, launcher |
| `test_video_source.py` | 113 | uploads, stream URLs, playback rates, the media clock |
| `test_dashboard_interactive.py` | 109 | filters, exports, controls, the front page |
| `test_analytics.py` | 90 | every analytics dataset and chart |
| `test_dashboard.py` | 83 | all 9 pages render, offline and live |
| `test_traffic_light.py` | 56 | signal timing against a synthetic clock |
| **Total** | **833** | |

**831 of the 833 pass in one run, and 833 is not reachable in a single run.**
Re-measured 11 August 2026. The six suites above `test_dashboard_interactive.py`
pass 724/724. The interactive suite is the one that cannot be fully satisfied at
once, because it wants the CV node in both states:

| CV node | result | what fails |
|---------|-------:|------------|
| not running | **107 / 109** | "no error box after commands" and "LCD message send succeeds" — both POST to `http://127.0.0.1:8502/command` and fail on the connection |
| running | 105 / 109 | four Live Camera checks written against the *simulated* feed — "camera falls back to the simulated feed", "detection table columns", "confidence slider actually filters detections", "the seek carries the position asked for" |

So the best single run is **831**, with the node down. Fixing the remaining two
properly means splitting the interactive suite in two — the pages that talk to
the node, and the pages that exercise the offline fallback — rather than asking
one run to be in both states.

**The node-up run used to crash rather than report.** `start_recording` was
clicked and never stopped, so a later check looked for the Start button while
the page was correctly showing Stop, and the suite died on a `KeyError` without
printing a total. That is why the "clean 833" claim was never contradicted: with
the node down the recording never begins and the ordering fault stays hidden.
The suite now stops the recording between the two, probing for whichever button
is showing, and completes in both states.

**Windows needs `set PYTHONIOENCODING=utf-8`.** Without it two suites fail on
console encoding rather than on logic — `test_dashboard.py` reports one
spurious failure, and `test_dashboard_interactive.py` crashes mid-run and never
prints a total. Both pass cleanly with it set. This is a property of the
Windows console codepage, not of the dashboard.

---

## 2. Speed accuracy — exact, against constructed ground truth

This is the one place the system has genuine ground truth, because the vehicle
is synthetic and its speed is therefore known rather than estimated.

| Test | Expected | Measured |
|------|----------|----------|
| Homography mode | 18 km/h | **18.00** |
| Homography mode | 36 km/h | **36.00** |
| Homography mode | 72 km/h | **72.00** |
| Automatic (height-based) mode | 18 km/h | **18.00** |
| Fixed-scale mode | 18 km/h | **18.00** |
| Parked vehicle | 0 km/h | **0.12** |

**Media clock** (`test_video_source.py`): a vehicle scripted at 16.2 km/h
through a real MP4, played at real time, 2×, half speed and unthrottled —
all four measure **16.2 km/h**, agreeing to 0.00. This is what stops the
reported speed becoming a property of the computer instead of the traffic.

Field accuracy is **±5% calibrated**, **±25% automatic**. Those are the
calibration modes' own error bars, not a measurement of these tests.

---

## 3. Detection throughput and yield

60 consecutive night frames, FP16. `veh/frame` is the mean number of *tracked*
vehicles returned per frame.

| model @ imgsz | ms/frame | median | p95 | FPS | veh/frame |
|---------------|---------:|-------:|----:|----:|----------:|
| yolo11s @ 640 | 14.2 | 14.1 | 16.0 | 70.6 | 18.4 |
| yolo11s @ 960 | 16.0 | 15.7 | 18.7 | 62.5 | 20.8 |
| yolo11s @ 1280 | 18.4 | 18.4 | 20.4 | 54.3 | 22.8 |
| yolo11m @ 640 | 17.2 | 16.9 | 19.3 | 58.0 | 17.0 |
| yolo11m @ 960 | 19.0 | 18.9 | 21.8 | 52.8 | 19.3 |
| **yolo11m @ 1280 (default)** | **27.3** | **27.2** | **29.4** | **36.6** | **23.0** |
| yolo11l @ 640 | 22.5 | 22.2 | 24.9 | 44.4 | 17.9 |
| yolo11l @ 960 | 24.1 | 24.1 | 26.4 | 41.6 | 19.8 |
| yolo11l @ 1280 | 32.8 | 32.4 | 35.5 | 30.5 | 22.5 |
| yolo12s @ 1280 | 24.0 | 24.0 | 25.6 | 41.7 | 23.6 |
| yolo12m @ 1280 | 38.7 | 38.5 | 41.9 | 25.8 | 22.1 |

### Re-measured 11 August 2026

The default row was re-run on the same clip and the same machine and came back
**31.7 ms / 31.5 fps**, against the 27.3 ms / 36.6 fps above — about 16% slower.
The *yield was identical at 23.0 vehicles per frame*, which is the number that
says the model and its settings are unchanged; only the clock moved. The likely
cause is machine state rather than code: the earlier run had a quieter machine,
where this one shared the GPU with Teams, WhatsApp and several Edge WebView
processes, on the Balanced power plan.

Worth knowing which number to trust for which purpose: this table is a
*comparison between settings*, and it is still valid for that because every row
would shift together. For "how fast does it run in the demo", use the live
figure — **27.1 fps end to end on `Dayroad.mp4`**, measured from the node's own
telemetry while it was actually running.

### What this shows

**Resolution buys detections. Model size does not.** Every model gains ~5
vehicles/frame from 640 → 1280. At fixed resolution, going `s` → `l` gains
nothing: yolo11l @ 1280 finds *fewer* than yolo11m (22.5 vs 23.0) for 20% more
time. The limit is how many pixels a distant vehicle occupies.

**YOLO12 is the wrong family here.** yolo12m @ 1280 is the slowest row in the
table (38.7 ms) and finds fewer than yolo11m. Its attention blocks cost real
time on an Ada laptop GPU and buy nothing at this resolution.

### ⚠ An open question this table raises

**yolo11s @ 1280 gets within 1% of yolo11m's yield (22.8 vs 23.0) at 48% more
throughput (54.3 vs 36.6 fps).** That is a real finding and it is deliberately
**not acted on**, because `veh/frame` is a count, not a correctness score.
With no labelled boxes for this footage, a model that draws *more* boxes cannot
be told apart from one that draws *better* ones — the extra detections could be
distant cars correctly found, or they could be duplicates and phantoms.

Settling it needs a few hundred hand-labelled frames and a real mAP. Until
someone does that, the default stays on yolo11m.

### Why there is no mAP in this document

There are no hand-drawn boxes for this footage, so there is nothing to score
against. `benchmark.py` deliberately refuses to print an accuracy figure, and
any mAP quoted for this project would have been copied from the model's
published COCO score — which describes Ultralytics' test set, not this road.

---

## 4. Per-stage cost

`python benchmark.py --stages`, yolo11m @ 1280:

| stage | night (23 veh/frame) | day (25.5 veh/frame) |
|-------|---------------------|----------------------|
| detection + tracking (YOLO) | 29.6 ms (66.2%) | 28.8 ms (82.1%) |
| livery recognition (CLIP) | 5.4 ms (12.0%) | 1.1 ms (3.2%) |
| beacon detection (siren) | 9.8 ms (21.8%) | 5.2 ms (14.7%) |
| everything else | ~0.0 ms | ~0.0 ms |
| **end to end** | **44.8 ms → 22.3 fps** | **35.1 ms → 28.5 fps** |

Detection dominates, so it is the only stage worth optimising — halving the
congestion maths would buy nothing measurable.

Both recognisers cost **more at night**, for different reasons: CLIP because
dark crops fail the readability gate and get retried instead of backed off, and
the beacon detector because at night there are actually beacons to measure.

---

## 5. Emergency recognition

### On unseen internet footage

Full run over an 11:43 YouTube compilation
("Police Cars Fire Trucks And Ambulances Responding Compilation Part 12"),
no ROI, no speed, nothing cherry-picked. `python demo_emergency_clip.py --scan`.

| | |
|---|---:|
| frames analysed | 21,083 |
| scene cuts detected | 301 |
| vehicles tracked | 2,511 |
| recognised as emergency | 166 (6.6%) |
| …distinct vehicles after merging track-id churn | ~96 |
| given priority (beacon confirmed) | 21 |
| by type | 60 fire truck, 59 ambulance, 47 police |
| confidence range | mostly 0.93–1.00 |
| beacon rates read | 1.2–4.8 Hz |

The **6.6%** is the figure that matters: the recogniser looked at 2,511
vehicles and singled out 166. A detector that flagged everything would score
100% recall and prove nothing.

**What this is not.** It is not a measured false-positive rate. Confirming that
would mean labelling all 166 by hand; a sample of the saved crops in
`out_demo/yt_shots/` was checked by eye and most were unmistakable, but a
couple were ambiguous crops from dense traffic where a neighbouring emergency
vehicle bled into the box.

Only 21 of 166 reached beacon-confirmed priority, and that is expected rather
than a fault: **in daylight a beacon bright enough to register blooms to white**,
and white fails the red/blue purity gates by design. Measured on a daylight
ambulance — beacon channels 0.04, livery 0.95.

### CLIP vs the trained model, on unseen cameras

| | Trained (`emergency_cls.pt`) | CLIP |
|---|---:|---:|
| daylight ambulance found | 0.55 ✗ | 0.95 ✓ |
| ordinary night cars wrongly flagged | 6 of 8 (at 0.94–1.00) | 0 of 8 |
| false alarms over 4,186 daylight detections | — | 0 |

### On the camera the trained model was built for

| over the 0.70 threshold | CLIP | Trained |
|---|---:|---:|
| ambulance recognised | 53.4% | 71.1% |
| police car recognised | 41.1% | 71.4% |
| fire engine recognised | 39.7% | 70.3% |
| false alarms, this camera | 0.0% | 0.0% |
| false alarms, ImageNet | 0.3% | 5.3% |
| cost per crop | 4.3 ms | 0.7 ms |

CLIP is the default because it is the one that travels. The trained model is
better *and seven times cheaper* on the camera it was trained for, and should
only be chosen for such a camera.

### Beacon detector, night footage

The switch-versus-slide test (a lamp is never caught half-lit; a red car
drifting under a street light constantly is) separates the two cases
completely: the real beacon scored **0.00**, every false positive **0.18–0.43**.

Making the beacon the trigger instead of the livery took false alarms on the
night footage from **88 of 299 vehicles (29%) to 0**.

---

## 6. Signal timing

`test_traffic_light.py`, driven against a synthetic clock rather than by
sleeping, so the interval is asserted exactly:

- the yellow interval is **exactly 5 s**, every time
- with the congestion level flipping **every 0.5 s for two minutes**, no phase
  is ever momentary and the `R→Y→G→Y→R` ordering never breaks
- minimum dwell (default 3 s) is held on green and red

---

## 7. Knowledge graph

`.graphifyignore` was excluding data directories but not the vendored trees, so
the graph was mostly other people's code and a query for the system's main
components returned 247 nodes without one of them being `traffic_vision.py`.

| | before | after |
|---|---:|---:|
| nodes | 12,490 | 1,623 |
| files pruned | — | 565 |
| communities | — | 78 |

Now excluded: `serena-src/` (412 .py, a vendored tool), `_superseded/`
(older copies of the dashboard, kept only as a rollback — their virtualenvs
were deleted on 2026-08-10), plus caches, media and generated output. The real
dashboard (`SmartTrafficSystem/`, 52 .py) is still indexed.

Rebuild with:

```bash
PYTHONPATH=graphify python -m graphify update .
```
