# CHAPTER 3 — COMPUTER VISION, DASHBOARD INTEGRATION AND DRIVER-FACING DISPLAYS

**Individual Chapter — Zeyad Khairy (TP074127)**

*Project: Computer Vision-Based Smart Traffic Congestion Analytics with Adaptive Signal Optimization*

> *Note on numbering: this chapter is drafted as Chapter 3. If the group orders the individual chapters differently, only the leading digit of each heading, figure and table number needs to change.*

---

## 3.1 Introduction

This chapter documents the work I carried out as an individual member of the group: the **computer vision subsystem** that turns a video stream into traffic measurements, the **integration layer** that carries those measurements to the group's control-centre dashboard, and the **LED guide strip and LCD displays** that present the system's decisions at the roadside.

These three areas are treated together because they form one continuous chain of custody for a single piece of information. A vehicle enters the camera's field of view; the vision subsystem measures it; the measurement is condensed into a congestion index and a signal phase; that phase is published simultaneously to a human operator through the dashboard and to a driver through the LEDs and the LCD. A fault anywhere along that chain — a mis-scaled speed, a stale telemetry packet, a display showing a phase the controller has already left — presents to the user in exactly the same way: as a system that cannot be trusted. Designing and testing the chain end to end, rather than as three independent deliverables, was therefore a deliberate decision rather than a convenience of task allocation.

The subsystem sits at the head of the group's architecture. The hardware member's signal heads, servo lane divider and speed bump are driven by the phase computed here; the dashboard member's analytics pages are populated entirely from the telemetry published here; and the emergency-priority member's response logic is triggered by the recognition results produced here. In practical terms, nothing else in the project runs until this part produces a number.

### 3.1.1 Individual objectives

1. To detect and track road vehicles from a fixed monocular camera in real time, at a frame rate sufficient to measure vehicle *motion* rather than merely count vehicles.
2. To convert image-plane motion into real-world speed in **km/h**, with a stated and verified error bound, rather than reporting pixel displacement.
3. To condense per-vehicle measurements into a single bounded **congestion index** stable enough to drive physical actuators without chattering.
4. To integrate the vision node with the group's dashboard over a network interface, such that the two may run on separate machines, and such that the dashboard can distinguish live data from stale data and from a simulated fallback.
5. To design and implement the **LED guide strip and LCD outputs** so that the state shown at the roadside is always the state the controller is actually in, and so that a wiring or configuration fault is announced on the display rather than presenting as dead hardware.

### 3.1.2 Scope and boundaries

Recognition of emergency vehicles is a computer vision problem and is therefore described here, since its output drives the LED and LCD hardware that also falls within my scope. The downstream *priority response* — what the junction does once an emergency vehicle has been recognised — belongs to another member's chapter and is referenced only at the interface. Likewise, the dashboard's own pages, charts and database are another member's work: this chapter covers the interface across which they are fed, not their internal design.

---

## 3.2 Literature Review

### 3.2.1 Vehicle detection in traffic video

Early automatic traffic surveillance relied on **background subtraction**: a statistical model of the static scene is maintained and pixels deviating from it are labelled foreground. The approach is computationally trivial and requires no training data, which is why it dominated fixed-camera traffic work for two decades. Its weaknesses are equally well established: it cannot classify what it has found, it merges vehicles that touch in the image, and it degrades under the illumination changes, headlight glare and wet-road reflections that characterise exactly the night-time conditions in which traffic monitoring is most valuable. Feature-based classifiers such as Haar cascades and HOG with a linear SVM added the ability to name an object, but at the cost of a sliding-window search that scales poorly with image size.

Convolutional detectors replaced both. The literature divides them into **two-stage** and **one-stage** families. Two-stage detectors, of which Faster R-CNN (Ren et al., 2015) is canonical, first propose candidate regions and then classify them; they remain competitive in accuracy, but two passes make real-time operation expensive. One-stage detectors — SSD (Liu et al., 2016) and the YOLO family beginning with Redmon et al. (2016) — regress class and box coordinates directly from a single forward pass over a grid, trading a degree of localisation accuracy for an order-of-magnitude speed advantage.

For this project the trade-off is not close. A congestion system that samples the road once every few seconds can report *counts*, but it cannot report *speed*, because speed requires the same vehicle to be located repeatedly within a short window. Frame rate is therefore a functional requirement rather than a performance nicety, and that requirement selects the one-stage family. The specific generation used, **YOLO11** (Jocher & Qiu, 2024), was chosen empirically rather than by publication date, for reasons set out in Section 3.3.1.

### 3.2.2 Multi-object tracking

A detector is memoryless: it reports the contents of one frame and has no notion that the car it has found is the same car it found a moment ago. Establishing that correspondence is the **multi-object tracking** problem, and the dominant paradigm is *tracking-by-detection*, in which detections from successive frames are associated into tracks.

SORT (Bewley et al., 2016) established the modern baseline: a constant-velocity Kalman filter (Kalman, 1960) predicts each track's next position and the Hungarian algorithm associates predictions with detections by intersection-over-union. It is extremely fast but fragile under occlusion, because identity is carried by geometry alone. DeepSORT (Wojke et al., 2017) added a learned appearance embedding so that a vehicle re-emerging from behind a bus can be re-identified. ByteTrack (Zhang et al., 2022) observed that low-confidence detections, normally discarded, are frequently occluded true positives, and recovered them in a second association pass. **BoT-SORT** (Aharon et al., 2022) combines these advances with camera-motion compensation and an improved Kalman state.

For a fixed traffic camera the critical failure mode is the **identity switch**: if two tracks exchange numbers, the apparent displacement of both is large and spurious, and any speed derived from it is meaningless. Association robustness — not raw tracker throughput — is therefore the criterion that matters here, and it is the reason BoT-SORT was selected over the cheaper SORT-class trackers.

### 3.2.3 Monocular speed estimation

Measuring speed from a single camera requires a mapping from image coordinates to world coordinates. Three approaches recur.

The first, most common in prototype work, is to report **pixel displacement per second** directly. This is not a speed. Under perspective projection the same physical velocity produces an image displacement inversely proportional to depth, so a vehicle 30 m from the camera and one 3 m from it yield readings differing by an order of magnitude while travelling identically.

The second is **inverse perspective mapping** via a ground-plane homography. Because a road surface is locally planar, the mapping between road plane and image plane is a projective transformation described by a 3×3 homography matrix recoverable from four point correspondences (Hartley & Zisserman, 2003). Once recovered, any image point known to lie on the road transforms to metric road coordinates. This is the most accurate practical method and is standard in the enforcement literature; its cost is a per-camera, per-position calibration step.

The third uses **known object dimensions** as a scale reference: if a class of object has a known physical size and its extent in pixels is measured, the local metres-per-pixel scale follows directly. This requires no calibration and adapts automatically to perspective, since each object supplies the scale at its own location, but it inherits the variance of the assumed dimension across real vehicles.

The design adopted here implements the second and third as selectable modes, allowing the system to produce a usable answer immediately on an uncalibrated camera and an accurate one after a short calibration.

### 3.2.4 Quantifying congestion

Traffic engineering does not define congestion by vehicle count alone. The *Highway Capacity Manual* (Transportation Research Board, 2016) grades a facility by **Level of Service**, a letter grade derived from density and operating speed rather than from flow, precisely because flow is ambiguous — a low count occurs both on an empty road and in a standstill. The fundamental speed–density relationship formalised by Greenshields (1935) captures the same insight: mean speed falls approximately linearly as density rises toward jam density.

Two further conventions are adopted directly. **Passenger car equivalents (PCE/PCU)** weight vehicles by the road space they consume, so that a bus is not counted as equivalent to a motorcycle. And a congestion metric intended to drive actuators must be **temporally stable**: a raw threshold comparison oscillates whenever the underlying signal sits on a boundary, a problem addressed in control engineering by hysteresis.

### 3.2.5 Emergency vehicle recognition

Two families exist. **Audio-based** systems detect the siren's frequency sweep and are robust to occlusion and darkness, but require a roadside microphone and degrade in traffic noise. **Vision-based** systems either train a supervised classifier on labelled emergency-vehicle imagery, or detect the flashing beacon as a temporal light signal.

A supervised classifier is accurate on the camera and the fleet it was trained for, and inexpensive at inference. Its weakness is generalisation: emergency liveries differ substantially between countries, and a classifier trained on one fleet has no basis for recognising another. The alternative is **zero-shot classification with a vision–language model**. CLIP (Radford et al., 2021) is trained on approximately 400 million image–caption pairs to embed images and text into a shared space, and can therefore score an arbitrary natural-language description against an image without task-specific training. This makes it capable of naming an ambulance it has never been shown, from a fleet it has never seen.

The two are not equivalent in what they establish, and the distinction is central to the design in Section 3.5.5. Livery recognition establishes what a vehicle **is**. Beacon detection establishes what a vehicle **is doing**. An ambulance parked outside a hospital satisfies the first and not the second, and only the second justifies intervening in traffic.

### 3.2.6 Gap addressed by this work

The individual techniques above are mature. What is comparatively rare in prototype literature is a system in which the measurement chain is *closed*: where the metric driving the physical actuator is validated against constructed ground truth, where the roadside display is provably the same state as the controller's, and where the limits of what has been demonstrated are stated explicitly. The contribution claimed here is therefore integrative rather than algorithmic — a defensible, measured pipeline from photons to LEDs — and the honesty of the measurement is treated throughout as a deliverable in its own right.

---

## 3.3 Investigation on Materials and Component Selection

Every selection below was made by measurement on the project's own footage rather than by published benchmark, because published accuracy figures describe the dataset a model was evaluated on and not the road this system watches.

### 3.3.1 Detector model and input resolution

Eleven detector/resolution combinations were benchmarked over 60 consecutive frames of the project's 1080p night footage on the development GPU under FP16 precision. The `veh/frame` column is the mean number of *tracked* vehicles returned per frame.

**Table 3.1 — Detector benchmark, 60 night frames, FP16, RTX 4050 Laptop**

| Model @ input size | ms/frame | p95 (ms) | FPS | Vehicles/frame |
|---|---:|---:|---:|---:|
| yolo11s @ 640 | 14.2 | 16.0 | 70.6 | 18.4 |
| yolo11s @ 960 | 16.0 | 18.7 | 62.5 | 20.8 |
| yolo11s @ 1280 | 18.4 | 20.4 | 54.3 | 22.8 |
| yolo11m @ 640 | 17.2 | 19.3 | 58.0 | 17.0 |
| yolo11m @ 960 | 19.0 | 21.8 | 52.8 | 19.3 |
| **yolo11m @ 1280 (selected)** | **27.3** | **29.4** | **36.6** | **23.0** |
| yolo11l @ 640 | 22.5 | 24.9 | 44.4 | 17.9 |
| yolo11l @ 960 | 24.1 | 26.4 | 41.6 | 19.8 |
| yolo11l @ 1280 | 32.8 | 35.5 | 30.5 | 22.5 |
| yolo12s @ 1280 | 24.0 | 25.6 | 41.7 | 23.6 |
| yolo12m @ 1280 | 38.7 | 41.9 | 25.8 | 22.1 |

Three findings follow, and each contradicted an initial assumption.

**Resolution buys detections; model capacity does not.** Every model family gains approximately five vehicles per frame moving from 640 to 1280 pixels. At fixed resolution, moving from `s` to `l` gains nothing: `yolo11l @ 1280` finds *fewer* vehicles than `yolo11m @ 1280` (22.5 against 23.0) for 20% more computation. The binding constraint is how many pixels a distant vehicle occupies, not how much model capacity is available to interpret them.

**A newer model generation was measurably worse.** `yolo12m @ 1280` is the slowest row in the table and finds fewer vehicles than `yolo11m`. Its attention blocks cost real time on a laptop-class GPU and return nothing at this resolution. The initial assumption that the later release should be preferred was therefore rejected on evidence.

**FP16 precision is what pays for the resolution.** Half-precision inference approximately doubles throughput on the tensor cores of the development GPU. Without it, 1280-pixel input at this model size would not sustain the frame rate that speed measurement requires.

An open question is recorded rather than resolved. `yolo11s @ 1280` reaches within 1% of the selected model's yield at 48% greater throughput. This is deliberately **not acted upon**, because `veh/frame` is a count and not a correctness score: with no hand-labelled boxes for this footage, a model drawing *more* boxes cannot be distinguished from one drawing *better* ones. Settling it requires several hundred manually labelled frames and a genuine mAP evaluation, which was outside the time available.

### 3.3.2 Tracker

BoT-SORT was selected over ByteTrack and SORT for the reason given in Section 3.2.2 — identity-switch robustness governs speed validity. Its configuration is held in a project-local YAML file rather than in code, so that association thresholds can be re-tuned for a new camera without modifying the pipeline.

### 3.3.3 Compute platform

| Component | Selection | Justification |
|---|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB | Ada-generation tensor cores make FP16 inference viable; 6 GB is sufficient for the detector plus CLIP held resident |
| Precision | FP16 | Roughly 2× throughput; no measurable yield loss on this footage |
| Framework | PyTorch 2.6 + CUDA 12.4, Ultralytics | Mature CUDA path; the tracker is integrated with the detector rather than bolted on |
| Language | Python 3.13 | Consistency with the group's dashboard stack |

The choice of a laptop-class GPU is itself a design constraint of consequence: it establishes that the pipeline is deployable on edge hardware at the roadside rather than requiring a datacentre, which is a claim the project can defend because it was never developed on anything larger.

### 3.3.4 Emergency-vehicle recogniser

Two candidates were evaluated: a supervised classifier trained on project footage (`emergency_cls.pt`), and zero-shot CLIP ViT-B/16.

**Table 3.2 — Recogniser comparison on *unseen* cameras**

| Criterion | Trained classifier | CLIP (zero-shot) |
|---|---:|---:|
| Daylight ambulance, confidence | 0.55 ✗ | 0.95 ✓ |
| Ordinary night cars wrongly flagged | 6 of 8 (at 0.94–1.00) | 0 of 8 |
| False alarms over 4,186 daylight detections | — | 0 |

**Table 3.3 — The same two on the camera the classifier was trained for**

| Criterion | CLIP | Trained classifier |
|---|---:|---:|
| Ambulance recognised (above 0.70) | 53.4% | 71.1% |
| Police car recognised | 41.1% | 71.4% |
| Fire engine recognised | 39.7% | 70.3% |
| False alarms, this camera | 0.0% | 0.0% |
| False alarms, ImageNet imagery | 0.3% | 5.3% |
| Cost per crop | 4.3 ms | 0.7 ms |

The trained classifier is both more accurate and roughly seven times cheaper **on the camera it was built for**, and is retained in the codebase as a selectable option for that case. CLIP is the default because it is the option that *travels*: a deployed traffic system is pointed at cameras nobody has collected training data from, and a recogniser confidently flagging six of eight ordinary cars on an unseen camera is not merely less accurate but actively harmful, since every false alarm proposes an unwarranted intervention in live traffic.

### 3.3.5 Display and indicator hardware

| Component | Selected | Alternatives considered | Justification |
|---|---|---|---|
| Alphanumeric display | 16×2 character LCD, HD44780 controller with PCF8574 I²C backpack | 0.96″ OLED; 2.4″ SPI TFT | Two I²C lines instead of six parallel pins; readable in direct sunlight, where OLED contrast collapses; the messages are short fixed-width strings, so a graphical display would add cost and code for no information gain |
| Guide/warning strip | WS2815 addressable LED strip, 60 LEDs | Plain LED bar; WS2812B | Per-LED colour under one data line; WS2815 runs at 12 V, so voltage drop along the strip is far smaller than the 5 V WS2812B, and it carries a backup data line so a single failed LED does not blank the remainder |
| Emergency indicator (signal-head board) | Discrete LED plus dedicated PWM-capable output | Reuse of a signal lamp | An emergency indication must be visually distinct from a signal indication; reusing a lamp would make the two states ambiguous to a driver |
| Signal lamps | 5 mm LEDs with 330 Ω series resistors, two heads | Traffic-light module | Two independent heads were needed for the two merging approaches; discrete LEDs allowed the second head to be optional |
| LED library | FastLED | Adafruit NeoPixel | Lower per-frame overhead and richer colour utilities for the flash patterns |
| LCD library | LiquidCrystal_I2C | — | De facto standard for the PCF8574 backpack |

Two electrical constraints determined the topology rather than merely the parts. First, an ATmega328P output pin sources at most about 40 mA, whereas a 60-LED strip at full white draws on the order of amps; the strip therefore has its own supply, with the microcontroller providing data only. Second, the 3.3 V logic of the ESP32 and the 5 V logic of the Uno are not symmetrically compatible: the ESP32's 3.3 V high level exceeds the Uno's ~3.0 V input threshold and needs no shifting, whereas 5 V driven into an ESP32 input does. This asymmetry is why the signal relay in Section 3.6.7 is a deliberately unidirectional single wire.

### 3.3.6 Integration and software stack

| Layer | Selected | Justification |
|---|---|---|
| Vision node HTTP service | Flask | Minimal footprint alongside the capture loop; MJPEG streaming is a few lines |
| Video transport to dashboard | MJPEG over HTTP | Renders in any browser with no plugin or client-side decoder; per-frame independence means a dropped frame costs one frame, not a GOP |
| Telemetry transport | JSON over HTTP, polled at 1 Hz | Human-inspectable with `curl`, trivially testable, and firewall-friendly compared with a custom socket protocol |
| Dashboard | Streamlit (group member's work) | Interface consumed, not selected, by this chapter |
| Serial link | pySerial, 9600 baud | 9600 is the highest rate at which the Uno's `SoftwareSerial` is reliable, and the relay chain contains one |

---

## 3.4 Methodology

### 3.4.1 Development approach

The subsystem was developed iteratively and **measurement-led**. The governing rule adopted was that no design decision would be defended by argument where it could be settled by measurement, and that a measurement contradicting an assumption would be acted upon. This rule produced several of the most significant results in this chapter, including the rejection of the newer detector generation (Section 3.3.1), the rejection of the trained classifier as default (Section 3.3.4), and the discovery of a systematic false-positive fault described in Section 3.6.5.

### 3.4.2 Data

| Dataset | Content | Purpose |
|---|---|---|
| `Test1_day_fotage.mp4` | 1920×1080 daylight road footage | Detection yield, per-stage timing, daylight recognition behaviour |
| `Test2_dark_fotage.mp4` | 1920×1080 night footage | Worst-case detection; beacon detection; the false-positive investigation |
| YouTube emergency compilation, 11 min 43 s | 21,083 frames, unseen camera, mixed conditions | Generalisation test for recognition on footage with no relationship to development data |
| Synthetic constructed clips | Vehicle scripted to travel at exactly 18, 36, 72 and 16.2 km/h | Ground-truth validation of the speed pipeline |

The synthetic clips deserve particular note. Real traffic footage has no ground-truth speed: the true velocity of a car in a video is unknown, so a measurement against it can only be compared with another estimate. By generating footage in which a target's velocity is *constructed* and therefore known exactly, the speed pipeline can be validated in the strict sense rather than merely cross-checked. This is the only point in the subsystem where genuine ground truth exists, and the methodology deliberately exploits it.

### 3.4.3 Verification strategy

Verification is automated. A regression suite of **833 individual checks** across seven test files covers the vision pipeline, hardware command mapping and serial framing, video sourcing and playback, the signal timing state machine, the analytics, and dashboard rendering. Two techniques within it are worth stating as method rather than result:

- **Synthetic clocks.** Signal timing is asserted against an injected clock rather than by sleeping in real time. A test can therefore assert that a yellow interval is *exactly* five seconds, and can advance two minutes of simulated traffic in milliseconds, neither of which is possible with wall-clock timing.
- **Constructed motion.** As above, speed is asserted against scripted rather than observed motion.

### 3.4.4 Position on what is not proven

A methodological commitment was made early that the report would state explicitly what the measurements do *not* establish. Three such statements appear in this chapter: that no mean average precision figure is quoted because no hand-labelled boxes exist for this footage (Section 3.3.1); that the emergency flag rate of 6.6% is a selectivity figure and not an accuracy figure (Section 3.6.8); and that the automatic speed mode's ±25% is the calibration mode's own error bound rather than a measured field error (Section 3.6.8). Each is a place where a more impressive number could have been quoted by quietly changing what was being claimed.

---

## 3.5 Concept Design from Fundamental Engineering Principles

### 3.5.1 Perspective projection and the metric problem

Under the pinhole camera model, a world point **X** = (X, Y, Z) projects to image coordinates

    x = f · X / Z,    y = f · Y / Z    …………………………………………… (3.1)

where *f* is focal length in pixels. Differentiating with respect to time for motion parallel to the image plane gives

    dx/dt = (f / Z) · dX/dt    ……………………………………………………… (3.2)

Equation (3.2) is the formal statement of why pixel velocity is not physical velocity: the observed image velocity is scaled by *f/Z*, and depth *Z* varies continuously across a road scene. Recovering dX/dt requires knowledge of *Z*, or an equivalent constraint.

### 3.5.2 The planar constraint and the homography

The equivalent constraint used here is that road vehicles are supported by a plane. For points on a plane, the projective relationship between world and image coordinates collapses to a **homography**:

    [u  v  w]ᵀ = **H** · [X  Y  1]ᵀ,    x = u/w,  y = v/w    ………………… (3.3)

where **H** is a 3×3 matrix defined up to scale, and therefore has eight degrees of freedom recoverable from four point correspondences. Calibration consists of the operator clicking four points on the road that form a rectangle of known real dimensions. Thereafter any image point known to lie on the road plane maps to metric coordinates, and displacement in those coordinates is a true distance in metres.

This yields the single most important geometric decision in the pipeline. The homography is valid **only for points on the plane**, and a bounding box centroid floats in mid-air at roughly bumper height. The **bottom-centre of the bounding box** — the tyre contact patch — is the one point on a vehicle that lies on the road surface, and it is the point the pipeline tracks. Using the centroid instead introduces an error that grows with vehicle height and varies with range, and would bias trucks and buses systematically relative to cars.

### 3.5.3 The uncalibrated alternative

Where no calibration exists, scale is recovered from known object dimensions. If a vehicle of class *c* has typical real height *H_c* and its bounding box height is *h* pixels, the local scale is

    s = H_c / h   [metres per pixel]    ……………………………………………… (3.4)

Height is used rather than width because height is nearly invariant to the vehicle's heading relative to the camera: a car viewed from the side subtends roughly 4.5 m horizontally but still about 1.5 m vertically. Because each vehicle supplies *s* at its own image position, perspective is compensated implicitly and without calibration.

The mode carries a documented failure case that follows directly from the principle. For a camera pointing near-vertically downward, the box's vertical extent is filled by the vehicle's ~4.5 m *length* rather than its ~1.5 m height, so equation (3.4) under-estimates scale by a factor of roughly three and every reported speed is correspondingly too low. The system therefore provides a two-click fixed-scale calibration for near-vertical views rather than silently producing a plausible wrong answer.

### 3.5.4 The measurement clock

Speed is displacement over time, and the choice of clock is not arbitrary. For a live camera the wall clock is correct: frames arrive as fast as the world produces them. For a recorded video it is wrong. If a clip recorded at 30 fps is processed at 20 fps, wall-clock timing stretches each inter-frame interval by 50% and reports speeds one third too low; process the same clip faster than real time and speeds are inflated. The reported speed becomes a property of the computer rather than of the traffic.

The subsystem therefore gives a file-backed source its own **media clock**, advancing by exactly 1/*fps* per frame regardless of processing rate:

    t_n = n / fps_source    …………………………………………………………… (3.5)

This is a small change with a strong verification consequence, described in Section 3.6.8: the same clip played at four different rates must return the same speed, and does.

Displacement is measured over a fixed **0.8 s window** rather than between consecutive frames. Bounding-box regression jitters by a few pixels frame to frame; differentiating over one frame amplifies that jitter into apparent motion, whereas a window of *N* frames attenuates zero-mean noise by roughly √N while remaining short enough to resolve genuine acceleration.

### 3.5.5 The congestion index

A single bounded scalar is required, because the downstream consumers — a signal state machine, an LED strip, a 16-character display — cannot act on a vector of per-vehicle measurements. Following the speed–density basis of Section 3.2.4, the index combines a speed term and a density term:

    CI = w_s · (1 − v̄/v_f) · p + w_d · (D/D_max)    ………………………… (3.6)

where v̄ is mean measured speed, v_f is free-flow speed (50 km/h), *D* is PCE-weighted density, D_max is the weighted count treated as a full road (20), and w_s = 0.60 and w_d = 0.40. Both terms are clamped to [0, 1], so CI ∈ [0, 1]. PCE weights are car 1.0, motorcycle 0.5, bus 3.0, truck 3.0.

The term *p* is a **presence factor** and encodes a correction that the first implementation lacked:

    p = min(1, N / N_min),   N_min = 3    ……………………………………… (3.7)

Its purpose is to prevent a single slow vehicle on an otherwise empty road from registering as congestion. One slow driver is one slow driver; congestion requires enough vehicles to be collectively slow. A second guard sets CI = 0 whenever N = 0, since an empty road is free-flowing by definition — the first version retained the last observed speed and could report heavy congestion with nothing in frame at all.

The continuous index is mapped to FREE / MODERATE / HEAVY at thresholds 0.30 and 0.60. Because a bare comparator oscillates when the input sits on a threshold, and because the consumers of this level are physical devices, the mapping applies **hysteresis** of ±0.05 together with a minimum dwell of 1.5 s. This is functionally a Schmitt trigger, and its justification is mechanical rather than aesthetic: without it, an index hovering at 0.60 would cause the LED strip and the servo to chatter between states at frame rate.

### 3.5.6 Beacon detection as a temporal signal

The insight underlying beacon detection is that a beacon is defined by its behaviour in time, not by its appearance in one frame. A red pixel region is a tail lamp; a red pixel region that *switches* is a beacon. The detector therefore evaluates four conditions over a 3.4 s observation window:

1. **Spatial gate.** Only the upper portion of the bounding box is examined, extended 18% above the roofline because a bright lamp blooms beyond the metalwork it sits on. This also excludes brake lights, which are red and located low on the body.
2. **Chromatic purity.** For red, G < 0.55·R; for blue, R < 0.70·B; with luminance ≥ 165/255. The luminance threshold is empirically grounded: measured on the night footage, genuine beacon pixels occupy 200–255 while bluish paintwork and reflections occupy 130–155, so 165 separates the populations.
3. **Rate plausibility.** At least three cycles, at 1–6 Hz. Beacon flash rates are legally regulated; below 1 Hz is not a beacon and above 6 Hz is sensor noise.
4. **Switching character.** This is the discriminating test. A lamp is a two-state device — fully on or fully off — and is almost never sampled mid-transition. A red car passing beneath a street lamp brightens and dims *continuously* and spends much of its time at intermediate luminance. The detector therefore computes the fraction of samples in the intermediate band and rejects the candidate above 15%.

Condition 4 is the design element I would single out as the most instructive: it separates the two cases by a physical property of the light source rather than by a tuned threshold on appearance, and Section 3.6.8 shows it separating them completely.

### 3.5.7 Separation of decision from display

The final principle governs the LED and LCD work. The signal phase is computed in exactly one place — the state machine on the PC — and the microcontrollers *render* it. The alternative, in which each board derives the phase from the congestion level it receives, was rejected: it duplicates the timing rules in two languages that can drift apart, and it makes it possible for two displays to disagree about the state of one junction. The seven-field serial protocol of Section 3.6.7 therefore carries the lamp to light as an authoritative field, not the data from which a lamp could be inferred.

---

## 3.6 System Implementation

### 3.6.1 Architecture

The subsystem is implemented as two co-operating processes.

```
   ┌──────────────────────────────────────────────────────────┐
   │  CV NODE  (broadcast_server.py)                          │
   │                                                          │
   │  camera / video / stream URL                             │
   │        │                                                 │
   │        ▼                                                 │
   │  YOLO11-m @1280 FP16  ──►  BoT-SORT  ──►  speed (km/h)   │
   │        │                        │              │         │
   │        ▼                        ▼              ▼         │
   │  CLIP livery            beacon detector   congestion CI  │
   │        └────────┬───────────────┘              │         │
   │                 ▼                              ▼         │
   │          emergency fusion  ────────►  signal state m/c   │
   │                 │                              │         │
   │                 ▼                              ▼         │
   │            CSV/XLSX logging          hardware renderer   │
   └───────┬──────────────────────────────────┬───────────────┘
           │ HTTP :8502                       │ USB serial 9600
           │ /video_feed /telemetry           ▼
           │ /history /command         ┌─────────────┐
           ▼                           │  ESP32 hub  │ WS2815 strip + LCD
   ┌────────────────┐                  └──┬───────┬──┘
   │   DASHBOARD    │           UART1 ────┘       └──── UART2
   │  (Streamlit)   │             │                       │
   └────────────────┘             ▼                       ▼
                          ┌──────────────┐        ┌──────────────┐
                          │ Signal Uno   │        │ Lane-changer │
                          │ 2 heads, LCD │        │ Uno: LEDs,   │
                          │ emerg. strip │        │ LCD, servo   │
                          └──────────────┘        └──────────────┘
```

**Figure 3.1 — Subsystem architecture and data paths**

The split into two processes is not incidental. The node must run where the camera and GPU are; the dashboard needs only a browser. Separating them allows the dashboard to run on a different machine in a different room, which is how the system is intended to be operated and how it was demonstrated.

### 3.6.2 Detection and tracking

Detection runs at 1280-pixel input under FP16, restricted to the four COCO vehicle classes (car, motorcycle, bus, truck) so that a pedestrian or a traffic light cannot be admitted as a vehicle. Class-agnostic non-maximum suppression is enabled, and detections are capped at 100 per frame.

Two quality gates sit between the tracker and everything downstream.

**Track maturity.** A track must survive **3 frames** before it is counted. Single-frame detections are overwhelmingly spurious, and admitting them would inflate counts and inject phantom vehicles into the congestion index.

**Class voting.** A vehicle's class is decided by majority vote over a **30-frame** history, and a challenger must strictly exceed the incumbent to displace it. Without this, a van alternates between "car" and "truck" indefinitely as viewing angle changes, which matters because the PCE weight differs by a factor of three between those classes and would make the density term oscillate.

An optional region of interest, drawn during calibration, restricts counting to the carriageway of interest so that a pavement or an opposing carriageway does not contribute to the measurement. Detections falling outside it are reported separately in telemetry rather than silently discarded, so that an incorrectly drawn region is visible rather than invisible.

### 3.6.3 Speed measurement

For each mature track the bottom-centre point is transformed to metric coordinates by whichever calibration mode is active, accumulated over a 0.8 s window (minimum 0.25 s and 3 samples), and differentiated. Readings above 200 km/h are discarded as tracker identity switches rather than reported, and an exponential moving average (α = 0.35) smooths the per-vehicle result. Vehicles below 3 km/h are classified as stopped, which feeds the moving/stopped split shown on the dashboard.

Calibration is performed once per camera position by `calibrate_speed.py`: the operator clicks four road-plane points, enters the real dimensions, and may press a key to overlay a one-metre grid — if the grid follows the road markings, the homography is sound. The result is persisted to JSON and can be reloaded live without restarting the node.

A calibration belongs to one camera in one position, and the implementation treats this as a correctness issue rather than a caveat: switching the node to a different source does not carry the calibration across, because a homography from another viewpoint is geometrically meaningless for the new view and would produce confidently wrong metric readings.

### 3.6.4 Congestion and signal phase

The congestion index of equation (3.6) is computed once per frame and mapped to a level through the hysteresis and dwell logic of Section 3.5.5. The signal state machine consumes the level and applies two rules that make its behaviour that of a real signal rather than a switch:

- **Committed transitions.** Once yellow begins it runs for the full five seconds and always terminates at the phase it was moving toward. A real signal never abandons an amber part-way. If traffic conditions reverse during the interval, a fresh transition is started afterwards.
- **Minimum dwell.** Green and red are each held for at least 3 s, which prevents a level change arriving during a yellow from producing a green lasting a single frame.

The controller supports two mappings, selected at launch. A junction approach shows green when there is traffic to serve; a **merging side road is metered in the opposite sense** — held red while the carriageway it feeds is HEAVY, and released when there is room — because the signal is on the ramp rather than on the carriageway. Only the set of levels calling for green differs between the two; the timing rules are identical, which is why the same state machine serves both.

### 3.6.5 Emergency recognition and a corrected fault

Vehicle crops are scored by CLIP against 38 natural-language prompts covering emergency and ordinary vehicle descriptions. Four safeguards surround the call:

- **Readability gating.** Crops with mean brightness below 28 or contrast below 12 are not submitted at all. A vision–language model will return a confident answer for an image containing no information, so it is not shown one.
- **Repeat confirmation.** At least three agreeing observations are required before the flag is raised.
- **Asymmetric thresholds.** 0.70 to set the flag, 0.50 to clear it — hysteresis again, for the same reason.
- **Adaptive scheduling.** Vehicles are re-checked every 5 frames, backing off progressively to every 60 for tracks that consistently score as ordinary, which concentrates compute on ambiguous cases.

**A systematic fault found by measurement.** The first implementation summed the scores of all 14 emergency prompts and compared the total against the 15 ordinary prompts. Because CLIP's softmax distributes mass across all prompts, an uninformative crop produces a near-uniform distribution, and 14 of 29 prompts sum to approximately 0.48 — against a decision threshold of 0.45. Every unreadable crop was therefore classified as an emergency vehicle. On the night footage this flagged **88 of 299 vehicles (29%)** as emergency traffic.

The correction was to restructure the comparison as a one-against-one contest — best emergency prompt against best ordinary prompt — rather than a group aggregate. The same uninformative crop then scores 0.25, far below threshold. This is the clearest illustration in the project of a fault that is invisible to inspection and obvious to measurement: the code contained no error in the ordinary sense, and the flaw lay entirely in the statistical structure of the comparison.

The beacon detector implements Section 3.5.6 and, when it declines a candidate, reports **which** condition failed — `too-dim`, `steady-light`, `shallow-modulation`, `gradual-not-switched`, `not-repeating`, `implausible-rate`, `red-below-roofline`, `window-too-short`. Diagnosability was treated as a functional requirement: a detector that reports only "no" cannot be tuned, whereas one that names the failing gate can be.

Fusion of the two recognisers follows the distinction of Section 3.2.5. A confirmed flashing beacon yields **"AMBULANCE — PRIORITY"** and justifies intervention; livery recognition alone yields **"AMBULANCE"**, which is displayed but does not intervene. Where the beacon is blue, the vehicle is police and no further evidence is needed; where it is red, CLIP disambiguates ambulance from fire appliance, since only the livery separates those two.

### 3.6.6 Integration with the dashboard

The node exposes an HTTP interface on port 8502. The dashboard is a client of it and holds no camera, no model and no serial port.

**Table 3.4 — Principal endpoints consumed by the dashboard**

| Endpoint | Method | Purpose |
|---|---|---|
| `/video_feed` | GET | Annotated MJPEG stream |
| `/telemetry` | GET | Latest snapshot: per-vehicle detections, speeds, congestion index and level, signal phase, hardware state, FPS |
| `/history` | GET | Rolling ~1 hour of 1 Hz telemetry, for the charts |
| `/sessions`, `/records` | GET | Session summaries and archived rows across all runs |
| `/export`, `/export/all` | GET | Workbook and CSV downloads for the current session and the full archive |
| `/light` | GET/POST | Read the phase; force one, or return to automatic |
| `/source`, `/sources`, `/source/upload`, `/source/control` | GET/POST | Change and control what is being watched; upload a video; pause, seek, loop, set playback rate |
| `/command` | POST | Manual hardware control, forwarded to the boards |
| `/health` | GET | Liveness, FPS, calibration mode, phase, recording state, source |

Five implementation decisions in this layer are worth setting out, because each addresses a failure that occurred before it was made.

**Command forwarding rather than a second serial port.** On Windows, two processes cannot open the same COM port; the attempt fails outright. The dashboard therefore POSTs hardware commands to `/command` and the node relays them down the link it already owns. This is what allows the dashboard to run on a different machine entirely, and it means the serial port has exactly one owner at all times.

**Delivery confirmation.** Every command response carries a `delivered` flag reporting whether the bytes actually reached the wire, obtained for ESP32 commands by waiting for the board's JSON acknowledgement. An unplugged board is therefore visible on the dashboard as a failed command rather than as a button that appears to work.

**Staleness detection.** A reachable `/telemetry` endpoint is *not* proof of a running pipeline: the node continues serving its last snapshot after a video ends or a camera drops. The dashboard tracks the telemetry timestamp and treats data older than **6 seconds** as stale, because without this the interface displays the final frame's numbers indefinitely and looks entirely healthy while measuring nothing.

**Source state read live.** The current source and playback position are read at request time rather than attached to the most recent frame. A paused video produces no frames, so a paused flag carried on a frame could never be delivered — the state would freeze at "playing" precisely when it stopped being true.

**Graceful degradation.** When the node is unreachable the dashboard falls back to a clearly labelled simulated feed rather than an error page, which allows the interface to be developed, demonstrated and tested independently of the GPU machine. The fallback is always identified as simulated, so a demonstration cannot accidentally present synthetic data as measured.

The commanded hardware state is echoed back through `/telemetry`, so the dashboard displays the position hardware was actually moved to rather than the last position a user requested. Notably, the lane divider position is deliberately *not* derived from the servo angle: that servo also gates the signal phase and moves on every light change, whereas the divider moves only when commanded.

### 3.6.7 LED and LCD implementation

**Topology.** A single USB cable connects the PC to the ESP32, which acts as the hub. Two Arduino Unos hang off its hardware UARTs: the signal-head controller on UART1 (GPIO33 → Uno D7 plus a shared ground) and the lane-changer/LED board on UART2 (GPIO25/26). The link to the signal head is a single unidirectional wire for the electrical reason given in Section 3.3.5, and the Uno reads it on a `SoftwareSerial` port rather than on pins 0/1 so that its USB connection remains available for uploads and monitoring while the relay wire is attached.

**Protocol.** The node emits one line per second in a seven-field format:

```
PHASE,LEVEL,CI,VEHICLES,EMG,LINE1,LINE2

GREEN,HEAVY,0.740,11,0,LIGHT: GREEN,HEAVY CI:0.74
YELLOW,FREE,0.120,0,0,YELLOW 3s>RED,FREE CI:0.12
RED,FREE,0.050,1,1,LIGHT: RED,!! AMBULANCE !!
```

`PHASE` is authoritative — the lamp to light, not data from which a lamp might be inferred (Section 3.5.7). `EMG` drives the emergency strip. `LINE1` and `LINE2` are the two 16-character LCD rows, rendered and padded on the PC.

Two properties of this format do useful work. The ESP32 distinguishes telemetry from commands **by shape**: a comma-separated line is signal telemetry and is relayed downstream untouched, whereas a JSON object or a bare word is a command addressed to the hub itself. And the signal-head sketch discards any line without exactly seven comma-separated fields, so the bare-word LED commands intended for the lane-changer board pass it by harmlessly. One serial link therefore carries two protocols to three boards with no addressing scheme.

**LCD rendering.** Line 1 always carries the signal, counting down during yellow (`YELLOW 3s>RED`) so an observer sees the transition progressing rather than a static caption. Line 2 carries the congestion level and index, or, when an emergency vehicle is present, the vehicle's identified type — `!! AMBULANCE !!`, `!! POLICE !!`, `!! FIRE TRUCK !!` — since naming the vehicle is more useful to a driver than a generic warning. Both rows are padded to exactly 16 characters on the PC so that a shorter message cannot leave fragments of the previous one on the display.

**LED behaviour.** On the signal-head board, exactly one lamp is lit per head at any moment, and an unrecognised phase leaves all lamps dark — visibly wrong rather than quietly wrong. Two heads are supported for the two merging approaches, and the second is optional: leaving its three pins unwired changes nothing. The emergency indication is a *dedicated* output on a PWM-capable pin, held steady rather than flashed, because a steady red reads unambiguously as "emergency vehicle present" and holding a level costs the main loop nothing.

On the lane-changer board, a 60-LED WS2815 strip driven through FastLED provides the driver-facing guide, with a labelled mode per situation:

| Mode | Strip behaviour | LCD text |
|---|---|---|
| `NORMAL` | Steady green | `NORMAL TRAFFIC / 4 LANES OPEN` |
| `AMBULANCE` | Double blue flash, 10 s, then steady blue | `AMBULANCE / PRIORITY ACTIVE` → `LANE READY` |
| `POLICE` | Double red flash, 10 s, then steady red | `POLICE / PRIORITY ACTIVE` → `LANE READY` |
| `ACCIDENT` | Slow red flash | `ACCIDENT / ROAD BLOCKED` → `DRIVE SLOW` |
| `SPEED` | Yellow flash, then steady yellow | `SPEED WARNING / REDUCE SPEED` |

The two-phase structure of each mode — an attention-getting flash followed by a steady state — is deliberate: the flash acquires attention, the steady colour communicates the persisting condition without continuing to demand it.

**Automatic strip control.** Initially the strip only changed when a dashboard button was pressed, so a *detected* ambulance lit the signal-head indicator while the hub's strip stayed dark. This was closed by having the same detection that drives the signal also drive the strip, sending `ACTIVATE_EMERGENCY` / `DEACTIVATE_EMERGENCY` on transition only. Because the node updates its own record of hardware state when it does so, the dashboard then shows the board's real state rather than the last thing a human clicked. Only the emergency condition is pushed automatically; congestion data is not, because the hub's LCD reports what its strip is doing and nothing else, while level, index and countdown belong on the signal head's LCD, which already receives them.

**Diagnostics as a feature.** A significant portion of the display work exists to make failure legible, prompted by a debugging session in which a silent misconfiguration was indistinguishable from dead hardware. The signal-head LCD now distinguishes four states:

| Condition | Display | Meaning |
|---|---|---|
| No bytes received after 8 s | `No data from PC / Check D7 + GND` | Nothing is arriving on either route — usually the relay wire or its missing shared ground |
| Bytes arriving but never parsing | `Serial garbled / Need baud 9600` | The link is live but the baud rate is wrong |
| Packets stopped for 5 s | `NO SIGNAL / Check PC cable` | Contact was established and then lost |
| Running | Phase, level, index, countdown | Normal operation |

The distinction between the first two required a specific implementation choice: the "last packet" timestamp is updated only when a line *parses into seven fields*, not when a newline arrives. A wrong baud rate delivers a stream of noise that occasionally contains a newline byte, and treating that as contact would report a healthy link that has never carried one usable packet — while suppressing the very hint that names the fault.

Two further safeguards complete the board behaviour. A **startup self-test** cycles every output in the documented wiring order, so a miswired LED is identified by which pin lights out of turn. And on loss of signal the board **fails safe to red on both heads** rather than holding its last phase: a green left standing would invite traffic to merge onto a carriageway that is no longer being measured.

### 3.6.8 Results and testing

**Speed accuracy against constructed ground truth.**

**Table 3.5 — Speed validation (`test_traffic_system.py`)**

| Test | Expected | Measured |
|---|---|---:|
| Homography mode | 18 km/h | **18.00** |
| Homography mode | 36 km/h | **36.00** |
| Homography mode | 72 km/h | **72.00** |
| Automatic (height-based) mode | 18 km/h | **18.00** |
| Fixed-scale mode | 18 km/h | **18.00** |
| Parked vehicle | 0 km/h | **0.12** |

**Table 3.6 — Media-clock independence (`test_video_source.py`)**

A vehicle scripted at 16.2 km/h through a real MP4, decoded at four different rates:

| Playback rate | Measured speed |
|---|---:|
| Real time (1×) | 16.2 km/h |
| Double speed (2×) | 16.2 km/h |
| Half speed (0.5×) | 16.2 km/h |
| Unthrottled | 16.2 km/h |

All four agree to 0.00, confirming that reported speed is a property of the traffic and not of the machine. Field accuracy is **±5% calibrated** and **±25% automatic**; these are the calibration modes' own error bounds and are stated as such, not as a measured field error, which would require instrumented reference vehicles the project did not have.

**Throughput.** Per-stage cost was measured on both clips at the default configuration.

**Table 3.7 — Per-stage cost, yolo11m @ 1280, FP16**

| Stage | Night (23.0 veh/frame) | Day (25.5 veh/frame) |
|---|---|---|
| Detection + tracking | 29.6 ms (66.2%) | 28.8 ms (82.1%) |
| Livery recognition (CLIP) | 5.4 ms (12.0%) | 1.1 ms (3.2%) |
| Beacon detection | 9.8 ms (21.8%) | 5.2 ms (14.7%) |
| Everything else | ~0.0 ms | ~0.0 ms |
| **End to end** | **44.8 ms → 22.3 FPS** | **35.1 ms → 28.5 FPS** |

Detection dominates at two-thirds to four-fifths of the budget, which identifies it as the only stage where optimisation would return anything measurable. Both recognisers cost more at night, for different reasons: CLIP because dark crops fail the readability gate and are retried rather than backed off, and the beacon detector because at night there are genuinely beacons to measure. Live end-to-end throughput during demonstration, taken from the node's own telemetry, was **27.1 FPS**.

A re-measurement on 11 August returned 31.7 ms against the 27.3 ms of Table 3.1 for the same configuration — roughly 16% slower — while the yield was **identical at 23.0 vehicles per frame**. The unchanged yield indicates the model and settings were unchanged and only the clock moved; the earlier run had a quieter machine, whereas the later one shared the GPU with several background applications. Table 3.1 remains valid for its purpose, which is comparison *between* settings, since every row would shift together.

**Emergency recognition on unseen footage.** A full run over the 11 min 43 s compilation, with no region of interest and nothing cherry-picked:

**Table 3.8 — Recognition on unseen internet footage**

| Measure | Value |
|---|---:|
| Frames analysed | 21,083 |
| Scene cuts detected | 301 |
| Vehicles tracked | 2,511 |
| Recognised as emergency | 166 (6.6%) |
| Distinct vehicles after merging track-ID churn | ~96 |
| Beacon-confirmed (priority) | 21 |
| By type | 60 fire, 59 ambulance, 47 police |
| Confidence range | mostly 0.93–1.00 |
| Beacon rates measured | 1.2–4.8 Hz |

**The 6.6% is a selectivity figure, and it is stated here as one.** It is *not* an accuracy figure and *not* a false-positive rate: confirming either would require labelling all 166 crops by hand. A sample of the saved crops was inspected and most were unmistakable, but a small number were ambiguous crops from dense traffic where a neighbouring emergency vehicle bled into the box. What the number does establish is that the recogniser is discriminating rather than indiscriminate — a detector flagging everything would achieve perfect recall and demonstrate nothing. The measured beacon rates of 1.2–4.8 Hz falling within the legally regulated band is a useful independent corroboration.

That only 21 of 166 reached beacon-confirmed priority is expected rather than a defect. In daylight a beacon bright enough to register saturates the sensor and blooms to white, and white correctly fails the red/blue purity gates. Measured on a daylight ambulance: beacon channel score **0.04**, livery score **0.95**. The two recognisers therefore cover different conditions by design — CLIP carries daylight, the beacon detector carries night.

**Beacon discrimination.** On the night footage the switch-versus-slide test of Section 3.5.6 separated the populations completely: the genuine beacon scored **0.00** on the intermediate-luminance fraction, while every false positive scored between **0.18 and 0.43**. Combined with making the beacon rather than the livery the priority trigger, false alarms on that footage fell from **88 of 299 vehicles (29%) to 0**.

**Signal timing.** Asserted against a synthetic clock: the yellow interval is exactly 5 s on every transition; with the congestion level flipped every 0.5 s for two minutes, no phase is ever momentary and the R→Y→G→Y→R ordering never breaks; minimum dwell is honoured on both green and red.

**Regression suite.**

**Table 3.9 — Automated test coverage**

| Suite | Checks | Covers |
|---|---:|---|
| `test_traffic_system.py` | 217 | Vision pipeline: speed, congestion, ROI, emergency fusion, logging |
| `test_server.py` | 165 | Hardware command mapping, serial format, ESP32 relay, HTTP endpoints |
| `test_video_source.py` | 113 | Uploads, stream URLs, playback rates, media clock |
| `test_dashboard_interactive.py` | 109 | Filters, exports, controls |
| `test_analytics.py` | 90 | Analytics datasets and charts |
| `test_dashboard.py` | 83 | All 9 pages render, offline and live |
| `test_traffic_light.py` | 56 | Signal timing against a synthetic clock |
| **Total** | **833** | |

**831 of 833 pass in a single run, and 833 is not reachable in one run.** This is reported precisely rather than rounded up. The first six suites pass 724/724 unconditionally. The interactive suite cannot be fully satisfied at once because it requires the CV node in both states: with the node down it scores 107/109, failing two checks that POST to the node; with the node up it scores 105/109, failing four checks written against the simulated fallback feed. Resolving this properly means splitting that suite along the live/offline boundary rather than asking one run to occupy both states — a known and documented limitation rather than an intermittent failure.

### 3.6.9 Sources of error and limitations

| Source | Effect | Mitigation applied |
|---|---|---|
| Camera motion | All measurement assumes a static camera; motion makes stationary vehicles appear to move | Documented as a deployment requirement; BoT-SORT's camera-motion compensation provides partial tolerance only |
| Homography validity | Valid only for one camera in one position | Calibration is not carried across a source change; per-camera calibration files supported |
| Near-vertical camera views | Automatic mode under-scales by ~3× | Detected as a documented failure case; two-click fixed-scale mode provided |
| Track identity switches | Spurious large displacements | 200 km/h plausibility rejection; BoT-SORT selected for association robustness |
| Bounding-box jitter | Apparent motion when stationary | 0.8 s measurement window; EMA smoothing; parked vehicle measures 0.12 km/h against 0 |
| Track-ID churn | One vehicle counted more than once | Reported openly: 166 flags correspond to ~96 distinct vehicles |
| Daylight beacon saturation | Beacons bloom to white and fail colour gates | Compensated by CLIP, which carries daylight; behaviour quantified (0.04 vs 0.95) |
| No labelled ground truth for detection | mAP cannot be computed | No accuracy figure is quoted; the benchmark tool refuses to print one |
| Machine load variability | ~16% throughput variation between runs | Yield used as the invariant; live telemetry quoted for demonstration figures |

---

## 3.7 Summary

This chapter has documented the computer vision subsystem, its integration with the group's dashboard, and the LED and LCD outputs that present the system's decisions at the roadside.

The vision pipeline detects and tracks vehicles with YOLO11-m at 1280-pixel input under FP16 precision and BoT-SORT association, a configuration selected by benchmarking eleven alternatives on the project's own footage — a process that overturned two initial assumptions, since the newer detector generation proved both slower and less productive, and additional model capacity at fixed resolution returned nothing. Image motion is converted to km/h through either a ground-plane homography or a known-height scale, anchored at the tyre contact patch because it is the only point on a vehicle that lies on the road plane, and timed against a media clock so that reported speed is a property of the traffic rather than of the computer. Validated against constructed ground truth, the pipeline returns 18.00, 36.00 and 72.00 km/h for targets scripted at 18, 36 and 72, and returns an identical 16.2 km/h across four playback rates. Per-vehicle measurements are condensed into a bounded congestion index combining speed and PCE-weighted density, guarded so that an empty road cannot register as congested and a single slow driver cannot register as a jam, and stabilised by hysteresis so that it can drive physical actuators without chatter.

Emergency vehicles are recognised by two complementary means — zero-shot livery recognition, which establishes what a vehicle is, and beacon detection, which establishes what it is doing — with only the latter justifying intervention. Correcting the statistical structure of the livery comparison, and making the beacon rather than the livery the trigger, reduced false alarms on the night footage from 29% of all traffic to zero.

Integration with the dashboard is over HTTP, with the vision node as the single owner of both the camera and the serial link. This allows the two to run on separate machines, gives hardware commands genuine delivery confirmation, and — through staleness detection and an explicitly labelled simulated fallback — ensures the interface cannot present stale or synthetic data as live measurement.

The display subsystem renders, rather than infers, the controller's state: the phase is computed in one place and transmitted authoritatively, so no two displays can disagree about the junction. The LCDs carry the signal, the congestion reading and the identified type of any emergency vehicle; the LED strip provides a labelled driver-facing guide whose flash-then-hold structure acquires attention and then communicates a persisting condition. A substantial part of this work exists to make failure legible — distinguishing an absent link from a misconfigured one, failing safe to red on signal loss, and self-testing every output at boot.

The subsystem is covered by 833 automated checks, of which 831 pass in a single run for a documented structural reason. Where a measurement does not support a claim, the claim is not made: no mAP is quoted because no labelled boxes exist, the 6.6% emergency flag rate is presented as selectivity rather than accuracy, and the automatic speed mode's ±25% is identified as the mode's error bound rather than a measured field error. Objectives 1 to 5 of Section 3.1.1 are met, with the limitations of each stated in Section 3.6.9.

---

## References

Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust associations multi-pedestrian tracking*. arXiv:2206.14651.

Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking. *2016 IEEE International Conference on Image Processing (ICIP)*, 3464–3468.

Greenshields, B. D. (1935). A study of traffic capacity. *Proceedings of the Highway Research Board*, 14, 448–477.

Hartley, R., & Zisserman, A. (2003). *Multiple view geometry in computer vision* (2nd ed.). Cambridge University Press.

Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11* (Version 11.0.0) [Computer software].

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.

Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., & Berg, A. C. (2016). SSD: Single shot multibox detector. *European Conference on Computer Vision (ECCV)*, 21–37.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 8748–8763.

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779–788.

Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 28, 91–99.

Transportation Research Board. (2016). *Highway capacity manual* (6th ed.). National Academies of Sciences, Engineering, and Medicine.

Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep association metric. *2017 IEEE International Conference on Image Processing (ICIP)*, 3645–3649.

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. *European Conference on Computer Vision (ECCV)*, 1–21.
