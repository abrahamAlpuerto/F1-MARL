# The visualizer feed

Everything the renderer needs to draw a race, and nothing it does not.

This is the contract between the engine and whatever draws the cars. If you are
building the visualizer, this file plus
[`docs/visualizer-example/`](visualizer-example/) is all you need — you never
have to build or run the simulator.

---

## Get some data

```bash
python examples/race.py --laps 3 --name demo
```

Writes `episodes/demo/`:

```
track.json     the circuit. Static per track — cache it across episodes.
episode.json   the header: how many frames, who is racing, what happened.
frames.f32     the positions. Binary, little-endian float32.
```

Add `--live` and it also writes `stream.jsonl` alongside, which is the same race
as a line-per-frame feed you can tail while it runs.

No toolchain? Ask for the three files and drop them anywhere. Nothing in the
example imports the engine.

---

## The one rule

**Index fields by name. Never by a hardcoded number.**

`episode.json` ships a `fields` array and a `stride`:

```json
"stride": 27,
"fields": ["x","y","z","heading","speed","steer","throttle","downforce",
           "drag","wake","lateral_offset","position","lap","lap_fraction",
           "gap_ahead","flags","tyre_temp_front","tyre_temp_rear",
           "tyre_wear_front","tyre_wear_rear","tyre_grip","fuel_kg",
           "gear","rpm","ers_charge","lateral_g","longitudinal_g"]
```

Fields get **appended** over time. A reader that looks up `fields.indexOf("speed")`
keeps working; one that hardcodes `4` starts rendering the wrong channel the
first time something is added ahead of it, and it will not throw — it will just
be quietly wrong. Existing fields are never reordered or removed, so name lookup
is always safe.

**This has already happened once.** The stride went from 16 to 27 when the tyre,
fuel and powertrain physics arrived. Everything that existed before is still
there, under the same name, at the same index — but any reader that had `16`
written into it broke, and one that asked the header did not. Use `has("rpm")`
(or the Python `"rpm" in ep.fields`) to degrade gracefully rather than assuming.

Same for `n_cars`. It is 20 by default — ten teams of two — and it is explicitly allowed to change.

---

## `frames.f32`

A flat little-endian `float32` array, laid out **frame → car → field**. Car `c`
at frame `f` starts at:

```
(f * n_cars + c) * stride
```

That is the whole format.

Size is worth planning for: a 3-lap, 20-car race at 60 Hz is **56 MB**, and it
scales linearly in all three of laps, cars and frame rate. A full 57-lap Grand
Prix at 60 Hz would be about a gigabyte. If that is more than the renderer wants
to hold, drop the sample rate — `--fps 20` is a third of the size and still
smooth for a map view — or ask for fewer laps.

### The fields

| name | units | notes |
|---|---|---|
| `x`, `y` | metres | world coordinates. **Y grows upward** |
| `z` | metres | elevation, from the source telemetry |
| `heading` | radians | **not wrapped** — it accumulates. Fine for `sin`/`cos`; wrap it yourself before comparing or interpolating two headings |
| `speed` | m/s | multiply by 3.6 for kph |
| `steer` | −1…1 | fraction of full lock (±26°) |
| `throttle` | −1…1 | negative is braking |
| `downforce` | 0…1 | 1.0 is clean air. Below that the car has less grip because it is following someone |
| `drag` | 0…1 | 1.0 is clean air. Below that it is getting a tow |
| `wake` | 0…1 | how deep in another car's wake it is. The single number to shade a car by |
| `lateral_offset` | metres | left of the reference line. Redundant with x/y but handy |
| `position` | 1…n | race position. Exactly one car holds each position in every frame |
| `lap` | integer | laps completed |
| `lap_fraction` | 0…1 | how far round the current lap |
| `gap_ahead` | seconds | to the car classified in front. 0 for the leader |
| `flags` | bit field | see below |
| `tyre_temp_front` | °C | core temperature. Optimum is around 95 |
| `tyre_temp_rear` | °C | the rear runs hotter — it is the driven axle |
| `tyre_wear_front` | 0…1 | 0 fresh, 1 worn out |
| `tyre_wear_rear` | 0…1 | |
| `tyre_grip` | ~0.7…1.0 | combined multiplier from temperature and wear |
| `fuel_kg` | kg | starts near 100 and burns off |
| `gear` | 1…8 | 1-based, so it reads like a dashboard |
| `rpm` | rev/min | 4000 idle to 15000 redline |
| `ers_charge` | 0…1 | fraction of the hybrid store remaining |
| `lateral_g` | g | signed; the g-force trace |
| `longitudinal_g` | g | signed; negative under braking |

`flags` bits: `1` off track, `2` in contact this frame, `4` finished,
`8` retired, `16` DRS open, `32` wheelspin, `64` a locked wheel,
`128` deploying hybrid power.

---

## `episode.json`

```json
{
  "format": "racing-feed",
  "version": 3,
  "n_frames": 14948,
  "n_cars": 20,
  "frame_rate": 60.0,
  "stride": 27,
  "fields": ["x", "y", ...],
  "track": "bahrain",
  "lap_length_m": 5357.55,
  "laps": 3,
  "race_distance_m": 16072.7,
  "teams": [{"index": 0, "name": "Vermilion", "color": "#e2412c"}, ...],
  "cars":  [{"index": 0, "name": "V1", "team": 0}, ...],
  "conditions": {"air_temperature_c": 22.0, "track_temperature_c": 28.0,
                 "air_density": 1.1918, "wind_speed": 0.0,
                 "tyre_compound": "medium", "fuel_start_kg": 100.0},
  "drs_zones": [{"detection_s": 848.0, "start_s": 968.0, "end_s": 1423.0}, ...],
  "events": [...],
  "result": {"finish_order": [...], "team_scores": [...], "classification": [...]}
}
```

Team colours are supplied so twenty cars can be told apart on a screen. Override
them from Python with `Feed.set_teams` if the renderer would rather choose.

### Events

Every event carries `frame`, `t` (seconds), `type`, `car`, and `lap`. The list
is **sorted by frame**, so it can be walked alongside playback rather than
searched.

| type | extra | meaning |
|---|---|---|
| `overtake` | `other` | `car` took a place from `other` |
| `contact` | `other`, `value` | `car` (the one behind) touched `other`. `value` is severity, 0–1 |
| `off_track` | `value` | `car` left the circuit; `value` is metres beyond the edge |
| `rejoin` | `value` | back on. `value` of 1 means it was recovered after being stranded |
| `lap` | `value` | `car` completed lap `lap` in `value` seconds |
| `finish` | `value` | `car` took the flag in position `value` |
| `retire` | — | `car` is out |

**Ignore unknown types, do not throw.** More will be added.

### Conditions and DRS zones

`conditions` says what the race was run in — a viewer comparing two races wants
to know whether the second was quicker because the driving was better or because
the air was colder. `drs_zones` gives the arc lengths where the wing may be
opened; drawing them on the track map makes it obvious why a pass happened where
it did. Both are worked out per race, and the zones are derived from the
circuit's own geometry rather than hand-entered, so they exist for any track.

---

## `track.json`

```json
{
  "name": "bahrain",
  "lap_length_m": 5357.55,
  "n": 2048,
  "line": [[x, y, z], ...],
  "half_width_left":  [7.5, ...],
  "half_width_right": [7.5, ...],
  "start_finish_index": 0,
  "sector_indices": [0, 512, 1103],
  "closed": true
}
```

`line` is the reference line — the actual racing line from a real 2024 Bahrain
pole lap, not a centreline — sampled every ~2.6 m. `closed` means the last point
joins the first.

**There are no normals in the file, on purpose.** Derive the edges yourself with
a **central** difference around the loop:

```js
const a = line[(i - 1 + n) % n], b = line[(i + 1) % n];
const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
const nx = -dy / len, ny = dx / len;         // left normal
```

A *forward* difference leaves a visible kink at the start/finish point, because
the circuit is a closed loop. `trackEdges()` in `episode.js` does this.

---

## The live stream

With `--live`, `stream.jsonl` is one self-contained JSON object per line:

```jsonc
{"type":"header", ...}                        // same shape as episode.json
{"type":"frame","f":0,"t":0.0,"cars":[[...],[...]]}
{"type":"event","f":42,"event":{...}}
{"type":"result","finish_order":[...], ...}
```

Each `cars` entry is one car's `stride` floats, in the same field order. The
header is written and flushed **first**, so a reader that attaches late still
knows the layout.

It is far larger and slower to parse than the binary format — a 4-minute race is
~30 MB — so use it to watch a race arrive, not to store one.

---

## Frame rate

Positions are sampled at `frame_rate` hertz of **simulated** time, 60 by
default:

```bash
python examples/race.py --fps 30      # smaller files
python examples/race.py --fps 100     # every physics step, no decimation
```

The physics runs at 100 Hz, so anything up to 100 involves no interpolation on
the engine's side. Above that you get 100 anyway.

Two details that cost time if you find them yourself:

- The policy only acts at 25 Hz. The feed does **not** sample at the policy
  rate — it hangs off the physics, so 60 Hz is 60 genuinely distinct positions,
  not 25 positions with padding.
- 100/60 is not an integer, so frames are sampled against elapsed simulated time
  rather than by dropping every Nth step. Frame intervals therefore vary by one
  physics tick; the long-run rate is exact. Do not assume a fixed step between
  consecutive frames — use `frame / frame_rate` for the timestamp.

---

## Things that will cost you an afternoon

**Flip the Y axis.** World Y grows upward, canvas Y grows downward. On a roughly
symmetric circuit a mirrored render is easy to miss for a surprisingly long time.

**Validate the frame count against the header.** A truncated download is the
common failure and it must not render as a blank screen. Both reference loaders
throw with the expected and actual counts. To see it:

```bash
python - <<'PY'
raw = open('episodes/demo/frames.f32','rb').read()
open('episodes/broken.f32','wb').write(raw[:len(raw)//3])
PY
```

**`heading` is not wrapped.** It reaches −5 rad and beyond over a lap.

**Cars can be off the circuit.** Check `flags & 1` before assuming a car is on
the road; `lateral_offset` can exceed the half width, and a stranded car gets
put back on after a few seconds (you will see a `rejoin` event with `value` 1).

**Position is dense and unique.** Exactly one car holds each of 1…n in every
frame, including retired ones, which are classified last. If your leaderboard
ever shows two cars in P3, that is an engine bug — report it rather than
working around it, because a workaround silently forks the format.

---

## Reference implementations

- [`docs/visualizer-example/episode.js`](visualizer-example/episode.js) — the
  loader. Plain ES module with JSDoc types; runs in a browser with no build step
  and type-checks under TypeScript with `allowJs`/`checkJs`. **This is the file
  to copy.**
- [`docs/visualizer-example/index.html`](visualizer-example/index.html) — a
  runnable viewer: circuit, playback, scrubbing, wake shading, live
  classification, event ticker.
- [`python/racing/feed.py`](../python/racing/feed.py) — the same thing in
  Python, for analysis. Depends on numpy and nothing else.

Nothing in any of them imports the engine. The three files are the seam.
