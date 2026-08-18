# Racing

Teams of AI cars racing each other, and a feed so you can watch it happen.

Twenty cars, ten teams of two, on real Bahrain geometry, driven by a
multi-agent policy that learns racecraft by racing copies of itself. The engine
streams every car's position at 60 Hz into a format a visualizer can read
directly.

```bash
python examples/race.py --laps 3 --name demo
```

```
      car    team       laps      time  best lap
  1.  V1     Vermilion     3   371.000   123.560
  2.  J1     Jade          3   388.560   122.560
  3.  I1     Indigo        3   394.120   123.280
  4.  C1     Cobalt        3   396.320   130.440
  ...
teams: Vermilion 26  Cobalt 22  Jade 18  Amber 28  Indigo 26 ...

142 overtakes, 109 notable contacts, 25836 frames over 430.6 s
```

Slower than the 92.8 s the car is capable of, and for reasons the engine
models: a hundred kilos of fuel, tyres that are not yet at temperature, and a
scripted driver running at 86% of the limit. Watch the lap times come down as
the fuel burns off.

---

## What makes the racing interesting

Three things, and they are the whole design.

**The wake.** A car sitting behind another one is in its dirty air. That does
two opposite things: it removes drag, so the follower is quicker in a straight
line — a tow — and it removes downforce, so the follower has less grip in
corners. You gain on the straight and pay for it in the next corner. Get that
balance wrong in one direction and nobody can ever pass; wrong in the other and
position means nothing. It is the single most consequential block of config in
the repository, and it is [documented as
such](include/racing/config.hpp).

**Teams.** Ten of them, two cars each, and one dial — `team_weight` — that
decides whose result a car cares about:

| `team_weight` | what the field does |
|---|---|
| 0.0 | every car for itself; team mates race each other as hard as rivals |
| **0.5** | a car will give a place to a faster team mate, hold a rival up while its team mate escapes, or run nose-to-tail with a team mate so both get a tow |
| 1.0 | fully cooperative, and oddly selfless |

A car's reward is `(1 - team_weight) * own + team_weight * (team mean)`. That is
the entire cooperative mechanism. Everything a team appears to *decide* is a
consequence of it.

**Racing for position, not lap time.** Progress reward alone produces a field of cars
doing perfect, lonely time trials past each other. A car is paid for the
*position* it gains and loses, so it cares that someone else is there.

## The car

Everything that decides how a lap goes is in the loop, every step:

| | |
|---|---|
| chassis | dynamic bicycle model, RK4 at 100 Hz, kinematic blend at low speed |
| tyres | load-sensitive grip, **four corner loads** with lateral transfer, friction ellipse with a falling tail past the peak — so over-driving gives you *less* |
| tyre state | temperature per axle from frictional work, wear, three compounds |
| aero | downforce and drag, air density from temperature/pressure/humidity, wind, the wake of the car ahead, DRS |
| powertrain | torque curve, eight-speed box with shift cuts, hybrid that harvests under braking and deploys on throttle within a per-lap allowance |
| world | fuel burning off as mass, road gradient, run-off grip, car-to-car contact |
| damage | accumulated from contact and from the barrier at the edge of the run-off; costs downforce, and past a threshold ends the car's race |

Fitted to a real 2024 Bahrain pole lap:

| | model | real | error |
|---|---|---|---|
| lap time | 92.759 s | 92.608 s | +0.16% |
| top speed | 298.5 kph | 301.0 kph | −0.8% |
| minimum speed | 67.6 kph | 66.6 kph | +1.5% |
| aero efficiency | 3.54 | 3.5–4.5 | in range |

Only five parameters are fitted; mass, wheelbase, weight distribution and
steering lock are regulation figures held fixed, because an optimiser allowed to
move those buys lap time by inventing a car that does not exist. Check it
yourself with `./build/racing_calibrate`.

[`docs/RACING_ENGINE.md`](docs/RACING_ENGINE.md) lists the whole model **and
what it deliberately leaves out** — suspension travel, Pacejka tyres, wheel
rotational dynamics, rain — with the reason for each.

The circuit is the line a real car actually drove, reconstructed from position
telemetry — see [`tools/build_track.py`](tools/build_track.py) for why that
needs care.

---

## Watching it

The feed is the point. Every car's position, heading, speed, controls, race
position, gap, aero state, tyre temperatures and wear, fuel, gear, rpm, battery
and g-forces — 28 fields per car, sampled at 60 Hz of simulated time:

```bash
python examples/race.py --laps 3 --name demo          # replay
python examples/race.py --laps 3 --name demo --live   # + stream it as it runs
python examples/race.py --fps 100                     # every physics step
```

Writes `episodes/demo/` — `track.json`, `episode.json`, `frames.f32`, and
optionally `stream.jsonl`.

**If you are building the visualizer, [`docs/VISUALIZER_FEED.md`](docs/VISUALIZER_FEED.md)
is the only file you need.** You do not have to build or run the simulator; ask
for an episode folder and point a loader at it. There is a runnable reference
viewer in [`docs/visualizer-example/`](docs/visualizer-example/):

```bash
python -m http.server 8777
# http://127.0.0.1:8777/docs/visualizer-example/index.html
```

The format is self-describing — `episode.json` names every field and its
stride — so it can grow without breaking a reader that looks fields up by name.
It already has, twice: the stride went from 16 to 27 when the tyre and
powertrain physics landed and from 27 to 28 when damage did, and every field
that existed before is still there under the same name.

A 3-lap 20-car race at 60 Hz is 56 MB. Drop `--fps` if that is too much.

---

## Build

C++20, CMake ≥3.20, and pybind11. Nothing else at runtime: nlohmann/json and
Catch2 are vendored, so a fresh clone builds with no network access.

```bash
python -m venv .venv
.venv/bin/pip install pybind11 pytest numpy      # Windows: .venv\Scripts\pip

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build
```

On Windows use the `x64 Native Tools` prompt (or run `vcvars64.bat` first) so
CMake finds MSVC, and pass `-G Ninja`.

```bash
./build/racing_tests                 # 76 cases
python -m pytest tests/              # bindings and the feed contract
./build/racing_bench 256 8           # throughput
./build/racing_calibrate             # is the car still the right car
```

The Python module lands in `python/racing/`, so `PYTHONPATH=python` is enough to
import it from the source tree.

## Training

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python examples/train_marl.py --envs 256 --threads 8
python examples/race.py --policy runs/latest.pt --name trained
```

Multi-agent PPO with parameter sharing: one network drives every car in every
race, and improves by racing copies of itself. What makes two cars behave
differently is not their weights but what they can see — their own state, the
road ahead, and the nearest few cars, each flagged as team mate or rival.

It trains in two phases, because they are different problems: first one car on
an empty circuit learning to drive, then the full field learning to race.
Skipping the first works and takes far longer — a policy that cannot get round
Turn 1 never survives long enough to be near anybody.

Throughput is about 128k race-steps/s at 256 races × 20 cars across 8 threads —
roughly 1.0M car-steps/s, or 5,100× real time. The learner is the bottleneck,
not the simulator; check on your own machine with `./build/racing_bench 256 8`.

## Racing without training

[`examples/drivers.py`](examples/drivers.py) is a scripted field with actual
racecraft: it uses the tow, pulls out of dirty air to attack, defends where
there is room to, holds a safe following distance into braking zones, and backs
off when its tyres are cold or its tank is full. It is
the baseline a policy has to beat, it proves the environment is raceable, and it
gives the visualizer something to render on day one.

```bash
python examples/race.py --laps 3          # scripted by default
```

## Layout

```
include/racing/   config.hpp     the dials: aero, contact, reward, teams
                  car.hpp        everything known about one car
                  interaction.hpp  the wake, and contact
                  race.hpp       the environment: N cars, T teams
                  feed.hpp       the visualizer seam
                  track.hpp  vehicle.hpp  qss.hpp  rng.hpp
src/racing/       the implementations
bindings/         pybind11 module
python/racing/    the package, plus a feed reader
examples/         race.py  drivers.py  train_marl.py
tools/            build_track.py     telemetry -> circuit
                  extract_circuit.py raw telemetry pull
                  calibrate_vehicle.cpp  the realism check, and the fit
                  bench.cpp
docs/             VISUALIZER_FEED.md   the format contract
                  RACING_ENGINE.md     how the engine works
                  visualizer-example/  a runnable reference viewer
```

## A note on the history

This started as a research project about team strategy — pit stops, tyre
compounds, radio messages, a leaky communication channel. That engine has been
removed; it is in the git history if it is ever wanted back:

```bash
git log --oneline -- include/f1marl
```

What survives from it is the discipline rather than the design: counter-based
randomness so a race replays bit-exactly under any thread count, batched
stepping so the language boundary is crossed once per policy step, and the idea
that the visualizer gets a stable, documented seam rather than whatever the
engine happens to have lying around.

## License

MIT. See [LICENSE](LICENSE).
