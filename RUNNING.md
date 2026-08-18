# Running everything

Every command in this repository, in the order you would actually need them.
Run all of them from the repository root.

Windows commands are given first because that is where this checkout lives; the
Linux/macOS equivalent follows each one.

---

## 0. The short version

If you only want to see a race:

```bash
python -m venv .venv
.venv\Scripts\pip install pybind11 pytest numpy
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=%CD%\.venv\Scripts\python.exe
cmake --build build --config Release
.venv\Scripts\python examples\race.py --laps 3 --name demo
```

That writes `episodes/demo/`. Jump to [Watching it](#6-watching-it) to get it on
screen.

---

## 1. Prerequisites

| | |
|---|---|
| C++20 compiler | MSVC 2019+ on Windows, GCC 10+ / Clang 12+ elsewhere |
| CMake | >= 3.20 |
| Python | >= 3.9 |
| Ninja | optional, but recommended on Windows |

Nothing else is needed at build time. `nlohmann/json` and Catch2 are vendored in
`third_party/`, so a fresh clone builds with no network access.

**Windows only:** open the **x64 Native Tools Command Prompt for VS**, or run
`vcvars64.bat` in your shell first. Without it CMake will not find MSVC. Every
Windows command below assumes you are in that shell.

---

## 2. One-time setup

Create a virtualenv and install the build/test dependencies.

**Windows**

```bash
python -m venv .venv
.venv\Scripts\pip install --upgrade pip pybind11 pytest numpy
```

**Linux / macOS**

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip pybind11 pytest numpy
```

Optional extras, only if you need them:

```bash
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv\Scripts\pip install fastf1 pandas
```

`torch` is for training (§7). `fastf1` + `pandas` is for rebuilding the circuit
from live telemetry (§8). The engine, the scripted drivers and the feed need
neither.

---

## 3. Build

The `-DPython_EXECUTABLE` flag matters: it is how CMake finds the interpreter
that pybind11 was installed into. Point it at the venv, not at whatever `python`
happens to be on PATH, or the bindings are silently skipped.

**Windows**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=%CD%\.venv\Scripts\python.exe
```

```bash
cmake --build build --config Release --parallel
```

**Linux / macOS**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PWD/.venv/bin/python
```

```bash
cmake --build build --parallel
```

This produces:

| target | what it is |
|---|---|
| `racing_core` | the engine, static library |
| `racing_tests` | the C++ test suite (Catch2) |
| `racing_bench` | throughput benchmark |
| `racing_calibrate` | the realism check against the real Bahrain lap |
| `_racing` | the pybind11 module, written into `python/racing/` |

The Python module lands **inside the package directory** on purpose, so
`import racing` works from the source tree with no install step. The example
scripts add `python/` to `sys.path` themselves; for your own scripts set
`PYTHONPATH=python`.

Two CMake options exist if you want a narrower build:

```bash
cmake -S . -B build -DRACING_BUILD_TESTS=OFF -DRACING_BUILD_BINDINGS=OFF
```

### Where the binaries land

Single-config generators (Ninja, Make) put them in `build/`. Multi-config
generators (Visual Studio) put them in `build/Release/` with an `.exe` suffix.
Both layouts appear below.

---

## 4. Verifying the build

Run these in order. Each one checks something the next assumes.

**C++ tests** — the whole suite:

```bash
ctest --test-dir build --output-on-failure -C Release
```

Or the binary directly, which lets you select tags:

```bash
build\Release\racing_tests.exe
```

```bash
build\Release\racing_tests.exe "[determinism]"
```

The four tags CI gates on individually are `[determinism]`, `[feed]`, `[aero]`
and `[contact]`. A determinism failure means a replay is no longer bit-exact
across thread counts, which invalidates every recorded episode — treat that as
the serious one.

**Python tests** — the bindings and the feed format contract:

```bash
.venv\Scripts\python -m pytest tests/ -v
```

**Throughput** — arguments are `<n_envs> <n_threads>`:

```bash
build\Release\racing_bench.exe 256 8
```

**Calibration** — is the car still the right car:

```bash
build\Release\racing_calibrate.exe
```

This refits against the real 2024 Bahrain pole lap and prints lap time, top
speed, minimum speed and aero efficiency against the real figures. If lap time
has drifted well away from ~92.8 s, something in the physics changed.

---

## 5. Running a race

The default is the scripted field from `examples/drivers.py` — no training, no
torch, works the moment the build finishes.

```bash
.venv\Scripts\python examples\race.py --laps 3 --name demo
```

That writes `episodes/demo/` containing `track.json`, `episode.json` and
`frames.f32`. Add `--live` and you also get `stream.jsonl`, the same race
written a line per frame as it runs.

```bash
.venv\Scripts\python examples\race.py --laps 3 --name demo --live
```

### The flags

| flag | default | what it does |
|---|---|---|
| `--laps` | 3 | race distance |
| `--teams` | 10 | number of teams |
| `--cars-per-team` | 2 | field size is `teams x cars-per-team` |
| `--seed` | 1 | the same seed replays bit-exactly |
| `--fps` | 60 | position samples per *simulated* second; 100 is every physics step |
| `--name` | `latest` | episode folder name under `episodes/` |
| `--out` | — | full output path, overrides `--name` |
| `--live` | off | also write `stream.jsonl` as the race runs |
| `--policy` | — | checkpoint from `train_marl.py`; scripted drivers if omitted |
| `--track` | `data/tracks/bahrain.json` | circuit to race on |
| `--team-weight` | 0.5 | 0 = every car for itself, 1 = fully cooperative |
| `--tow` | 0.32 | drag reduction behind another car |
| `--dirty-air` | 0.35 | downforce loss behind another car |
| `--pace` | 0.86 | scripted driver pace, as a fraction of the limit |

A 3-lap 20-car race at 60 Hz is about 56 MB. Drop `--fps` if that is too much.

---

## 6. Watching it

There is a runnable reference viewer in `docs/visualizer-example/`. It fetches
the episode over HTTP, so `file://` will not work — serve the repository root:

```bash
.venv\Scripts\python -m http.server 8777
```

Then open <http://127.0.0.1:8777/docs/visualizer-example/index.html>.

The box at the top of the page is the episode directory, relative to the page.
Point it at any episode you have generated — `../../episodes/demo`.

You do **not** need a C++ toolchain to build a visualizer against this. Ask for
an `episodes/<name>/` folder, drop it anywhere, and read
[`docs/VISUALIZER_FEED.md`](docs/VISUALIZER_FEED.md) — that is the whole
contract.

---

## 7. Training a policy

Needs `torch`. CPU is fine; the learner is the bottleneck, not the simulator.

```bash
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv\Scripts\python examples\train_marl.py --envs 256 --threads 8
```

Training runs in two phases, because they are two different problems: phase 1 is
one car on an empty circuit learning to drive, phase 2 is the full field
learning to race. Skipping phase 1 works and takes far longer.

The checkpoint is written to `runs/latest.pt`. Race it:

```bash
.venv\Scripts\python examples\race.py --policy runs\latest.pt --name trained
```

### The flags

| flag | default | what it does |
|---|---|---|
| `--envs` | 128 | parallel races |
| `--threads` | 8 | worker threads stepping them |
| `--drive-iters` | 150 | phase 1 iterations; `0` skips straight to racing |
| `--iters` | 250 | phase 2 iterations |
| `--horizon` | 64 | rollout length per iteration |
| `--epochs` | 4 | PPO epochs per batch |
| `--minibatch` | 4096 | minibatch size |
| `--lr` | 3e-4 | learning rate |
| `--clip` | 0.2 | PPO clip range |
| `--vf-coef` | 0.5 | value loss coefficient |
| `--ent-coef` | 0.003 | entropy bonus |
| `--hidden` | 256 | network width |
| `--team-weight` | 0.5 | the cooperation dial, as in a race |
| `--position-weight` | 1.0 | how much a car is paid for position vs progress |
| `--episode-distance` | 3000.0 | metres per training episode |
| `--log-every` | 10 | iterations between log lines |
| `--out` | `runs/latest.pt` | checkpoint path |
| `--seed` | 0 | RNG seed |

Sanity-check your own machine before tuning `--envs` and `--threads`:

```bash
build\Release\racing_bench.exe 256 8
```

**Note:** a checkpoint records the observation dimension it was trained on. Race
it with a field size it was not trained for and `race.py` refuses, rather than
silently producing nonsense.

---

## 8. Rebuilding the circuit (optional)

`data/tracks/bahrain.json` is committed, so you never need this in order to
race. It is here for when the reconstruction itself changes.

Rebuild the racing line from the committed raw telemetry — needs only numpy:

```bash
.venv\Scripts\python tools\build_track.py --out data\tracks\bahrain.json
```

| flag | default |
|---|---|
| `--raw` | `data/bahrain_track.json` |
| `--out` | `data/tracks/bahrain.json` |
| `--n` | 2048 uniform arc-length samples |
| `--order` | 4 (Butterworth order) |

Re-pull the raw telemetry from the timing API — needs `fastf1`, `pandas` and
network access:

```bash
.venv\Scripts\python tools\extract_circuit.py --year 2024 --event Bahrain --session R
```

| flag | default |
|---|---|
| `--year` | 2024 |
| `--event` | `Bahrain` |
| `--session` | `R` |
| `--points` | 512 |
| `--out` | `data/bahrain_track.json` |
| `--cache` | `.fastf1_cache` |

Downloaded sessions land in `.fastf1_cache/`, which is gitignored.

---

## 9. Everything at a glance

```bash
python -m venv .venv
.venv\Scripts\pip install pybind11 pytest numpy
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=%CD%\.venv\Scripts\python.exe
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure -C Release
.venv\Scripts\python -m pytest tests/
build\Release\racing_calibrate.exe
build\Release\racing_bench.exe 256 8
.venv\Scripts\python examples\race.py --laps 3 --name demo
.venv\Scripts\python -m http.server 8777
```

---

## 10. When it does not work

**`pybind11 not found -- skipping python bindings`** during configure.
CMake found a different interpreter than the one holding pybind11. Delete
`build/`, reconfigure with an explicit absolute `-DPython_EXECUTABLE` pointing
into the venv.

**`ModuleNotFoundError: No module named 'racing._racing'`.**
The native module did not build, or built somewhere else. It has to exist as
`python/racing/_racing*.pyd` (Windows) or `python/racing/_racing*.so`. Rebuild;
if configure printed the pybind11 warning above, that is the cause.

**`No module named racing` from a script of your own.**
The example scripts add `python/` to `sys.path` themselves. Yours does not — set
`PYTHONPATH=python`, or run from the repository root with `-m`.

**CMake cannot find a compiler, on Windows.**
You are not in the x64 Native Tools prompt. Open that shell, or run
`vcvars64.bat`, then delete `build/` and reconfigure — the generator choice is
cached.

**Determinism tests fail.**
Check nobody added `-ffast-math` or `/fp:fast`. Float reassociation breaks
bit-exact replay in ways that only show up at some optimisation levels, which is
why `CMakeLists.txt` pins `-fno-fast-math` and `/fp:precise` explicitly.

**The viewer shows nothing.**
You are probably on `file://`. Serve the repository root over HTTP and load the
page from `127.0.0.1`. Then check that the episode directory in the box at the
top of the page really points at a folder containing `episode.json`.

**Episodes are enormous.**
Lower `--fps`. The physics runs at 100 Hz, so anything up to 100 is free of
interpolation and anything below is a straight sampling reduction.

---

## Related documents

- [`README.md`](README.md) — what the project is, and why it is built this way
- [`docs/RACING_ENGINE.md`](docs/RACING_ENGINE.md) — the physics model, and what
  it deliberately leaves out
- [`docs/VISUALIZER_FEED.md`](docs/VISUALIZER_FEED.md) — the feed format
  contract; the only file a visualizer author needs
- [`docs/visualizer-example/README.md`](docs/visualizer-example/README.md) — the
  reference viewer
