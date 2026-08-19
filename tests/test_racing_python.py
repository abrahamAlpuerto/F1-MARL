"""Python-side tests: the bindings, the race, and the feed the visualizer eats.

The feed tests here are deliberately written the way the visualizer developer
will write them -- read the header, index fields by name, stride by what the
header says -- so that if the format ever stops being self-describing, these
fail rather than quietly still working because they knew the layout.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "examples"))

racing = pytest.importorskip("racing", reason="native module not built")

TRACK = os.path.join(REPO, "data", "tracks", "bahrain.json")


def make_config(teams=2, per_team=2, distance=700.0):
    c = racing.EnvConfig()
    c.track.path = TRACK
    c.track.randomize_start = False
    c.track.episode_distance = distance
    c.field.n_teams = teams
    c.field.cars_per_team = per_team
    c.seed = 11
    return c


@pytest.fixture(scope="module")
def track():
    return racing.Track.load(TRACK)


# --- track -----------------------------------------------------------------

def test_track_loads_with_elevation(track):
    assert track.n > 1000
    assert 5000 < track.length < 6000
    xy = track.xy()
    assert xy.shape == (track.n, 2)
    # Elevation is optional in the file but present in the shipped circuit, and
    # the visualizer draws with it.
    zs = [track.z_at(s) for s in np.linspace(0, track.length, 50)]
    assert max(zs) - min(zs) > 1.0


def test_frenet_round_trips(track):
    for s in (0.0, 1234.5, track.length - 1.0):
        x, y = track.to_world(s, 2.0)
        f = track.project_global(x, y, 0.0)
        assert track.delta_s(f.s, s) == pytest.approx(0.0, abs=1.5)
        assert f.e_y == pytest.approx(2.0, abs=0.3)


# --- the race --------------------------------------------------------------

def test_race_runs_and_classifies():
    env = racing.RaceEnv(make_config(), 0)
    env.reset(0)
    assert env.n_cars == 4
    assert list(env.teams) == [0, 0, 1, 1]

    act = np.zeros((env.n_cars, 2), dtype=np.float32)
    act[:, 1] = 0.35
    steps = 0
    while not env.done and steps < 5000:
        rewards, info = env.step(act)
        assert rewards.shape == (env.n_cars,)
        assert np.all(np.isfinite(rewards))
        steps += 1

    assert env.done
    positions = sorted(c.position for c in env.cars)
    assert positions == list(range(1, env.n_cars + 1))
    assert len(env.finish_order()) == env.n_cars
    assert sum(env.team_scores()) == env.n_cars * (env.n_cars - 1) // 2


def test_observation_shape_and_range():
    env = racing.RaceEnv(make_config(), 0)
    env.reset(0)
    obs = env.observe()
    assert obs.shape == (env.n_cars, env.obs_dim)
    assert np.all(np.isfinite(obs))
    # Normalised: nothing in here is a world coordinate.
    assert np.abs(obs).max() < 50.0


def test_vec_env_matches_single_env():
    """A batched race and a single race with the same seed must agree.

    They are the same code path, but the batched one auto-resets and threads,
    and a divergence here is exactly the kind of bug that only shows up as
    "training learned something the replay does not do".
    """
    cfg = make_config()
    vec = racing.VecRaceEnv(cfg, 2, 1)
    single = racing.RaceEnv(cfg, 0)

    vec.reset_all(0)
    single.reset(0)

    obs = np.zeros((2, vec.n_cars, vec.obs_dim), np.float32)
    rew = np.zeros((2, vec.n_cars), np.float32)
    done = np.zeros(2, np.uint8)
    act = np.zeros((2, vec.n_cars, 2), np.float32)
    act[:, :, 1] = 0.4

    for _ in range(50):
        vec.step(act, obs, rew, done)
        single_rew, _ = single.step(act[0])
        np.testing.assert_allclose(rew[0], single_rew, rtol=0, atol=0)


def test_team_weight_one_gives_a_team_one_reward():
    cfg = make_config(teams=2, per_team=2)
    cfg.reward.team_weight = 1.0
    env = racing.RaceEnv(cfg, 0)
    env.reset(0)

    act = np.zeros((env.n_cars, 2), dtype=np.float32)
    act[:, 0] = [0.0, 0.2, -0.2, 0.1]
    act[:, 1] = 0.4
    for _ in range(30):
        rewards, _ = env.step(act)
        assert rewards[0] == pytest.approx(rewards[1])
        assert rewards[2] == pytest.approx(rewards[3])


def test_wake_helper_matches_the_engine():
    aero = racing.AeroConfig()
    assert racing.wake_strength(aero, 5.0, 0.0) > racing.wake_strength(aero, 50.0, 0.0)
    assert racing.wake_strength(aero, 5.0, 0.0) > racing.wake_strength(aero, 5.0, 4.0)
    assert racing.wake_strength(aero, -5.0, 0.0) == 0.0


# --- the feed ---------------------------------------------------------------

@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    """One short race, written out, shared by every feed test below."""
    out = tmp_path_factory.mktemp("episode")
    cfg = make_config(teams=2, per_team=2, distance=900.0)
    # Held at a fixed throttle with no steering, these cars leave the circuit and
    # are carried round by the recovery mechanism. That is fine for exercising
    # the feed FORMAT, which is what everything below is about, but with damage
    # on it is four cars driving into barriers -- they all retire, nobody
    # finishes, and the tests that check a finish event stop meaning anything.
    # Damage in the feed has its own test at the bottom of this file.
    cfg.damage.enabled = False
    env = racing.RaceEnv(cfg, 0)
    env.reset(0)

    feed = racing.Feed(env, 60.0)
    feed.attach(env)
    feed.open_stream(str(out / "stream.jsonl"), env)

    act = np.zeros((env.n_cars, 2), dtype=np.float32)
    act[:, 1] = 0.4
    steps = 0
    while not env.done and steps < 5000:
        env.step(act)
        feed.collect_events(env)
        steps += 1

    feed.close_stream(env)
    feed.write(str(out), env)
    return str(out)


def test_feed_writes_all_three_files(episode):
    for name in ("track.json", "episode.json", "frames.f32"):
        assert os.path.exists(os.path.join(episode, name)), name


def test_feed_is_self_describing(episode):
    """Read it the way a visualizer must: by name, never by a literal offset."""
    with open(os.path.join(episode, "episode.json"), encoding="utf-8") as f:
        header = json.load(f)

    assert header["format"] == "racing-feed"
    fields = header["fields"]
    stride = header["stride"]
    assert len(fields) == stride
    for required in ("x", "y", "z", "heading", "speed", "position", "lap",
                     "damage"):
        assert required in fields

    raw = np.fromfile(os.path.join(episode, "frames.f32"), dtype="<f4")
    assert raw.size == header["n_frames"] * header["n_cars"] * stride

    frames = raw.reshape(header["n_frames"], header["n_cars"], stride)
    x = frames[:, :, fields.index("x")]
    y = frames[:, :, fields.index("y")]
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    # The cars went somewhere.
    assert np.ptp(x) > 100.0


def test_reader_round_trips(episode):
    ep = racing.read_episode(episode)
    assert ep.n_cars == 4
    assert ep.n_frames > 100
    assert ep.frame_rate == 60.0
    assert ep.duration > 1.0

    speed = ep.field("speed")
    assert speed.shape == (ep.n_frames, ep.n_cars)
    assert speed.max() > 20.0

    car = ep.car(0)
    assert set(car) == set(ep.fields)
    assert ep.car_name(0)
    assert ep.team_of(3) == 1
    assert ep.team_color(0).startswith("#")

    with pytest.raises(KeyError):
        ep.index_of("tyre_temperature")

    line = ep.line()
    assert line.shape[1] == 3  # x, y, z


def test_positions_are_a_permutation_every_frame(episode):
    ep = racing.read_episode(episode)
    pos = ep.field("position").astype(int)
    expected = np.arange(1, ep.n_cars + 1)
    for frame in range(0, ep.n_frames, 7):
        assert np.array_equal(np.sort(pos[frame]), expected)


def test_events_reference_real_frames_and_cars(episode):
    ep = racing.read_episode(episode)
    assert ep.events
    last = -1
    for e in ep.events:
        assert 0 <= e["frame"] < ep.n_frames
        assert e["frame"] >= last  # monotonic, so a reader can walk it in order
        last = e["frame"]
        assert 0 <= e["car"] < ep.n_cars
        assert e["type"]
    assert ep.events_of_type("finish")
    assert len(ep.result["finish_order"]) == ep.n_cars


def test_truncated_frames_file_is_an_error_not_a_blank_screen(episode, tmp_path):
    """The common real-world failure, and it must say so."""
    import shutil

    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("episode.json", "track.json"):
        shutil.copy(os.path.join(episode, name), broken / name)
    raw = open(os.path.join(episode, "frames.f32"), "rb").read()
    (broken / "frames.f32").write_bytes(raw[: len(raw) // 3])

    with pytest.raises(ValueError, match="frames.f32"):
        racing.read_episode(str(broken))


def test_live_stream_is_line_delimited_json(episode):
    from racing.feed import read_stream

    path = os.path.join(episode, "stream.jsonl")
    kinds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                kinds.append(json.loads(line)["type"])

    assert kinds[0] == "header"       # so a late reader still knows the layout
    assert kinds[-1] == "result"
    assert kinds.count("header") == 1
    assert kinds.count("result") == 1
    assert kinds.count("frame") > 10

    streamed = read_stream(path)
    replay = racing.read_episode(episode)
    assert streamed.n_cars == replay.n_cars
    assert streamed.n_frames == replay.n_frames
    # Same race, same numbers, whichever way it was written.
    np.testing.assert_allclose(streamed.frames, replay.frames, rtol=0, atol=0)


# --- the examples still run -------------------------------------------------

def test_scripted_field_races_cleanly():
    """The scripted drivers must get a full field round without falling apart.

    This is the test that would have caught every driver bug found so far: the
    unit tests all passed while the field piled into Turn 1 and stopped.
    """
    import drivers as scripted

    cfg = make_config(teams=4, per_team=2, distance=-1.0)
    cfg.track.laps = 1
    env = racing.RaceEnv(cfg, 0)
    env.reset(3)
    field = scripted.build_field(env, seed=3)

    off_track = 0
    steps = 0
    while not env.done and steps < 6000:
        env.step(scripted.drive(field, env))
        off_track += sum(1 for e in env.events if e.name == "off_track")
        steps += 1

    assert env.done, "the scripted field did not finish the race"

    # A retirement is now a legitimate outcome rather than a bug, so this asks
    # that the field gets round rather than that every single car does.
    #
    # This assertion used to be `all(c.lap >= 1)`, and it was failing on Linux
    # and macOS while passing on Windows. The cause is worth recording: one car
    # takes a genuine ~21 m/s hit into the barrier and ends the lap on 98.7%
    # damage, one and a bit percent short of the threshold that retires it.
    # Which side of that line it lands on comes down to floating point, so the
    # platforms disagreed -- on Linux it retired and finished the lap on zero.
    #
    # Requiring every car home was only ever right in a world with no DNFs in
    # it. The scripted field averages 0.4 retirements a race by design, so the
    # check is now that at most one car is out and everyone still running
    # completed the lap. A field that actually falls apart -- the thing this
    # test exists to catch -- still fails it loudly.
    out = [c.index for c in env.cars if c.retired]
    assert len(out) <= 1, f"the field fell apart: {len(out)} cars retired"
    assert all(c.lap >= 1 for c in env.cars if not c.retired), \
        f"cars failed to complete a lap: {[c.lap for c in env.cars]}"
    # A handful of excursions across eight cars is racing; hundreds is a bug.
    assert off_track < 40, f"{off_track} off-track excursions in one lap"
    # Only from cars that set one. A car that retired before completing a lap
    # carries best_lap_time 0.0, which would sink this min() to zero and fail
    # the lower bound for a reason that has nothing to do with lap times.
    laps_set = [c.best_lap_time for c in env.cars if c.best_lap_time > 0.0]
    assert laps_set, "nobody completed a timed lap"
    best = min(laps_set)
    assert 95.0 < best < 150.0, f"implausible best lap {best:.1f} s"


def test_race_example_runs_end_to_end(tmp_path):
    out = tmp_path / "race"
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, "examples", "race.py"),
         "--laps", "1", "--teams", "2", "--cars-per-team", "2",
         "--seed", "5", "--out", str(out), "--live"],
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    ep = racing.read_episode(str(out))
    assert ep.n_frames > 100
    assert os.path.exists(out / "stream.jsonl")


# --- damage and retirement in the feed --------------------------------------

def test_damage_and_retirement_reach_the_feed(tmp_path):
    """A DNF has to be legible to a renderer: the flag, the field, the event.

    Four cars held at full lock and full throttle go off, cross the run-off and
    hit the wall, which is the whole causal chain the damage model exists for.
    """
    cfg = make_config(teams=2, per_team=2, distance=4000.0)
    cfg.reward.terminate_off_track = False
    assert cfg.damage.enabled

    env = racing.RaceEnv(cfg, 0)
    env.reset(0)
    feed = racing.Feed(env, 60.0)
    feed.attach(env)

    act = np.zeros((env.n_cars, 2), dtype=np.float32)
    act[:, 0] = 1.0
    act[:, 1] = 1.0

    reasons = []
    steps = 0
    while not env.done and steps < 5000:
        env.step(act)
        feed.collect_events(env)
        for e in env.events:
            if e.name == "retire":
                reasons.append(e.reason)
        steps += 1
    feed.write(str(tmp_path), env)

    assert reasons, "nobody retired, so this test proves nothing"
    assert set(reasons) <= {"collision", "barrier", "off_track"}

    ep = racing.read_episode(str(tmp_path))
    assert ep.header["version"] >= 4

    # Damage is published in 0..1 and never decreases.
    dmg = ep.field("damage")
    assert dmg.min() >= 0.0
    assert dmg.max() <= 1.0 + 1e-6
    assert np.all(np.diff(dmg, axis=0) >= -1e-6)
    assert dmg.max() > 0.0

    # The retire events carry a reason, and the classification agrees with them.
    retires = ep.events_of_type("retire")
    assert retires
    for e in retires:
        assert e["reason"] in ("collision", "barrier", "off_track")

    retired_cars = {e["car"] for e in retires}
    for row in ep.result["classification"]:
        if row["car"] in retired_cars:
            assert row["retired"] is True
            assert row["retire_reason"] in ("collision", "barrier", "off_track")

    # And the RETIRED flag is set for every frame after the car is out.
    flags = ep.field("flags").astype(np.int64)
    for car in retired_cars:
        set_at = np.argmax((flags[:, car] & 8) != 0)
        assert (flags[set_at:, car] & 8).all(), "a retired car came back"


def test_a_retired_car_stops_moving(tmp_path):
    cfg = make_config(teams=1, per_team=2, distance=4000.0)
    cfg.reward.terminate_off_track = False
    env = racing.RaceEnv(cfg, 0)
    env.reset(0)
    feed = racing.Feed(env, 60.0)
    feed.attach(env)

    act = np.zeros((env.n_cars, 2), dtype=np.float32)
    act[:, 0] = 1.0
    act[:, 1] = 1.0
    steps = 0
    while not env.done and steps < 5000:
        env.step(act)
        steps += 1
    feed.write(str(tmp_path), env)

    ep = racing.read_episode(str(tmp_path))
    flags = ep.field("flags").astype(np.int64)
    x, y = ep.field("x"), ep.field("y")
    speed = ep.field("speed")

    out = [c for c in range(ep.n_cars) if (flags[-1, c] & 8) != 0]
    assert out, "expected at least one retirement"
    for c in out:
        first = int(np.argmax((flags[:, c] & 8) != 0))
        if first + 2 >= ep.n_frames:
            continue
        after = slice(first + 1, None)
        assert np.allclose(x[after, c], x[first + 1, c])
        assert np.allclose(y[after, c], y[first + 1, c])
        assert np.all(speed[after, c] < 1e-3)
