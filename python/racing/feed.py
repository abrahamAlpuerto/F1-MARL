"""Reading the visualizer feed back.

The engine writes this format; this reads it. It exists for three reasons: the
tests need to check the engine wrote what it said it wrote, analysis scripts
need the frames as an array, and -- mostly -- it is a short, complete worked
example of parsing the format for whoever is building the renderer.

It deliberately depends on nothing but numpy and the standard library, and it
does not import the engine. That is the point: if this file can read an episode,
so can anything else, and the format does not secretly require the simulator to
be installed.

    from racing import read_episode
    ep = read_episode("episodes/latest")
    xy = ep.field("x"), ep.field("y")          # each [n_frames, n_cars]
    ep.car(3)["speed"]                          # one car, all frames
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Episode:
    """One race, loaded from disk.

    `frames` is [n_frames, n_cars, stride]. Index the last axis by *name* --
    `index_of("speed")` or `field("speed")` -- never by a literal. Fields get
    appended to the format, and a hardcoded offset is how a viewer starts
    rendering throttle as lap number.
    """

    header: dict[str, Any]
    frames: np.ndarray
    track: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_cars(self) -> int:
        return int(self.frames.shape[1])

    @property
    def fields(self) -> list[str]:
        return list(self.header["fields"])

    @property
    def frame_rate(self) -> float:
        return float(self.header["frame_rate"])

    @property
    def duration(self) -> float:
        """Simulated seconds the episode covers."""
        return self.n_frames / self.frame_rate

    def index_of(self, name: str) -> int:
        try:
            return self.fields.index(name)
        except ValueError:
            raise KeyError(
                f"no field {name!r} in this episode; it has {self.fields}"
            ) from None

    def field(self, name: str) -> np.ndarray:
        """One field for every car and every frame, as [n_frames, n_cars]."""
        return self.frames[:, :, self.index_of(name)]

    def car(self, index: int) -> dict[str, np.ndarray]:
        """Every field for one car, as name -> [n_frames]."""
        return {n: self.frames[:, index, i] for i, n in enumerate(self.fields)}

    def car_name(self, index: int) -> str:
        return str(self.header["cars"][index]["name"])

    def team_of(self, index: int) -> int:
        return int(self.header["cars"][index]["team"])

    def team_color(self, team: int) -> str:
        return str(self.header["teams"][team]["color"])

    def events_of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == kind]

    def line(self) -> np.ndarray:
        """The circuit as [n, 3] of x, y, z. Requires the track to be loaded."""
        if self.track is None:
            raise RuntimeError("this episode was loaded without its track.json")
        return np.asarray(self.track["line"], dtype=np.float64)


def read_episode(directory: str, with_track: bool = True) -> Episode:
    """Load `episode.json` + `frames.f32` (+ `track.json`) from a directory."""
    with open(os.path.join(directory, "episode.json"), encoding="utf-8") as f:
        header = json.load(f)

    n_frames = int(header["n_frames"])
    n_cars = int(header["n_cars"])
    stride = int(header["stride"])

    raw = np.fromfile(os.path.join(directory, "frames.f32"), dtype="<f4")
    expected = n_frames * n_cars * stride
    if raw.size != expected:
        # A truncated download is the common failure and it must not render as
        # a blank screen with no explanation.
        raise ValueError(
            f"frames.f32 has {raw.size} floats, but episode.json describes "
            f"{n_frames} frames x {n_cars} cars x {stride} fields = {expected}"
        )

    track = None
    if with_track:
        path = os.path.join(directory, "track.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                track = json.load(f)

    return Episode(
        header=header,
        frames=raw.reshape(n_frames, n_cars, stride),
        track=track,
        events=list(header.get("events", [])),
        result=dict(header.get("result", {})),
    )


def read_stream(path: str) -> Episode:
    """Load a live `.jsonl` stream as if it were a replay.

    Slower and larger than the binary format -- it is meant to be tailed while
    a race runs, not stored -- but being able to load a finished one the same
    way is convenient enough to be worth the twenty lines.
    """
    header: dict[str, Any] | None = None
    rows: list[list[list[float]]] = []
    events: list[dict[str, Any]] = []
    result: dict[str, Any] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "header":
                header = obj
            elif kind == "frame":
                rows.append(obj["cars"])
            elif kind == "event":
                events.append(obj["event"])
            elif kind == "result":
                result = obj
            # Anything else is a type this reader predates. Ignore it rather
            # than failing: more of them are expected.

    if header is None:
        raise ValueError(f"{path} has no header line")

    frames = np.asarray(rows, dtype=np.float32) if rows else np.zeros(
        (0, int(header["n_cars"]), int(header["stride"])), dtype=np.float32
    )
    header = dict(header)
    header["n_frames"] = int(frames.shape[0])
    return Episode(header=header, frames=frames, events=events, result=result)
