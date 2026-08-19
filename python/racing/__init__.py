"""Teams of AI cars racing each other.

The native module is built into this package directory, so importing works from
a source tree with no install step -- `PYTHONPATH=python` is enough.

Two entry points, for two jobs:

    racing.VecRaceEnv   many races at once, numpy in and out. For training.
    racing.RaceEnv      one race, with car states and events readable from
                        Python. For running a race you want to watch.

And one for getting it on screen:

    racing.Feed         writes the visualizer feed -- a replay, a live stream,
                        or both.
"""

from ._racing import (  # noqa: F401
    AeroConfig,
    CarState,
    ContactConfig,
    EnvConfig,
    Feed,
    FieldConfig,
    DamageConfig,
    Frenet,
    ObservationConfig,
    QssResult,
    RaceConfig,
    RaceEnv,
    RaceEvent,
    RewardConfig,
    SimConfig,
    StepInfo,
    TeamInfo,
    Track,
    TrackConfig,
    Vehicle,
    VehicleParams,
    VehicleState,
    VehicleTelemetry,
    VecRaceEnv,
    solve_qss,
    wake_strength,
)

from .feed import Episode, read_episode  # noqa: F401

__all__ = [
    "AeroConfig",
    "CarState",
    "ContactConfig",
    "EnvConfig",
    "Episode",
    "Feed",
    "FieldConfig",
    "DamageConfig",
    "Frenet",
    "ObservationConfig",
    "QssResult",
    "RaceConfig",
    "RaceEnv",
    "RaceEvent",
    "RewardConfig",
    "SimConfig",
    "StepInfo",
    "TeamInfo",
    "Track",
    "TrackConfig",
    "Vehicle",
    "VehicleParams",
    "VehicleState",
    "VehicleTelemetry",
    "VecRaceEnv",
    "read_episode",
    "solve_qss",
    "wake_strength",
]
