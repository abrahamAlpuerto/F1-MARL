"""Run a race and write it out for the visualizer.

    python examples/race.py                          # scripted drivers
    python examples/race.py --policy runs/latest.pt  # a trained policy
    python examples/race.py --live                   # also stream it as it runs

Writes `episodes/<name>/` containing `track.json`, `episode.json` and
`frames.f32` -- the replay -- and optionally `stream.jsonl` alongside, which is
the same race as a line-per-frame live feed.

The positions come out at `--fps` hertz of simulated time, 60 by default. That
is the number to turn if the renderer wants more or fewer; the engine is
sampling its own 100 Hz physics down to it, so anything up to 100 is free of
interpolation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import racing  # noqa: E402
import drivers as scripted  # noqa: E402


def build_config(args):
    cfg = racing.EnvConfig()
    cfg.track.path = args.track
    cfg.track.laps = args.laps
    cfg.track.randomize_start = False
    cfg.field.n_teams = args.teams
    cfg.field.cars_per_team = args.cars_per_team
    cfg.seed = args.seed
    cfg.reward.team_weight = args.team_weight
    # A race, not a training episode: a car that runs wide rejoins rather than
    # being deleted from the field.
    cfg.reward.terminate_off_track = False
    cfg.aero.tow_max = args.tow
    cfg.aero.wash_max = args.dirty_air
    return cfg


def load_policy(path, obs_dim):
    """Load a checkpoint written by train_marl.py."""
    import torch  # imported here so a scripted race needs no torch

    from train_marl import policy_from_checkpoint  # noqa: WPS433

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # The checkpoint says which architecture it is, so a race does not have to
    # know or care -- and an old checkpoint from before there was a choice
    # still loads as the MLP it was.
    net = policy_from_checkpoint(ckpt)
    if ckpt["obs_dim"] != obs_dim:
        raise SystemExit(
            f"{path} was trained on obs_dim {ckpt['obs_dim']}, but this race "
            f"has {obs_dim}. The observation layout must match."
        )
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", default=os.path.join(REPO, "data", "tracks", "bahrain.json"))
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--cars-per-team", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=float, default=60.0,
                    help="position samples per simulated second")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--name", default="latest")
    ap.add_argument("--live", action="store_true",
                    help="also write stream.jsonl as the race runs")
    ap.add_argument("--policy", default=None,
                    help="checkpoint from train_marl.py; scripted drivers if omitted")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy instead of taking its mean action. "
                         "Without this a policy races IDENTICALLY for every "
                         "--seed: the grid is fixed and the engine draws no "
                         "randomness during a race, so there is nothing left "
                         "for a seed to vary")
    ap.add_argument("--team-weight", type=float, default=0.5)
    ap.add_argument("--tow", type=float, default=0.32)
    ap.add_argument("--dirty-air", type=float, default=0.35)
    ap.add_argument("--pace", type=float, default=0.86,
                    help="scripted driver pace, as a fraction of the limit")
    args = ap.parse_args()

    cfg = build_config(args)
    if args.policy:
        # The checkpoint decides what the car can see; see train_marl.py.
        import torch
        ck = torch.load(args.policy, map_location="cpu", weights_only=False)
        if ck.get("observation", "frenet") == "sensor":
            cfg.observation.mode = racing.ObservationConfig.SENSOR
    env = racing.RaceEnv(cfg, 0)
    env.reset(args.seed)

    out_dir = args.out or os.path.join(REPO, "episodes", args.name)
    os.makedirs(out_dir, exist_ok=True)

    feed = racing.Feed(env, args.fps)
    feed.attach(env)
    if args.live:
        feed.open_stream(os.path.join(out_dir, "stream.jsonl"), env)

    net = None
    if args.policy:
        net = load_policy(args.policy, env.obs_dim)
        field = None
        if args.stochastic:
            # Seed torch as well as the engine, or --seed changes the race
            # without being able to reproduce it.
            import torch
            torch.manual_seed(args.seed)
    else:
        field = scripted.build_field(env, seed=args.seed, base_pace=args.pace)

    n_cars = env.n_cars
    print(f"{env.track.name}: {n_cars} cars, {args.teams} teams, "
          f"{args.laps} laps, {env.race_distance / 1000:.2f} km")
    print(f"feed: {args.fps:g} Hz -> {out_dir}")

    wall0 = time.perf_counter()
    overtakes = contacts = retired = 0
    while not env.done:
        if net is not None:
            import torch
            with torch.no_grad():
                obs = torch.from_numpy(env.observe())
                act = net.act(obs, deterministic=not args.stochastic).numpy()
        else:
            act = scripted.drive(field, env)

        env.step(act.astype(np.float32))
        feed.collect_events(env)

        for e in env.events:
            if e.name == "overtake":
                overtakes += 1
                print(f"  [{e.time:6.1f}s] lap {e.lap}  "
                      f"{feed_name(env, e.car)} passes {feed_name(env, e.other)}")
            elif e.name == "contact" and e.value > 0.25:
                contacts += 1
                print(f"  [{e.time:6.1f}s] contact: {feed_name(env, e.car)} "
                      f"into {feed_name(env, e.other)} ({e.value:.2f})")
            elif e.name == "rejoin" and e.value > 0.5:
                print(f"  [{e.time:6.1f}s] {feed_name(env, e.car)} recovered")
            elif e.name == "retire":
                retired += 1
                print(f"  [{e.time:6.1f}s] RETIRED: {feed_name(env, e.car)} "
                      f"({e.reason})")

    wall = time.perf_counter() - wall0
    if args.live:
        feed.close_stream(env)
    feed.write(out_dir, env)

    print()
    print(f"{'':>3}  {'car':<6} {'team':<10} {'laps':>4} {'time':>9} "
          f"{'best lap':>9} {'dmg':>5}  status")
    for idx in env.finish_order():
        car = env.cars[idx]
        t = car.finish_time if car.finished else car.race_time
        best = f"{car.best_lap_time:9.3f}" if car.best_lap_time > 0 else "        -"
        status = f"DNF ({car.retire_reason})" if car.retired else ""
        print(f"{car.position:>3}. {feed_name(env, idx):<6} "
              f"{team_name(idx, cfg):<10} {car.lap:>4} {t:9.3f} {best} "
              f"{car.damage * 100:4.0f}%  {status}")

    scores = env.team_scores()
    print()
    print("teams: " + "  ".join(
        f"{team_name(t * cfg.field.cars_per_team, cfg)} {s}"
        for t, s in enumerate(scores)))
    print(f"\n{overtakes} overtakes, {contacts} notable contacts, "
          f"{retired} retirement{'' if retired == 1 else 's'}, "
          f"{feed.n_frames} frames over {env.race_time:.1f} s "
          f"(simulated in {wall:.1f} s wall)")
    print(f"written to {out_dir}")


# The engine names cars inside the feed; these mirror it for the console so the
# printed commentary and the replay agree on who is who. Keep them in step with
# kDefaultTeams in src/racing/feed.cpp -- including the distinct initials, which
# is what stops two teams' cars both being called V1.
_TEAM_NAMES = ["Vermilion", "Cobalt", "Jade", "Amber", "Indigo",
               "Slate", "Rose", "Teal", "Ochre", "Fuchsia"]


def team_name(car_index, cfg):
    t = car_index // cfg.field.cars_per_team
    return _TEAM_NAMES[t] if t < len(_TEAM_NAMES) else f"Team {t + 1}"


def feed_name(env, car_index):
    if car_index < 0:
        return "?"
    per_team = env.config.field.cars_per_team
    return f"{team_name(car_index, env.config)[0]}{car_index % per_team + 1}"


if __name__ == "__main__":
    main()
