"""Measure a field: how quick it is, how watchable it is, and whether the teams
are actually working together.

    python examples/evaluate.py                          # the scripted baseline
    python examples/evaluate.py --policy runs/latest.pt  # a trained policy
    python examples/evaluate.py --json runs/eval.json    # keep the numbers

This exists because "is the new policy better?" is otherwise a question answered
by watching a replay and forming an impression. Every architecture change wants
a number before and after, and two of the three things this project actually
cares about -- team tactics, and racing worth watching -- have no obvious one.

So they are given operational definitions here, and the definitions are the
interesting part of this file:

**Racecraft** is pace and survival. Beat the scripted field on race time without
retiring more cars, and the policy can drive.

**Watchability** is a proxy for whether anything is happening. A field that
strings out into a procession scores badly no matter how quick it is; so does
one that crashes constantly. What is rewarded is cars near each other, changing
places.

**Team tactics** is the hard one, and it is measured against CHANCE rather than
in absolute terms. With ten teams of two, the car in front of you is a team mate
1 time in 19 -- 5.3% -- so a field with no team behaviour at all still tows its
team mates 5.3% of the time. The number that means something is the ratio to
that baseline. A policy that has learned to run nose-to-tail with its team mate
scores well above 1.0; a policy that ignores teams scores 1.0; a policy that
actively avoids its team mate scores below.

Everything here is computed from what the engine already publishes on CarState
(`wake_source`, `team`, `position`, ...) rather than from anything added for the
benefit of the metric, so none of it can drift away from what the race did.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import racing  # noqa: E402
import drivers as scripted  # noqa: E402

# A wake this strong is a car deliberately sitting in someone's air rather than
# happening to be within a hundred metres of them.
TOW_WAKE = 0.15
# Inside this, two cars are having a fight rather than sharing a circuit.
BATTLE_GAP_S = 1.0


def build_config(args):
    cfg = racing.EnvConfig()
    cfg.track.path = args.track
    cfg.track.laps = args.laps
    cfg.track.randomize_start = False
    cfg.field.n_teams = args.teams
    cfg.field.cars_per_team = args.cars_per_team
    cfg.reward.team_weight = args.team_weight
    # A race, not a training episode.
    cfg.reward.terminate_off_track = False
    return cfg


def apply_checkpoint_observation(cfg, path):
    """Set the env's observation mode to whatever the checkpoint expects.

    obs_dim differs between the two modes, so a mismatch is not a subtle
    degradation -- it is a shape error three frames in, or worse, silently
    feeding a policy numbers that mean something else entirely.
    """
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("observation", "frenet") == "sensor":
        cfg.observation.mode = racing.ObservationConfig.SENSOR
    return cfg


def load_policy(path, obs_dim):
    import torch
    from train_marl import policy_from_checkpoint

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = policy_from_checkpoint(ckpt)
    if ckpt["obs_dim"] != obs_dim:
        raise SystemExit(
            f"{path} was trained on obs_dim {ckpt['obs_dim']}, but this race "
            f"has {obs_dim}. The observation layout must match."
        )
    return net


def run_one(cfg, seed, policy_path=None, pace=0.86, deterministic=False):
    """One race. Returns the raw counters; the ratios are formed later.

    `deterministic` takes the mean action instead of sampling. Careful with it:
    the grid is fixed, the physics is deterministic and the engine draws no
    randomness during a race, so a deterministic policy produces THE SAME RACE
    for every seed. Averaging five of those is not a five-race average, it is
    one race reported five times with error bars of zero. Sampling is also the
    honest thing to measure -- a stochastic policy is what actually races.
    """
    cfg.seed = seed
    if policy_path:
        apply_checkpoint_observation(cfg, policy_path)
    env = racing.RaceEnv(cfg, 0)
    env.reset(seed)

    n = env.n_cars
    per_team = cfg.field.cars_per_team
    team_of = [c.team for c in env.cars]

    net = None
    field = None
    if policy_path:
        net = load_policy(policy_path, env.obs_dim)
        import torch
        # Seed torch too, or sampling makes every "seed" the same draw sequence
        # and the races differ for no reproducible reason.
        torch.manual_seed(seed)
    else:
        field = scripted.build_field(env, seed=seed, base_pace=pace)

    tow_steps = 0            # car-steps spent in a meaningful tow
    tow_teammate = 0         # ...of which the car ahead was a team mate
    battle_steps = 0         # car-steps within BATTLE_GAP_S of the car ahead
    car_steps = 0
    lead_changes = 0
    leader = None
    overtakes = 0
    intra_team_passes = 0
    contacts = 0
    retire_reasons = []

    steps = 0
    while not env.done and steps < 40000:
        if net is not None:
            import torch
            with torch.no_grad():
                act = net.act(torch.from_numpy(env.observe()),
                              deterministic=deterministic).numpy()
        else:
            act = scripted.drive(field, env)
        env.step(act.astype(np.float32))
        steps += 1

        for e in env.events:
            if e.name == "overtake":
                overtakes += 1
                if team_of[e.car] == team_of[e.other]:
                    intra_team_passes += 1
            elif e.name == "contact" and e.value > 0.25:
                contacts += 1
            elif e.name == "retire":
                retire_reasons.append(e.reason)

        for c in env.cars:
            if c.retired or c.finished:
                continue
            car_steps += 1
            if c.wake > TOW_WAKE and c.wake_source >= 0:
                tow_steps += 1
                if team_of[c.wake_source] == c.team:
                    tow_teammate += 1
            if 0.0 < c.gap_ahead < BATTLE_GAP_S:
                battle_steps += 1
            if c.position == 1 and leader != c.index:
                if leader is not None:
                    lead_changes += 1
                leader = c.index

    finishers = [c for c in env.cars if c.finished]
    times = sorted(c.finish_time for c in finishers)
    # How far apart the team mates ended up. A team running as a pair finishes
    # closer together than two cars that happen to share a colour.
    gaps = []
    for t in range(cfg.field.n_teams):
        mates = [c.position for c in env.cars if c.team == t]
        if len(mates) > 1:
            gaps.append(max(mates) - min(mates))

    # With `per_team` cars on a team and `n` on the grid, any given other car is
    # a team mate this often. Everything team-flavoured is reported against it.
    chance = (per_team - 1) / (n - 1) if n > 1 else 0.0

    return dict(
        seed=seed,
        steps=steps,
        n_cars=n,
        chance=chance,
        winner_time=times[0] if times else float("nan"),
        last_time=times[-1] if times else float("nan"),
        spread=(times[-1] - times[0]) if len(times) > 1 else float("nan"),
        best_lap=min((c.best_lap_time for c in env.cars if c.best_lap_time > 0),
                     default=float("nan")),
        n_finished=len(finishers),
        n_retired=sum(1 for c in env.cars if c.retired),
        retire_reasons=retire_reasons,
        mean_damage=float(np.mean([c.damage for c in env.cars])),
        overtakes=overtakes,
        intra_team_passes=intra_team_passes,
        lead_changes=lead_changes,
        contacts=contacts,
        tow_steps=tow_steps,
        tow_teammate=tow_teammate,
        battle_steps=battle_steps,
        car_steps=car_steps,
        team_pos_gap=float(np.mean(gaps)) if gaps else float("nan"),
    )


def aggregate(runs):
    """Pool the raw counters across seeds, then form the ratios.

    Pooled rather than averaged per race: a ratio of means is the right estimate
    here, and averaging per-race ratios would weight a race where almost nobody
    got a tow the same as one where everybody did.
    """
    def s(key):
        return sum(r[key] for r in runs)

    def m(key):
        vals = [r[key] for r in runs if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    chance = runs[0]["chance"]
    tow_share = s("tow_teammate") / s("tow_steps") if s("tow_steps") else 0.0
    pass_share = (s("intra_team_passes") / s("overtakes")
                  if s("overtakes") else 0.0)
    reasons = {}
    for r in runs:
        for x in r["retire_reasons"]:
            reasons[x] = reasons.get(x, 0) + 1

    return dict(
        races=len(runs),
        chance=chance,
        winner_time=m("winner_time"),
        last_time=m("last_time"),
        spread=m("spread"),
        best_lap=m("best_lap"),
        finished=m("n_finished"),
        retired=m("n_retired"),
        retire_reasons=reasons,
        mean_damage=m("mean_damage"),
        overtakes=m("overtakes"),
        lead_changes=m("lead_changes"),
        contacts=m("contacts"),
        battle_frac=s("battle_steps") / s("car_steps") if s("car_steps") else 0.0,
        tow_frac=s("tow_steps") / s("car_steps") if s("car_steps") else 0.0,
        tow_teammate_share=tow_share,
        tow_teammate_ratio=tow_share / chance if chance else float("nan"),
        intra_team_pass_share=pass_share,
        intra_team_pass_ratio=pass_share / chance if chance else float("nan"),
        team_pos_gap=m("team_pos_gap"),
    )


def report(a, label):
    print(f"\n=== {label} ===  ({a['races']} races, {a['finished']:.1f} of "
          f"{a['races'] and ''}{'':<0}{a['finished'] + a['retired']:.0f} finishing)")

    print("\n  racecraft")
    print(f"    winner                {a['winner_time']:8.1f} s")
    print(f"    best lap              {a['best_lap']:8.1f} s")
    print(f"    retired               {a['retired']:8.2f} per race   "
          f"{a['retire_reasons'] or '-'}")
    print(f"    mean damage           {a['mean_damage'] * 100:7.1f} %")

    print("\n  watchability")
    print(f"    overtakes             {a['overtakes']:8.1f} per race")
    print(f"    lead changes          {a['lead_changes']:8.1f} per race")
    print(f"    time in a battle      {a['battle_frac'] * 100:7.1f} %   "
          f"(< {BATTLE_GAP_S:g} s to the car ahead)")
    print(f"    field spread          {a['spread']:8.1f} s   first to last")
    print(f"    notable contacts      {a['contacts']:8.1f} per race")

    print("\n  team tactics        (1.00x = chance, i.e. no team behaviour)")
    print(f"    time in a tow         {a['tow_frac'] * 100:7.1f} %")
    print(f"    ...behind a team mate {a['tow_teammate_share'] * 100:7.1f} %   "
          f"{a['tow_teammate_ratio']:5.2f}x chance ({a['chance'] * 100:.1f}%)")
    print(f"    passes on a team mate {a['intra_team_pass_share'] * 100:7.1f} %   "
          f"{a['intra_team_pass_ratio']:5.2f}x chance")
    print(f"    team finishing gap    {a['team_pos_gap']:8.2f} places   "
          f"lower is a team running as a pair")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", default=os.path.join(REPO, "data", "tracks", "bahrain.json"))
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--cars-per-team", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5, help="races to average over")
    ap.add_argument("--seed0", type=int, default=1, help="first seed")
    ap.add_argument("--policy", default=None,
                    help="checkpoint to evaluate; scripted drivers if omitted")
    ap.add_argument("--pace", type=float, default=0.86,
                    help="scripted driver pace, ignored with --policy")
    ap.add_argument("--team-weight", type=float, default=0.5)
    ap.add_argument("--deterministic", action="store_true",
                    help="take the mean action. Note every seed then produces "
                         "the identical race, so use --seeds 1 with it")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None, help="append the result here")
    args = ap.parse_args()

    cfg = build_config(args)
    label = args.label or (os.path.basename(args.policy) if args.policy
                           else f"scripted pace={args.pace:g}")

    runs = []
    for k in range(args.seeds):
        seed = args.seed0 + k
        r = run_one(cfg, seed, args.policy, args.pace, args.deterministic)
        runs.append(r)
        print(f"  seed {seed}: {r['steps']:6d} steps  "
              f"{r['n_finished']:2d} finished  {r['n_retired']:2d} out  "
              f"{r['overtakes']:3d} overtakes")

    # If every race came out identical the seeds did nothing, and an average
    # over them is one race wearing a disguise. Say so rather than reporting a
    # confident-looking mean over n=1.
    if len(runs) > 1:
        key = ("steps", "overtakes", "n_retired", "winner_time")
        if all(tuple(r[k] for k in key) == tuple(runs[0][k] for k in key)
               for r in runs):
            print(f"\n  !! all {len(runs)} races were identical, so this is "
                  f"really n=1.\n     The grid is fixed and the engine draws no "
                  f"randomness during a race,\n     so a deterministic policy "
                  f"races the same way every time. Drop\n     --deterministic, "
                  f"or accept that the seeds are decorative here.")

    a = aggregate(runs)
    report(a, label)

    if args.json:
        blob = {"label": label, "policy": args.policy, "laps": args.laps,
                "teams": args.teams, "cars_per_team": args.cars_per_team,
                "team_weight": args.team_weight, "metrics": a}
        existing = []
        if os.path.exists(args.json):
            with open(args.json, encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(blob)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=1)
        print(f"\nappended to {args.json}")


if __name__ == "__main__":
    main()
