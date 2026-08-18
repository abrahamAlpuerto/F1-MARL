"""Multi-agent PPO: every car is an agent, all of them share one policy.

    python examples/train_marl.py --envs 256 --threads 8 --iters 400

Parameter sharing plus self-play. One network drives every car in every race,
and it improves by racing copies of itself. That is the cheapest arrangement
that produces racecraft, and it works here because the cars are identical --
what makes two cars behave differently is not their weights, it is what they can
see: their own speed, where the road goes, where the other cars are, and which
of those are team mates.

That last one is the whole trick. The observation carries a per-neighbour
"is this a team mate" flag, and the reward is mixed with the team's
(RewardConfig::team_weight). A shared policy can therefore still learn to treat
a team mate differently from a rival, without ever needing a separate network
per team.

Training happens in two phases, because they are genuinely different problems:

    phase 1  drive       one car, empty circuit, short episodes, terminate on
                         leaving the track. Learns the car and the corners.
    phase 2  race        the full field, full race distance, aero and contact
                         on, cars rejoin instead of being deleted.

Skipping phase 1 does eventually work and takes far longer: a policy that
cannot get round Turn 1 learns nothing about racecraft, because it never
survives long enough to be near anyone.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
import racing  # noqa: E402


# --- the network -----------------------------------------------------------

class Policy(nn.Module):
    """Gaussian policy over (steer, throttle), with a value head.

    Small on purpose. The observation is already the useful features -- the
    curvature ahead, the neighbours, the state of the air -- rather than pixels,
    so there is nothing here for a large network to discover, and the simulator
    can produce samples far faster than a big model can consume them.
    """

    arch = "mlp"

    def __init__(self, obs_dim, hidden=256, n_neighbours=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        # State-independent log-std, which is standard for continuous control
        # and much better behaved early on than a learned per-state one: a
        # network that can shrink its own exploration before it can drive will
        # do exactly that and then never find anything.
        self.log_std = nn.Parameter(torch.full((2,), -0.7))

        # Small final layers, so the initial policy is near-zero action rather
        # than full lock. A random policy that starts by spinning the car
        # spends its first million samples learning not to.
        for layer in (self.mu, self.v):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, obs):
        h = self.body(obs)
        return self.mu(h), self.v(h).squeeze(-1)

    def dist(self, obs):
        mu, value = self(obs)
        return torch.distributions.Normal(mu, self.log_std.exp()), value

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mu, _ = self(obs)
        if deterministic:
            return torch.tanh(mu)
        d = torch.distributions.Normal(mu, self.log_std.exp())
        return torch.tanh(d.sample())


class SetPolicy(nn.Module):
    """Policy that reads the other cars as a SET rather than as a fixed list.

    The engine hands over a block of ego features -- own dynamics, the road
    ahead, race context, the state of the air -- followed by `n_neighbours`
    slots of five numbers each: along-track gap, lateral offset, closing speed,
    a team-mate flag, and a flag saying whether anybody is in the slot at all.
    The slots are filled nearest-first by ABSOLUTE gap and zero-padded.

    Feeding that straight to an MLP has two problems, and both are about the
    sorting rather than the numbers.

    First, slot assignment is discontinuous. A car two metres ahead and a car
    three metres behind occupy slots 0 and 1; as they jostle, the one behind
    creeps to two and a half metres and the two SWAP SLOTS. Nothing physical
    happened, but every weight reading slot 0 now sees a different car. The MLP
    has to spend capacity learning to be robust to a permutation that carries no
    information at all.

    Second, it is hard-capped. In a pack of six, the fifth and sixth cars are
    simply invisible however close they are.

    A set encoder fixes both by construction: every neighbour goes through the
    same small network and they are pooled by attention with the ego state as
    the query, so the result cannot depend on the order they arrived in. The
    ordering the engine happens to use stops mattering, and the cap becomes a
    budget rather than a blind spot -- raise `RaceConfig::n_neighbours` and this
    reads the extra cars with no new parameters.

    Deliberately still small, and matched to the MLP on DEPTH: one layer to
    encode the ego features and one trunk layer before the heads, the same two
    the baseline has. That matters because the point is a better-shaped
    inductive bias, not more capacity -- if this wins it should be because of
    the structure. It still carries about 1.4x the parameters (the attention
    query and the per-slot encoder are not free), which is the honest cost and
    is asserted in the tests so it cannot quietly grow.
    """

    arch = "set"
    N_FEAT = 5  # gap, lateral, closing speed, team mate, occupancy

    def __init__(self, obs_dim, hidden=256, n_neighbours=4, embed=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_neighbours = n_neighbours
        self.embed_dim = embed
        self.ego_dim = obs_dim - n_neighbours * self.N_FEAT
        if self.ego_dim <= 0:
            raise ValueError(
                f"obs_dim {obs_dim} is too small for {n_neighbours} neighbour "
                f"slots of {self.N_FEAT}")

        self.ego = nn.Sequential(nn.Linear(self.ego_dim, hidden), nn.Tanh())
        # Shared across slots. This is what makes the encoder permutation
        # invariant, and what lets it read a neighbour count it never saw.
        self.slot = nn.Sequential(
            nn.Linear(self.N_FEAT, embed), nn.Tanh(),
            nn.Linear(embed, embed), nn.Tanh(),
        )
        self.query = nn.Linear(hidden, embed)

        # One layer, not two. With `ego` above that gives the same depth as the
        # baseline MLP; a second trunk layer here made this three layers deep
        # against the MLP's two and 2.2x its size, which would have made any
        # improvement impossible to attribute.
        self.body = nn.Sequential(
            nn.Linear(hidden + embed, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.7))

        for layer in (self.mu, self.v):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def encode(self, obs):
        ego_raw = obs[..., :self.ego_dim]
        slots = obs[..., self.ego_dim:].reshape(
            *obs.shape[:-1], self.n_neighbours, self.N_FEAT)

        h_ego = self.ego(ego_raw)
        h_slot = self.slot(slots)                     # [..., n, embed]

        # Occupancy is the last feature. An empty slot must not be attended to
        # at all: a softmax over a padded slot hands it a share of the weight
        # and lets zeros vote on what the car should do.
        mask = slots[..., -1] > 0.5                   # [..., n]

        q = self.query(h_ego).unsqueeze(-2)           # [..., 1, embed]
        logits = (h_slot * q).sum(-1) / (self.embed_dim ** 0.5)
        logits = logits.masked_fill(~mask, float("-inf"))

        # A car on its own has no neighbours at all, and a softmax over nothing
        # is NaN -- which would silently poison the whole batch. Fall back to an
        # all-zero summary, the honest encoding of "there is nobody here".
        any_near = mask.any(-1, keepdim=True)
        weights = torch.softmax(logits.masked_fill(~any_near, 0.0), dim=-1)
        weights = weights * any_near
        pooled = (weights.unsqueeze(-1) * h_slot).sum(-2)

        return self.body(torch.cat([h_ego, pooled], dim=-1))

    def forward(self, obs):
        h = self.encode(obs)
        return self.mu(h), self.v(h).squeeze(-1)

    def dist(self, obs):
        mu, value = self(obs)
        return torch.distributions.Normal(mu, self.log_std.exp()), value

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mu, _ = self(obs)
        if deterministic:
            return torch.tanh(mu)
        d = torch.distributions.Normal(mu, self.log_std.exp())
        return torch.tanh(d.sample())


ARCHITECTURES = {"mlp": Policy, "set": SetPolicy}


def build_policy(arch, obs_dim, hidden=256, n_neighbours=4):
    if arch not in ARCHITECTURES:
        raise SystemExit(
            f"unknown architecture {arch!r}; have {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](obs_dim, hidden=hidden, n_neighbours=n_neighbours)


def policy_from_checkpoint(ckpt):
    """Rebuild the network a checkpoint describes.

    Checkpoints written before there was more than one architecture carry no
    `arch` key. Those are all the plain MLP.
    """
    net = build_policy(ckpt.get("arch", "mlp"), ckpt["obs_dim"],
                       hidden=ckpt.get("hidden", 256),
                       n_neighbours=ckpt.get("n_neighbours", 4))
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


# --- configuration ---------------------------------------------------------

def drive_phase_config(args):
    """Phase 1: learn to drive. One car, no traffic."""
    cfg = racing.EnvConfig()
    cfg.track.path = args.track
    cfg.seed = args.seed
    cfg.field.n_teams = 1
    cfg.field.cars_per_team = 1
    # Short episodes from a random point on the circuit. Without the random
    # start a policy only ever sees the corners it survives to, so early on the
    # second half of the lap generates no experience at all.
    cfg.track.randomize_start = True
    cfg.track.episode_distance = 1200.0
    cfg.reward.terminate_off_track = True
    cfg.reward.position_weight = 0.0   # nobody to race
    cfg.reward.finish_weight = 0.0
    cfg.reward.team_weight = 0.0
    cfg.aero.enabled = False
    cfg.contact.enabled = False
    if args.retire_penalty is not None:
        cfg.reward.retire_penalty = args.retire_penalty
    return cfg


def race_phase_config(args):
    """Phase 2: learn to race. The full field."""
    cfg = racing.EnvConfig()
    cfg.track.path = args.track
    cfg.seed = args.seed
    cfg.field.n_teams = args.teams
    cfg.field.cars_per_team = args.cars_per_team
    cfg.track.randomize_start = True
    cfg.track.episode_distance = args.episode_distance
    # Still terminating on a trip off the circuit: during training that is a
    # much stronger signal than letting a car plough through the desert and
    # rejoin, and it stops the policy learning that cutting a corner is free.
    cfg.reward.terminate_off_track = True
    cfg.reward.team_weight = args.team_weight
    cfg.reward.position_weight = args.position_weight
    if args.retire_penalty is not None:
        cfg.reward.retire_penalty = args.retire_penalty
    return cfg


@dataclass
class Rollout:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor


# --- PPO -------------------------------------------------------------------

def pick_device(name):
    """Resolve --device. `auto` takes the GPU when there is one."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda, but torch reports no CUDA. This is almost always a "
            "CPU-only torch build: check torch.__version__ for a '+cpu' suffix "
            "and reinstall from the cu### index if so."
        )
    return torch.device(name)


class Trainer:
    def __init__(self, cfg, args, net=None):
        self.env = racing.VecRaceEnv(cfg, args.envs, args.threads)
        self.n_envs = args.envs
        self.n_cars = self.env.n_cars
        self.obs_dim = self.env.obs_dim
        # Every car is a sample. A batch of 256 races with 8 cars each is 2048
        # transitions per step, which is why this trains on CPU at all.
        self.n_agents = self.n_envs * self.n_cars

        # The simulator is C++ on the CPU and stays there; what moves is the
        # network. Which way round that is worth doing is not obvious and
        # depends on the batch: the rollout crosses the boundary once per step
        # with a small tensor, while the update is 4 epochs of large
        # minibatches and is where a GPU actually earns its place. Both paths
        # are identical apart from the device, so the two can be timed against
        # each other rather than argued about.
        self.device = pick_device(args.device)

        self.net = net if net is not None else build_policy(
            args.arch, self.obs_dim, args.hidden, cfg.race.n_neighbours)
        self.net.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.args = args
        self.gamma = cfg.reward.gamma
        self.lam = 0.95

        self._obs = np.zeros((self.n_envs, self.n_cars, self.obs_dim), np.float32)
        self._rew = np.zeros((self.n_envs, self.n_cars), np.float32)
        self._done = np.zeros(self.n_envs, np.uint8)
        self.env.reset_all(0)
        self.env.observe(self._obs)

    def collect(self, horizon):
        dev = self.device
        obs_buf = torch.zeros(horizon, self.n_agents, self.obs_dim, device=dev)
        act_buf = torch.zeros(horizon, self.n_agents, 2, device=dev)
        logp_buf = torch.zeros(horizon, self.n_agents, device=dev)
        rew_buf = torch.zeros(horizon, self.n_agents, device=dev)
        val_buf = torch.zeros(horizon, self.n_agents, device=dev)
        # `done` is per race, but a transition is per car, so it is broadcast
        # across the cars in that race. Every car in a race starts and ends
        # together, which is what makes that legitimate.
        done_buf = torch.zeros(horizon, self.n_agents, device=dev)

        stats = {"speed": 0.0, "reward": 0.0}
        for t in range(horizon):
            obs = torch.from_numpy(
                self._obs.reshape(self.n_agents, self.obs_dim)).to(dev)
            with torch.no_grad():
                d, value = self.net.dist(obs)
                raw = d.sample()
                logp = d.log_prob(raw).sum(-1)
            # tanh keeps the action in range without the log-prob correction a
            # proper squashed Gaussian would need. The environment clamps
            # anyway, so the untransformed sample is what the ratio is computed
            # against and the objective stays consistent.
            act = torch.tanh(raw)

            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, raw, logp, value

            a = act.cpu().numpy().reshape(self.n_envs, self.n_cars, 2)
            self.env.step(np.ascontiguousarray(a), self._obs, self._rew, self._done)

            rew_buf[t] = torch.from_numpy(self._rew.reshape(-1)).to(dev)
            done_buf[t] = torch.from_numpy(
                np.repeat(self._done, self.n_cars).astype(np.float32)).to(dev)
            stats["reward"] += float(self._rew.mean())

        with torch.no_grad():
            last_obs = torch.from_numpy(
                self._obs.reshape(self.n_agents, self.obs_dim)).to(dev)
            _, last_val = self.net(last_obs)

        # GAE.
        adv = torch.zeros_like(rew_buf)
        gae = torch.zeros(self.n_agents, device=dev)
        for t in reversed(range(horizon)):
            next_val = last_val if t == horizon - 1 else val_buf[t + 1]
            mask = 1.0 - done_buf[t]
            delta = rew_buf[t] + self.gamma * next_val * mask - val_buf[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        ret = adv + val_buf

        def flat(x, d):
            return x.reshape(-1, d) if d else x.reshape(-1)

        adv_f = flat(adv, 0)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        stats["reward"] /= horizon
        return Rollout(flat(obs_buf, self.obs_dim), flat(act_buf, 2),
                       flat(logp_buf, 0), adv_f, flat(ret, 0)), stats

    def update(self, roll):
        n = roll.obs.shape[0]
        idx = np.arange(n)
        clip = self.args.clip
        losses = []
        for _ in range(self.args.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.args.minibatch):
                b = torch.from_numpy(
                    idx[start:start + self.args.minibatch]).to(self.device)
                d, value = self.net.dist(roll.obs[b])
                logp = d.log_prob(roll.act[b]).sum(-1)
                ratio = (logp - roll.logp[b]).exp()

                a = roll.adv[b]
                pg = -torch.min(ratio * a,
                                ratio.clamp(1 - clip, 1 + clip) * a).mean()
                vf = ((value - roll.ret[b]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + self.args.vf_coef * vf - self.args.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                losses.append(loss.detach().item())
        return float(np.mean(losses))

    def evaluate(self, episodes=1):
        """Race the current policy and report what the field actually did."""
        cfg_states = self.env.states()
        speed = float(np.hypot(cfg_states[..., 4], cfg_states[..., 5]).mean())
        return {"speed_kph": speed * 3.6}


def run_phase(name, cfg, args, net, iters):
    trainer = Trainer(cfg, args, net)
    print(f"\n=== {name}: {trainer.n_envs} races x {trainer.n_cars} cars, "
          f"obs_dim {trainer.obs_dim} ===")
    print(f"{'iter':>5} {'reward':>9} {'loss':>9} {'kph':>8} {'samples/s':>10}")

    for it in range(iters):
        t0 = time.perf_counter()
        roll, stats = trainer.collect(args.horizon)
        loss = trainer.update(roll)
        dt = time.perf_counter() - t0
        sps = args.horizon * trainer.n_agents / dt
        ev = trainer.evaluate()
        if it % args.log_every == 0 or it == iters - 1:
            print(f"{it:>5} {stats['reward']:>9.3f} {loss:>9.3f} "
                  f"{ev['speed_kph']:>8.1f} {sps:>10.0f}")
    return trainer.net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", default=os.path.join(REPO, "data", "tracks", "bahrain.json"))
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--cars-per-team", type=int, default=2)
    ap.add_argument("--drive-iters", type=int, default=150,
                    help="phase 1 iterations; 0 skips straight to racing")
    ap.add_argument("--iters", type=int, default=250, help="phase 2 iterations")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="where the network lives; the simulator is always CPU")
    ap.add_argument("--arch", default="mlp", choices=sorted(ARCHITECTURES),
                    help="mlp is the flat baseline; set reads neighbours as a set")
    ap.add_argument("--retire-penalty", type=float, default=None,
                    help="override RewardConfig::retire_penalty for both phases")
    ap.add_argument("--team-weight", type=float, default=0.5)
    ap.add_argument("--position-weight", type=float, default=1.0)
    ap.add_argument("--episode-distance", type=float, default=3000.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "latest.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net = None
    if args.drive_iters > 0:
        net = run_phase("phase 1: drive", drive_phase_config(args), args, None,
                        args.drive_iters)

    race_cfg = race_phase_config(args)
    if net is None:
        net = Policy(
            racing.VecRaceEnv(race_cfg, 1, 1).obs_dim, args.hidden)
    net = run_phase("phase 2: race", race_cfg, args, net, args.iters)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    net_cpu = net.to("cpu")
    # What the net IS, rather than what its first layer happens to look like:
    # with a set encoder `body[0].in_features` is the trunk width, not the
    # observation, and a loader that inferred obs_dim from it would build the
    # wrong network and fail with a shape error three frames into a race.
    torch.save({"model": net_cpu.state_dict(),
                "arch": type(net).arch,
                "obs_dim": net.obs_dim,
                "hidden": args.hidden,
                "n_neighbours": race_cfg.race.n_neighbours,
                "config": race_cfg.to_json_string()}, args.out)
    print(f"\nsaved {args.out}")
    print("watch it race:")
    print(f"    python examples/race.py --policy {args.out}")


if __name__ == "__main__":
    main()
