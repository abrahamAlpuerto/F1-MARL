"""The policy networks, and specifically the claims the set encoder makes.

The whole argument for SetPolicy is that it is permutation invariant over the
neighbouring cars and that it ignores empty slots. Neither is obvious from
reading it -- both are properties of the masking and the pooling -- and both
would fail silently: a subtly broken mask still trains, just worse, and you
would spend a week blaming the reward.

Skipped entirely without torch, which is an optional dependency: the engine,
the scripted drivers and the feed do not need it.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "examples"))

torch = pytest.importorskip("torch", reason="training deps not installed")

from train_marl import (  # noqa: E402
    ARCHITECTURES, Policy, SetPolicy, build_policy, policy_from_checkpoint,
)

N_NEIGH = 4
N_FEAT = SetPolicy.N_FEAT
OBS_DIM = 51  # what the engine produces at the default lookahead


def make_obs(batch, occupied):
    """An observation with `occupied` of the neighbour slots holding a car."""
    torch.manual_seed(0)
    obs = torch.randn(batch, OBS_DIM)
    ego = OBS_DIM - N_NEIGH * N_FEAT
    for k in range(N_NEIGH):
        base = ego + k * N_FEAT
        obs[:, base + N_FEAT - 1] = 1.0 if k < occupied else 0.0
        if k >= occupied:
            obs[:, base:base + N_FEAT] = 0.0
    return obs


def swap_slots(obs, i, j):
    ego = OBS_DIM - N_NEIGH * N_FEAT
    out = obs.clone()
    a = slice(ego + i * N_FEAT, ego + (i + 1) * N_FEAT)
    b = slice(ego + j * N_FEAT, ego + (j + 1) * N_FEAT)
    out[:, a], out[:, b] = obs[:, b].clone(), obs[:, a].clone()
    return out


def test_set_policy_is_permutation_invariant():
    """The headline claim. Two cars swapping slots is not a change in the world.

    This is the failure mode the encoder exists to remove: under the engine's
    sort-by-absolute-gap, a car two metres ahead and one three metres behind
    swap slots as they jostle, and a flat MLP sees a completely different input.
    """
    net = SetPolicy(OBS_DIM, hidden=64, n_neighbours=N_NEIGH, embed=32).eval()
    obs = make_obs(8, occupied=4)

    with torch.no_grad():
        mu_a, v_a = net(obs)
        mu_b, v_b = net(swap_slots(obs, 0, 2))
        mu_c, v_c = net(swap_slots(swap_slots(obs, 1, 3), 0, 1))

    assert torch.allclose(mu_a, mu_b, atol=1e-6)
    assert torch.allclose(v_a, v_b, atol=1e-6)
    assert torch.allclose(mu_a, mu_c, atol=1e-6)
    assert torch.allclose(v_a, v_c, atol=1e-6)


def test_a_flat_mlp_is_not_permutation_invariant():
    """The control. If this ever passes, the test above proves nothing."""
    net = Policy(OBS_DIM, hidden=64).eval()
    obs = make_obs(8, occupied=4)
    with torch.no_grad():
        mu_a, _ = net(obs)
        mu_b, _ = net(swap_slots(obs, 0, 2))
    assert not torch.allclose(mu_a, mu_b, atol=1e-4)


def test_empty_slots_are_ignored():
    """Padding must not vote.

    A masked softmax that leaks would let whatever happens to be sitting in an
    unused slot influence the action -- and since padding is zeros, that reads
    as a phantom car exactly alongside, at identical speed.
    """
    net = SetPolicy(OBS_DIM, hidden=64, n_neighbours=N_NEIGH, embed=32).eval()
    obs = make_obs(8, occupied=2)

    polluted = obs.clone()
    ego = OBS_DIM - N_NEIGH * N_FEAT
    for k in (2, 3):  # the empty ones
        base = ego + k * N_FEAT
        polluted[:, base:base + N_FEAT - 1] = 99.0  # nonsense, but unoccupied

    with torch.no_grad():
        mu_a, v_a = net(obs)
        mu_b, v_b = net(polluted)

    assert torch.allclose(mu_a, mu_b, atol=1e-6)
    assert torch.allclose(v_a, v_b, atol=1e-6)


def test_a_car_on_its_own_does_not_produce_nan():
    """Softmax over an all-masked row is NaN, and one NaN poisons the batch.

    This happens for real: phase 1 of training is a single car on an empty
    circuit, so EVERY row has no neighbours.
    """
    net = SetPolicy(OBS_DIM, hidden=64, n_neighbours=N_NEIGH, embed=32).eval()
    obs = make_obs(8, occupied=0)

    with torch.no_grad():
        mu, v = net(obs)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(v).all()

    # And it must survive a backward pass, which is where a masked_fill of -inf
    # usually turns into a NaN gradient even when the forward looked fine.
    mu, v = net(obs)
    (mu.sum() + v.sum()).backward()
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_mixed_batch_of_lonely_and_crowded_cars():
    """The real case: a field where some cars are in traffic and some are not."""
    net = SetPolicy(OBS_DIM, hidden=64, n_neighbours=N_NEIGH, embed=32).eval()
    obs = torch.cat([make_obs(4, occupied=0), make_obs(4, occupied=3)], dim=0)
    with torch.no_grad():
        mu, v = net(obs)
    assert torch.isfinite(mu).all() and torch.isfinite(v).all()


def test_set_encoder_is_not_much_bigger_than_the_mlp():
    """A win has to come from the structure, not from smuggling in capacity.

    1.5x is the budget. The first version of this encoder had a two-layer trunk
    on top of the ego encoder, which made it three layers deep against the MLP's
    two and 2.2x its size -- at which point a better result would have proved
    nothing about set encoding.
    """
    mlp = sum(p.numel() for p in Policy(OBS_DIM, hidden=256).parameters())
    st = sum(p.numel() for p in
             SetPolicy(OBS_DIM, hidden=256, n_neighbours=N_NEIGH).parameters())
    assert st < 1.5 * mlp, f"set encoder {st} vs mlp {mlp}"


def test_both_architectures_are_the_same_depth():
    """Matched on depth, which is the comparison that makes the A/B fair."""
    def n_linear(module):
        return sum(1 for m in module.modules() if isinstance(m, torch.nn.Linear))

    mlp = Policy(OBS_DIM, hidden=256)
    st = SetPolicy(OBS_DIM, hidden=256, n_neighbours=N_NEIGH)
    # The ego path: trunk layers before the heads, ignoring the per-slot encoder
    # and the attention query, which sit off to the side.
    assert n_linear(mlp.body) == n_linear(st.ego) + n_linear(st.body)


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_checkpoints_round_trip(arch, tmp_path):
    net = build_policy(arch, OBS_DIM, hidden=64, n_neighbours=N_NEIGH)
    path = tmp_path / "ckpt.pt"
    torch.save({"model": net.state_dict(), "arch": arch, "obs_dim": OBS_DIM,
                "hidden": 64, "n_neighbours": N_NEIGH}, path)

    back = policy_from_checkpoint(
        torch.load(path, map_location="cpu", weights_only=False))
    assert type(back) is type(net)

    obs = make_obs(4, occupied=2)
    with torch.no_grad():
        assert torch.allclose(net(obs)[0], back(obs)[0], atol=1e-6)


def test_a_checkpoint_with_no_arch_key_loads_as_the_mlp():
    """Checkpoints written before there was a choice must keep working."""
    net = Policy(OBS_DIM, hidden=64)
    back = policy_from_checkpoint(
        {"model": net.state_dict(), "obs_dim": OBS_DIM, "hidden": 64})
    assert isinstance(back, Policy)


def test_both_architectures_expose_the_same_interface():
    """Trainer, race.py and evaluate.py all drive these through one interface."""
    obs = make_obs(4, occupied=2)
    for arch in ARCHITECTURES:
        net = build_policy(arch, OBS_DIM, hidden=64, n_neighbours=N_NEIGH)
        assert net.obs_dim == OBS_DIM
        assert type(net).arch == arch
        mu, v = net(obs)
        assert mu.shape == (4, 2) and v.shape == (4,)
        d, value = net.dist(obs)
        assert d.log_prob(d.sample()).sum(-1).shape == (4,)
        assert net.act(obs, deterministic=True).shape == (4, 2)
        assert value.shape == (4,)
