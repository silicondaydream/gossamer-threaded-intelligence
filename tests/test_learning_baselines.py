"""
Tests for gossamer.learning.baselines (the MAPPO comparators).

Covers the policy forward contracts, the action->control and reward wiring,
a trainability check (the learned-weight policy can fit the DMB schedule, which
is the basis for the paper's "DMB matches MAPPO" claim), and a smoke test that
the policies integrate with the reference MAPPO update.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gossamer.algorithms.coordination.dmb import DMBParams, density_modulated_weights
from gossamer.learning.baselines import (
    LearnedBoidsWeights,
    RelaySelectionPolicy,
    boids_features,
    boids_reward,
    learned_boids_step,
    relay_features,
    relay_reward,
    select_relays_learned,
)


def test_learned_boids_weights_forward_positive():
    policy = LearnedBoidsWeights()
    feats = torch.randn(10, 3)
    w = policy(feats)
    assert w.shape == (10, 3)
    assert torch.all(w >= 0)  # softplus output


def test_boids_features_shape_and_density_column():
    from gossamer.algorithms.coordination.dmb import local_density
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [100.0, 0, 0]])
    vel = np.zeros_like(pos)
    feats = boids_features(pos, vel, radius=2.0)
    assert feats.shape == (3, 3)
    assert np.allclose(feats[:, 0], local_density(pos, 2.0))


def test_learned_boids_step_shapes_and_clamp():
    rng = np.random.default_rng(0)
    pos = rng.normal(scale=10.0, size=(30, 3))
    vel = rng.normal(scale=1.0, size=(30, 3))
    policy = LearnedBoidsWeights()
    new_pos, new_vel, info = learned_boids_step(pos, vel, 0.1, policy,
                                                neighbor_radius=8.0, max_speed=5.0)
    assert new_pos.shape == pos.shape
    assert np.all(np.linalg.norm(new_vel, axis=1) <= 5.0 + 1e-6)
    assert set(info) == {"w_align", "w_coh", "w_sep"}


def test_relay_policy_and_selection():
    policy = RelaySelectionPolicy()
    feats = torch.randn(20, 3)
    logits = policy(feats)
    assert logits.shape == (20,)
    # fraction mode: top 10% of 20 -> 2 relays.
    mask = select_relays_learned(logits.detach().numpy(), fraction=0.1)
    assert mask.sum() == 2
    # threshold mode.
    lg = np.array([-1.0, 0.0, 2.0])
    assert list(select_relays_learned(lg, threshold=0.5)) == [False, False, True]


def test_relay_features_and_rewards():
    feats = relay_features(soc=[0.9, 0.2], degree=[4, 1], density=[10, 3])
    assert feats.shape == (2, 3)
    assert relay_reward(delta_freshness=5.0, energy=2.0, lam=0.5) == pytest.approx(4.0)
    assert boids_reward(psi=0.8, collision_rate=0.1, lam=2.0) == pytest.approx(0.6)


def test_learned_weights_can_fit_dmb_schedule():
    """The learned policy can represent DMB's hand-designed sigmoid weights."""
    torch.manual_seed(0)
    params = DMBParams(d_crit=12.0)
    d = np.linspace(0.0, 25.0, 512)
    target = np.stack(density_modulated_weights(d, params), axis=1)  # (512, 3)
    feats = np.stack([d, np.zeros_like(d), np.zeros_like(d)], axis=1)
    X = torch.as_tensor(feats, dtype=torch.float32)
    Y = torch.as_tensor(target, dtype=torch.float32)

    policy = LearnedBoidsWeights()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    init_loss = float(loss_fn(policy(X), Y).detach())
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(policy(X), Y)
        loss.backward()
        opt.step()
    final_loss = float(loss_fn(policy(X), Y).detach())
    assert final_loss < 0.25 * init_loss
    assert final_loss < 0.05   # learned weights match the schedule closely


def test_integrates_with_mappo_update():
    """A paper-shaped policy runs through the reference MAPPO update."""
    from gossamer.learning.mappo import MAPPOConfig, update
    from gossamer.learning.policy_base import GraphActorCritic, SharedMLPHead

    torch.manual_seed(0)
    T, N, obs_dim, act_dim = 4, 8, 6, 3
    policy = GraphActorCritic(
        actor_head=SharedMLPHead(obs_dim, act_dim),
        critic_head=SharedMLPHead(obs_dim, 1),
    )
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    batch = {
        "obs": torch.randn(T, N, obs_dim),
        "actions": torch.randn(T, N, act_dim),
        "log_probs": torch.randn(T, N),
        "values": torch.randn(T, N),
        "rewards": torch.randn(T, N),
        "dones": torch.zeros(T, N),
    }
    before = [p.detach().clone() for p in policy.parameters()]
    metrics = update(policy, opt, batch, MAPPOConfig(minibatch_size=16, update_epochs=2))
    assert all(np.isfinite(v) for v in metrics.values())
    after = list(policy.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))  # a step was taken
