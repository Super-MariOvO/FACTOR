"""Unit tests for the TAC component."""

import torch

from factor.tac import (
    MCTarget,
    TACModule,
    TACReplayBuffer,
    TACValueHead,
    boundary_adjusted_values,
    compute_tac_advantages,
    select_checkpoints,
)


def test_value_head_shape():
    head = TACValueHead(hidden_size=64, mlp_hidden=1024)
    h = torch.randn(7, 64)
    v = head(h)
    assert v.shape == (7,)


def test_value_head_architecture():
    head = TACValueHead(hidden_size=64, mlp_hidden=1024)
    layers = list(head.net)
    assert isinstance(layers[0], torch.nn.Linear)
    assert layers[0].in_features == 64 and layers[0].out_features == 1024
    assert isinstance(layers[1], torch.nn.GELU)
    assert isinstance(layers[2], torch.nn.Linear)
    assert layers[2].out_features == 1


# ---------------------------------------------------------------------------
# Checkpoint selection (paper supplement: random from each half)
# ---------------------------------------------------------------------------


def test_checkpoint_selection_two_halves():
    # T >= 4: one checkpoint uniformly from each half.
    gen = torch.Generator().manual_seed(0)
    for T in (4, 5, 8, 17, 50):
        for _ in range(25):
            anchors = select_checkpoints(T, generator=gen)
            assert len(anchors) == 2
            half = T // 2
            assert 0 <= anchors[0] < half
            assert half <= anchors[1] < T


def test_checkpoint_selection_short_trajectories():
    # 1 < T < 4: one checkpoint from the midpoint.
    assert select_checkpoints(3) == [1]
    assert select_checkpoints(2) == [1]
    # Single-action and empty trajectories contribute none.
    assert select_checkpoints(1) == []
    assert select_checkpoints(0) == []


def test_checkpoint_selection_reproducible():
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = [select_checkpoints(9, generator=g1) for _ in range(10)]
    b = [select_checkpoints(9, generator=g2) for _ in range(10)]
    assert a == b


# ---------------------------------------------------------------------------
# Boundary values and one-step TD residuals (paper Eq. 3-5)
# ---------------------------------------------------------------------------


def test_boundary_adjusted_values():
    values = torch.tensor([10.0, 20.0, 30.0])
    v_tilde = boundary_adjusted_values(values, baseline=0.5)
    # V~(x_0) = b_i overrides the value head; V~(x_T) = 0.
    assert v_tilde.tolist() == [0.5, 20.0, 30.0, 0.0]


def test_tac_advantages_formula():
    # values[0] must be ignored (replaced by the baseline).
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([100.0, 0.25, -0.75])
    baseline = 0.375
    adv = compute_tac_advantages(rewards, values, baseline)
    # A*_0 = 0 + 0.25 - 0.375, A*_1 = 0 - 0.75 - 0.25, A*_2 = 1 + 0 - (-0.75)
    assert torch.allclose(adv, torch.tensor([-0.125, -1.0, 1.75]))


def test_tac_telescoping_exact():
    """Paper Eq. 5: sum_t A*_t == A^seq exactly (bit-exact on dyadic inputs)."""
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    values = torch.tensor([0.5, 0.25, -0.75, 0.125], dtype=torch.float64)
    baseline = 0.375  # leave-one-out baseline b_i (exactly representable)
    a_seq = 1.0 - baseline  # G_i - b_i
    adv = compute_tac_advantages(rewards, values, baseline)
    assert adv.sum().item() == a_seq  # exact equality, no tolerance


def test_tac_telescoping_exact_multistep():
    # Longer trajectory, mixed signs, still bit-exact with dyadic values.
    rewards = torch.tensor([0.0, 0.5, 0.0, 0.0, 1.0], dtype=torch.float64)
    values = torch.tensor([1.5, -0.25, 0.75, 2.0, -1.0], dtype=torch.float64)
    baseline = -0.5
    a_seq = 1.5 - (-0.5)  # sum r - b
    adv = compute_tac_advantages(rewards, values, baseline)
    assert adv.sum().item() == a_seq


def test_tac_telescoping_random_inputs():
    # Arbitrary float64 inputs: identity holds to floating-point rounding.
    torch.manual_seed(0)
    for _ in range(20):
        T = int(torch.randint(1, 30, (1,)).item())
        rewards = torch.zeros(T, dtype=torch.float64)
        rewards[-1] = torch.randn(())
        values = torch.randn(T, dtype=torch.float64)
        baseline = float(torch.randn(()))
        adv = compute_tac_advantages(rewards, values, baseline)
        a_seq = rewards.sum().item() - baseline
        assert abs(adv.sum().item() - a_seq) < 1e-10


def test_tac_telescoping_single_action():
    rewards = torch.tensor([1.0])
    values = torch.tensor([7.0])  # overridden by the baseline
    adv = compute_tac_advantages(rewards, values, baseline=0.25)
    assert torch.allclose(adv, torch.tensor([0.75]))  # G - b


# ---------------------------------------------------------------------------
# Replay buffer / training
# ---------------------------------------------------------------------------


def test_replay_buffer_window():
    buf = TACReplayBuffer(window_steps=3)
    for k in range(5):
        buf.add_step([MCTarget(hidden_state=torch.zeros(4), target=float(k))])
    pooled = buf.sample_all()
    assert len(pooled) == 3
    assert [t.target for t in pooled] == [2.0, 3.0, 4.0]


def test_tac_training_step_and_stopgrad():
    torch.manual_seed(0)
    tac = TACModule(hidden_size=16, mlp_hidden=32, buffer_window=2)
    targets = [
        MCTarget(hidden_state=torch.randn(16), target=1.0) for _ in range(8)
    ]
    stats = tac.train_step(targets)
    assert stats["critic_loss"] >= 0.0
    assert stats["num_targets"] == 8

    # Stop-gradient: gradients reach the head parameters but never the
    # caller's input tensor (the backbone features are detached inside).
    h = torch.randn(4, 16, requires_grad=True)
    v = tac.value(h)
    assert v.requires_grad
    v.sum().backward()
    assert h.grad is None
    grads = [p.grad for p in tac.online_head.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_polyak_update():
    tac = TACModule(hidden_size=8, mlp_hidden=16, ema_decay=0.995)
    before = [p.clone() for p in tac.target_head.parameters()]
    with torch.no_grad():
        for p in tac.online_head.parameters():
            p.add_(1.0)
    tac.polyak_update()
    for tgt, old in zip(tac.target_head.parameters(), before):
        # target should have moved 0.5% toward the new online params
        assert torch.allclose(tgt, old * 0.995 + (old + 1.0) * 0.005,
                              atol=1e-6)


def test_mc_target_generation():
    # B=2 checkpoints x M=4 continuations of up to 15 turns.
    tac = TACModule(hidden_size=4, mlp_hidden=8, num_checkpoints=2,
                    continuations_per_checkpoint=4, continuation_horizon=15,
                    seed=0)
    h = torch.tensor([[0., 0., 0., 0.], [1., 1., 1., 1.], [3., 3., 3., 3.]])
    calls = []

    def continuation_fn(prefix, m, horizon):
        calls.append((m, horizon))
        return [1.0, 3.0]  # mean = 2.0

    targets = tac.build_targets_for_trajectory(
        trajectory_id=0,
        hidden_states=h,
        action_boundaries=[0, 1, 2],
        prefix_tokens_fn=lambda b: torch.arange(5),
        continuation_fn=continuation_fn,
        realized_return=0.0,
    )
    # T = 3 < 4: exactly one checkpoint from the midpoint (action 1).
    assert len(targets) == 1
    assert abs(targets[0].target - 2.0) < 1e-6
    assert torch.allclose(targets[0].hidden_state, h[1])
    # Continuation sampler receives (M, horizon) with paper semantics.
    assert calls == [(4, 15)]


def test_mc_target_generation_two_checkpoints():
    tac = TACModule(hidden_size=4, mlp_hidden=8, num_checkpoints=2,
                    continuations_per_checkpoint=4, continuation_horizon=15,
                    seed=1)
    h = torch.randn(8, 4)
    targets = tac.build_targets_for_trajectory(
        trajectory_id=0,
        hidden_states=h,
        action_boundaries=list(range(8)),
        prefix_tokens_fn=lambda b: torch.arange(5),
        continuation_fn=lambda prefix, m, horizon: [0.5],
        realized_return=0.0,
    )
    # T = 8 >= 4: two checkpoints, one from each half.
    assert len(targets) == 2
    assert all(abs(t.target - 0.5) < 1e-6 for t in targets)
