"""Integration tests for the FACTOR trainer pipeline (CPU, toy policy)."""

import torch
import torch.nn as nn

from factor.training.config import FACTORConfig
from factor.training.trainer import FACTORTrainer, Trajectory, TrajectoryBatch


class ToyPolicy(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.lm = nn.Linear(hidden, vocab)
        self.config = type("C", (), {"hidden_size": hidden})()

    def forward(self, input_ids=None, attention_mask=None):
        h = self.emb(input_ids)
        return type("Out", (), {"logits": self.lm(h)})()


def _make_traj(tid: int, group: int, reward: float, T: int = 12):
    return Trajectory(
        input_ids=torch.randint(0, 32, (T,)),
        action_spans=[slice(2, 5), slice(7, 11)],
        boundary_hidden=torch.randn(2, 16),
        action_rewards=torch.tensor([0.0, reward]),
        old_logprobs=torch.randn(T),
        group_id=group,
        terminal_reward=reward,
        trajectory_id=tid,
    )


# ---------------------------------------------------------------------------
# Per-environment reward construction (paper supplement)
# ---------------------------------------------------------------------------


def test_construct_rewards_alfworld_leave_one_out():
    batch = TrajectoryBatch(trajectories=[
        _make_traj(i, group=0, reward=r) for i, r in enumerate([1, 1, 0, 0])
    ])
    specs = batch.construct_rewards("alfworld")
    # Group sum = 2; K = 4. b_i = (2 - G_i) / 3.
    for i, tr in enumerate(batch.trajectories):
        g = tr.terminal_reward
        b = (2.0 - g) / 3.0
        assert abs(specs[i].baseline - b) < 1e-9
        assert abs(specs[i].a_seq - (g - b)) < 1e-9
        assert abs(specs[i].terminal - g) < 1e-9  # raw success as r_{T-1}


def test_construct_rewards_webshop_standardization():
    scores = [0.5, 0.7, 0.9, 0.1]
    batch = TrajectoryBatch(trajectories=[
        _make_traj(i, group=0, reward=s) for i, s in enumerate(scores)
    ])
    specs = batch.construct_rewards("webshop")
    g = torch.tensor(scores, dtype=torch.float64)
    mu, sigma = g.mean(), g.std(unbiased=False)
    for i, s in enumerate(scores):
        expected = float((g[i] - mu) / (sigma + 1e-8))
        assert abs(specs[i].a_seq - expected) < 1e-9
        # WebShop: terminal reward IS A^seq and the baseline is 0.
        assert abs(specs[i].terminal - expected) < 1e-9
        assert specs[i].baseline == 0.0


def test_construct_rewards_webshop_degenerate_group():
    # sigma < 1e-6 -> A^seq = 0 for the whole group.
    batch = TrajectoryBatch(trajectories=[
        _make_traj(i, group=0, reward=0.42) for i in range(4)
    ])
    specs = batch.construct_rewards("webshop")
    assert all(s.a_seq == 0.0 for s in specs)


def test_construct_rewards_scienceworld_raw():
    batch = TrajectoryBatch(trajectories=[
        _make_traj(i, group=0, reward=r) for i, r in enumerate([1, 0, 1, 0])
    ])
    specs = batch.construct_rewards("scienceworld")
    for i, tr in enumerate(batch.trajectories):
        assert specs[i].a_seq == tr.terminal_reward   # no baseline
        assert specs[i].terminal == tr.terminal_reward
        assert specs[i].baseline == 0.0


# ---------------------------------------------------------------------------
# TAC action credits inside the trainer
# ---------------------------------------------------------------------------


def test_action_credits_telescope_after_warmup():
    """sum_t A*_t == A^seq for the trainer's post-warmup TD credits."""
    torch.manual_seed(0)
    trainer = FACTORTrainer(policy=ToyPolicy(), config=FACTORConfig(),
                            hidden_size=16)
    batch = TrajectoryBatch(trajectories=[
        _make_traj(0, group=0, reward=1.0),
        _make_traj(1, group=0, reward=0.0),
    ])
    specs = batch.construct_rewards("alfworld")
    credits = trainer.compute_action_credits(batch, specs, step=20)
    for c, spec in zip(credits, specs):
        assert abs(c.sum().item() - spec.a_seq) < 1e-5


def test_action_credits_warmup_uniform():
    # During warm-up the credit falls back to the uniform split A^seq / T.
    trainer = FACTORTrainer(policy=ToyPolicy(), config=FACTORConfig(),
                            hidden_size=16)
    batch = TrajectoryBatch(trajectories=[
        _make_traj(0, group=0, reward=1.0),
        _make_traj(1, group=0, reward=0.0),
    ])
    specs = batch.construct_rewards("alfworld")
    credits = trainer.compute_action_credits(batch, specs, step=5)
    for c, spec in zip(credits, specs):
        T = c.numel()
        assert torch.allclose(c, torch.full((T,), spec.a_seq / T), atol=1e-6)
        assert abs(c.sum().item() - spec.a_seq) < 1e-5


# ---------------------------------------------------------------------------
# End-to-end advantage pipeline
# ---------------------------------------------------------------------------


def test_end_to_end_advantage_shapes():
    torch.manual_seed(0)
    policy = ToyPolicy()
    cfg = FACTORConfig()
    trainer = FACTORTrainer(policy=policy, config=cfg, hidden_size=16)
    batch = TrajectoryBatch(trajectories=[
        _make_traj(i, group=i // 4, reward=float(i % 2)) for i in range(8)
    ])
    advs = trainer.compute_token_advantages(batch)
    assert len(advs) == 8
    for tr, a in zip(batch.trajectories, advs):
        assert a.shape == tr.old_logprobs.shape
        # Non-action tokens carry zero advantage.
        mask = torch.zeros_like(a, dtype=torch.bool)
        for s in tr.action_spans:
            mask[s] = True
        assert torch.all(a[~mask] == 0.0)


def test_teacher_allocation_mean_preserving_per_action():
    torch.manual_seed(1)
    policy = ToyPolicy()
    cfg = FACTORConfig()
    trainer = FACTORTrainer(policy=policy, config=cfg, hidden_size=16)
    trainer.step_idx = 20  # inside the active window 11..49
    batch = TrajectoryBatch(trajectories=[
        _make_traj(0, group=0, reward=1.0),
        _make_traj(1, group=0, reward=0.0),
    ])
    # Teacher logprobs that differ from the old policy on action tokens.
    teacher_lps = [tr.old_logprobs + torch.randn_like(tr.old_logprobs) * 0.5
                   for tr in batch.trajectories]
    specs = batch.construct_rewards("alfworld")
    credits = trainer.compute_action_credits(batch, specs, step=21)
    advs = trainer.compute_token_advantages(batch, teacher_lps)
    for tr, a, cred in zip(batch.trajectories, advs, credits):
        for t, s in enumerate(tr.action_spans):
            assert torch.isfinite(a[s]).all()
            # APM: token advantages average to the action credit A*_t.
            assert abs(a[s].mean().item() - cred[t].item()) < 1e-5


# ---------------------------------------------------------------------------
# PPO objective: action-mean reduction and the full Eq. 8 objective
# ---------------------------------------------------------------------------


def _ppo_fixture_batch():
    """Two trajectories with hand-controlled spans and logprobs (ratio = 1)."""
    tr0 = Trajectory(
        input_ids=torch.randint(0, 32, (7,)),
        action_spans=[slice(0, 2), slice(2, 4)],
        boundary_hidden=torch.randn(2, 16),
        action_rewards=torch.tensor([0.0, 1.0]),
        old_logprobs=torch.zeros(7),
        group_id=0,
        terminal_reward=1.0,
        trajectory_id=0,
    )
    tr1 = Trajectory(
        input_ids=torch.randint(0, 32, (5,)),
        action_spans=[slice(0, 3)],
        boundary_hidden=torch.randn(1, 16),
        action_rewards=torch.tensor([0.0]),
        old_logprobs=torch.zeros(5),
        group_id=0,
        terminal_reward=0.0,
        trajectory_id=1,
    )
    return TrajectoryBatch(trajectories=[tr0, tr1])


def test_ppo_action_mean_reduction():
    """At ratio q = 1 the action-mean loss equals -(1/sum T_i) sum_t A*_t."""
    trainer = FACTORTrainer(policy=ToyPolicy(), config=FACTORConfig(),
                            hidden_size=16)
    batch = _ppo_fixture_batch()
    # Differentiable stand-ins for the actor's gathered log-probs (q == 1).
    new_logprobs = [torch.zeros(7, requires_grad=True),
                    torch.zeros(5, requires_grad=True)]
    adv0 = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0])
    adv1 = torch.zeros(5)
    action_masks = [torch.tensor([1, 1, 1, 1, 0, 0, 0]),
                    torch.tensor([1, 1, 1, 0, 0])]
    non_action_masks = [torch.tensor([0, 0, 0, 0, 1, 1, 1]),
                        torch.tensor([0, 0, 0, 1, 1])]
    stats = trainer.ppo_update(
        batch, new_logprobs, [adv0, adv1], action_masks,
        non_action_masks=non_action_masks,
    )
    # L_RL^act = -(mean([1,1]) + mean([0.5,0.5]) + mean([0,0,0])) / 3 = -0.5
    assert abs(stats["loss_rl_act"] - (-0.5)) < 1e-6
    assert stats["num_actions"] == 3
    # alfworld, group [1, 0]: a_seq = [1, -1].
    # L_RL^non-act = -(1*3 + (-1)*2) / 5 = -0.2
    assert abs(stats["loss_rl_nonact"] - (-0.2)) < 1e-6
    assert stats["num_nonact_tokens"] == 5
    # No teacher logits -> SERL term skipped.
    assert stats["serl_kl"] == 0.0
    assert abs(stats["policy_loss"] - (-0.7)) < 1e-6


def test_ppo_action_mean_weights_actions_equally():
    """Action-mean: a 4-token action and a 1-token action weigh equally."""
    trainer = FACTORTrainer(policy=ToyPolicy(), config=FACTORConfig(),
                            hidden_size=16)
    tr = Trajectory(
        input_ids=torch.randint(0, 32, (5,)),
        action_spans=[slice(0, 4), slice(4, 5)],   # L = 4 and L = 1
        boundary_hidden=torch.randn(2, 16),
        action_rewards=torch.tensor([0.0, 1.0]),
        old_logprobs=torch.zeros(5),
        group_id=0,
        terminal_reward=1.0,
        trajectory_id=0,
    )
    batch = TrajectoryBatch(trajectories=[tr])
    adv = torch.tensor([0.25, 0.25, 0.25, 0.25, 1.0])  # means 0.25 and 1.0
    mask = torch.ones(5, dtype=torch.float32)
    non_mask = torch.zeros(5)
    stats = trainer.ppo_update(batch, [torch.zeros(5, requires_grad=True)],
                               [adv], [mask], non_action_masks=[non_mask])
    # -(0.25 + 1.0) / 2, NOT the token-mean -(0.25*4 + 1.0)/5 = -0.4.
    assert abs(stats["loss_rl_act"] - (-0.625)) < 1e-6


def test_ppo_full_objective_with_serl_kl():
    """L = L_RL^act + L_RL^non-act + lambda_k * L_act^SERL (Eq. 8)."""
    torch.manual_seed(0)
    trainer = FACTORTrainer(policy=ToyPolicy(), config=FACTORConfig(),
                            hidden_size=16)
    trainer.step_idx = 20  # lambda_20 = 1 - 20/50 = 0.6
    batch = _ppo_fixture_batch()
    new_logprobs = [torch.zeros(7, requires_grad=True),
                    torch.zeros(5, requires_grad=True)]
    advs = [torch.tensor([1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0]), torch.zeros(5)]
    action_masks = [torch.tensor([1, 1, 1, 1, 0, 0, 0]),
                    torch.tensor([1, 1, 1, 0, 0])]
    non_action_masks = [torch.tensor([0, 0, 0, 0, 1, 1, 1]),
                        torch.tensor([0, 0, 0, 1, 1])]
    teacher_logits = [torch.randn(7, 32), torch.randn(5, 32)]
    policy_logits = [torch.randn(7, 32, requires_grad=True),
                     torch.randn(5, 32, requires_grad=True)]

    stats = trainer.ppo_update(
        batch, new_logprobs, advs, action_masks,
        non_action_masks=non_action_masks,
        teacher_logits=teacher_logits, policy_logits=policy_logits,
    )
    assert abs(stats["serl_lambda"] - 0.6) < 1e-9
    assert stats["serl_kl"] > 0.0
    expected_total = (
        stats["loss_rl_act"] + stats["loss_rl_nonact"]
        + stats["serl_lambda"] * stats["serl_kl"]
    )
    assert abs(stats["policy_loss"] - expected_total) < 1e-6


def test_ppo_serl_kl_zero_when_teacher_equals_policy():
    trainer = FACTORTrainer(policy=ToyPolicy(), config=FACTORConfig(),
                            hidden_size=16)
    trainer.step_idx = 10
    batch = _ppo_fixture_batch()
    new_logprobs = [torch.zeros(7, requires_grad=True),
                    torch.zeros(5, requires_grad=True)]
    advs = [torch.zeros(7), torch.zeros(5)]
    action_masks = [torch.ones(7), torch.ones(5)]
    torch.manual_seed(3)
    logits = [torch.randn(7, 32), torch.randn(5, 32)]
    stats = trainer.ppo_update(
        batch, new_logprobs, advs, action_masks,
        teacher_logits=[g.clone() for g in logits],
        policy_logits=[g.clone().requires_grad_(True) for g in logits],
    )
    assert abs(stats["serl_kl"]) < 1e-6
