"""Unit tests for the HTA component."""

import torch

from factor.hta import (
    HTAConfig,
    build_action_token_weights,
    compute_hindsight_gap,
    compute_outcome_score,
    compute_serl_action_kl,
    compute_token_allocation,
)


def test_eta_schedule():
    cfg = HTAConfig(eta_0=0.7, decay_steps=50, active_start=11, active_end=49)
    assert cfg.eta(1) == 0.0          # warmup: teacher inactive
    assert cfg.eta(10) == 0.0
    assert abs(cfg.eta(11) - 0.7 * (1 - 11 / 50)) < 1e-9  # first active 0.546
    assert cfg.eta(49) > 0.0
    assert cfg.eta(50) == 0.0         # inactive from step 50
    assert cfg.eta(100) == 0.0


def test_gap_computation():
    teacher = torch.tensor([0.0, -1.0, -2.0])
    policy = torch.tensor([-0.5, -1.5, -2.5])
    gap = compute_hindsight_gap(teacher, policy)
    assert torch.allclose(gap, torch.tensor([0.5, 0.5, 0.5]))


def test_gap_not_clipped():
    # Clipping happens in the score (Eq. 6), not in the raw gap.
    teacher = torch.tensor([5.0])
    policy = torch.tensor([-5.0])
    gap = compute_hindsight_gap(teacher, policy)
    assert torch.allclose(gap, torch.tensor([10.0]))


# ---------------------------------------------------------------------------
# Outcome-aligned score (paper Eq. 6): s = sgn(A*) * clip(Delta, -d, +d)
# ---------------------------------------------------------------------------


def test_outcome_score_positive_credit():
    gap = torch.tensor([0.5, -5.0, 10.0])
    s = compute_outcome_score(gap, action_advantage=2.0, d=3.0)
    # sgn > 0: scores keep the clipped gap.
    assert torch.allclose(s, torch.tensor([0.5, -3.0, 3.0]))


def test_outcome_score_negative_credit():
    gap = torch.tensor([0.5, -5.0, 10.0])
    s = compute_outcome_score(gap, action_advantage=-2.0, d=3.0)
    # sgn < 0: scores are the negated clipped gap.
    assert torch.allclose(s, torch.tensor([-0.5, 3.0, -3.0]))


def test_outcome_score_zero_credit():
    gap = torch.tensor([1.0, -2.0])
    s = compute_outcome_score(gap, action_advantage=0.0, d=3.0)
    assert torch.all(s == 0.0)


def test_score_drives_allocation_direction():
    # Paper Eq. 6: with A* > 0 the highest-gap token gets the most credit;
    # with A* < 0 the LOWEST-gap token gets the most (negative) credit.
    gaps = [torch.tensor([2.0, -2.0])]
    pos_scores = [compute_outcome_score(g, 1.0, 3.0) for g in gaps]
    neg_scores = [compute_outcome_score(g, -1.0, 3.0) for g in gaps]
    omega_pos = compute_token_allocation(pos_scores, eta_k=1.0, tau=1.0)[0]
    omega_neg = compute_token_allocation(neg_scores, eta_k=1.0, tau=1.0)[0]
    assert omega_pos[0] > omega_pos[1]
    assert omega_neg[1] > omega_neg[0]


# ---------------------------------------------------------------------------
# Allocation (paper Eq. 7)
# ---------------------------------------------------------------------------


def test_allocation_uniform_when_eta_zero():
    scores = [torch.randn(5)]
    omegas = compute_token_allocation(scores, eta_k=0.0, tau=1.0)
    assert torch.allclose(omegas[0], torch.ones(5))


def test_allocation_mean_one():
    torch.manual_seed(0)
    scores = [torch.randn(8), torch.randn(3)]
    for eta in (0.1, 0.5, 0.7):
        omegas = compute_token_allocation(scores, eta_k=eta, tau=1.0)
        for w, g in zip(omegas, scores):
            assert abs(w.mean().item() - 1.0) < 1e-5
            # rho sums to one per action
            rho = w / g.numel()
            assert abs(rho.sum().item() - 1.0) < 1e-5


def test_allocation_sharpness_increases_with_eta():
    s = torch.tensor([2.0, 0.0, 0.0, 0.0])
    low = compute_token_allocation([s], eta_k=0.2, tau=1.0)[0]
    high = compute_token_allocation([s], eta_k=0.7, tau=1.0)[0]
    assert high[0] > low[0]  # more mass on the highest-score token


def test_dense_weights_builder():
    cfg = HTAConfig()
    gaps = torch.zeros(1, 7)
    spans = [[slice(1, 4), slice(4, 7)]]
    advs = [torch.tensor([1.0, -1.0])]
    gaps[0, 1] = 1.0
    gaps[0, 5] = -1.0
    w = build_action_token_weights(gaps, spans, advs, step=20, config=cfg)
    assert w.shape == (1, 7)
    assert w[0, 0] == 0.0                      # outside actions
    assert abs(w[0, 1:4].mean().item() - 1.0) < 1e-5
    assert abs(w[0, 4:7].mean().item() - 1.0) < 1e-5


def test_dense_weights_respect_score_sign():
    # With A* < 0, within the action the LOW-gap token must receive the
    # larger multiplier (it bears more of the negative credit).
    cfg = HTAConfig(eta_0=1.0)  # force full teacher concentration
    gaps = torch.zeros(1, 4)
    gaps[0, 1] = 2.0    # high-gap token
    gaps[0, 2] = -2.0   # low-gap token
    spans = [[slice(1, 4)]]
    advs = [torch.tensor([-1.0])]
    w = build_action_token_weights(gaps, spans, advs, step=20, config=cfg)
    assert w[0, 2] > w[0, 1]


# ---------------------------------------------------------------------------
# SERL auxiliary action-only KL (SERL Eq. 14, weighted by lambda_k)
# ---------------------------------------------------------------------------


def test_serl_action_kl_zero_when_identical():
    logits = torch.randn(5, 11)
    mask = torch.tensor([0, 1, 1, 0, 1])
    kl = compute_serl_action_kl(logits, logits.clone(), mask)
    assert abs(kl.item()) < 1e-6


def test_serl_action_kl_mask_and_reduction():
    torch.manual_seed(0)
    teacher = torch.randn(4, 7)
    policy = torch.randn(4, 7)
    mask = torch.tensor([1, 0, 1, 0])
    kl_mean = compute_serl_action_kl(teacher, policy, mask, reduction="mean")
    kl_sum = compute_serl_action_kl(teacher, policy, mask, reduction="sum")
    kl_none = compute_serl_action_kl(teacher, policy, mask, reduction="none")
    # KL is nonnegative on masked tokens and reductions are consistent.
    assert torch.all(kl_none[mask.bool()] >= 0.0)
    assert torch.all(kl_none[~mask.bool()] == 0.0)
    assert abs(kl_sum.item() - kl_none.sum().item()) < 1e-6
    assert abs(kl_mean.item() - kl_sum.item() / 2.0) < 1e-6


def test_serl_action_kl_value():
    # Two-point distribution with known forward KL.
    teacher_logits = torch.tensor([[2.0, 0.0]])
    policy_logits = torch.tensor([[0.0, 2.0]])
    mask = torch.ones(1, dtype=torch.float32)
    p_t = torch.softmax(teacher_logits, dim=-1)
    log_ratio = teacher_logits.log_softmax(-1) - policy_logits.log_softmax(-1)
    expected = (p_t * log_ratio).sum(-1)
    kl = compute_serl_action_kl(teacher_logits, policy_logits, mask)
    assert abs(kl.item() - expected.item()) < 1e-6
