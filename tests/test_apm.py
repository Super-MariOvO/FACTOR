"""Unit tests for the APM component."""

import torch

from factor.apm import (
    action_mean_reduce,
    apm_token_advantages,
    compute_action_advantages,
)


def test_action_mean_reduce():
    toks = [torch.tensor([1.0, 3.0]), torch.tensor([2.0, 2.0, 2.0])]
    advs = action_mean_reduce(toks)
    assert torch.allclose(advs[0], torch.tensor(2.0))
    assert torch.allclose(advs[1], torch.tensor(2.0))


def test_apm_mean_preservation():
    action_adv = torch.tensor([1.5, -0.5])
    omegas = [
        torch.tensor([0.5, 1.0, 1.5]),   # mean one
        torch.tensor([2.0, 0.0]),        # mean one
    ]
    out = apm_token_advantages(action_adv, omegas)
    assert abs(out[0].mean().item() - 1.5) < 1e-6
    assert abs(out[1].mean().item() - (-0.5)) < 1e-6
    assert torch.allclose(out[0], omegas[0] * 1.5)


def test_compute_action_advantages_td_residual():
    # One-step TD residual on boundary-adjusted values (paper Eq. 3-5).
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.2, 0.3, 0.5])
    # v_tilde = [b=0.1, 0.3, 0.5, 0]; A* = r + v_tilde[1:] - v_tilde[:-1]
    adv = compute_action_advantages(rewards, values, baseline=0.1)
    assert torch.allclose(adv, torch.tensor([0.2, 0.2, 0.5]), atol=1e-6)


def test_compute_action_advantages_telescopes():
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.2, 0.3, 0.5])
    baseline = 0.1
    adv = compute_action_advantages(rewards, values, baseline=baseline)
    a_seq = rewards.sum().item() - baseline
    assert abs(adv.sum().item() - a_seq) < 1e-6
