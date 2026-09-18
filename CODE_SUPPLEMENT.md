# FACTOR Code Supplement

Complete implementation of FACTOR (Credit-Conserving Action-to-Token Assignment for Multi-Turn Agent Reinforcement Learning).

## Directory Structure

```
factor/
├── tac.py              # TAC: TD Action Credit (one-step TD residual, telescoping identity)
├── hta.py              # HTA: Hindsight Token Allocation (sgn(A*)·clip(Δ,-3,+3), mean-one allocation)
├── apm.py              # APM: Per-Action Mean Preservation (action-mean reduction)
├── environments/       # ALFWorld, WebShop, ScienceWorld wrappers
└── training/           # Trainer + config (full objective: L_RL^act + L_RL^non-act + λ_k L_act^SERL)
scripts/
├── train_factor.py     # Main training script
├── evaluate.py         # Evaluation script
└── make_split_manifests.py  # Split manifest generator
tests/                  # 47 unit tests (TAC/HTA/APM/trainer)
```

## Key Implementation Details

- **TAC (tac.py)**: One-step TD residual A*_t = r_t + V~(x_{t+1}) - V~(x_t) with boundary values V~(x_0)=b_i, V~(x_T)=0. Telescoping identity sum_t A*_t = A^seq holds exactly (verified by unit tests).
- **HTA (hta.py)**: Outcome-aligned score s_{t,j} = sgn(A*_t) · clip(Δ_{t,j}, -3, +3), allocation ρ = (1-η)/L + η·softmax(s/τ), mean-one multiplier ω = L·ρ.
- **APM (apm.py)**: Action-mean reduction (average tokens within action, then across actions). Per-action sum(ρ)=1 preservation.
- **Full objective (trainer.py)**: L = L_RL^act + L_RL^non-act + λ_k·L_act^SERL with λ_k = α_k = max(1-k/50, 0).

## Running Tests

```bash
cd code
python -m pytest tests/ -v
```

47 tests covering TAC telescoping identity, HTA sign/clip behavior, APM action-mean reduction, and full objective composition.

## Known Limitations

This is an algorithm-level component library. The following integration components are NOT included:
- verl/SERL policy wrapper (build_policy, generate_trajectories, compute_logprobs)
- Live ALFWorld/WebShop/ScienceWorld simulators (environment backends are stubs)
- Distributed rollout/vLLM serving layer

These are integration seams that must be connected to the verl/SERL framework (commit b338174, serl_action_mask branch) for end-to-end training.
