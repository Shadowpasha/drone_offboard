# GLBATT: Global-Local Belief Attention

Official repository for **GLBATT (Global-Local Belief Attention)**, a memory-efficient and highly stable architecture for solving long-horizon Partially Observable Markov Decision Processes (POMDPs) in Deep Reinforcement Learning.

> **Paper**: *GLBATT: Addressing Long-Horizon POMDPs via Tiered Striding and Belief Abstraction*

This directory contains the complete, clean codebase prepared for **publication and reproducibility**. It includes the core GLBATT library, all training scripts for the proposed method, and all baseline scripts needed to reproduce every result reported in the paper.

---

## Repository Structure

```
GLBATT_publishing/
├── glbatt/                                   # Core neural network & utility library
│   ├── __init__.py
│   ├── utils.py                              # Flat replay buffer (for MLP baselines)
│   ├── utils_transformer.py                  # High-efficiency sequence replay buffer (GLBATT)
│   ├── utils_T_TD3.py                        # Baseline sequence replay buffer (Unstabilized GLBATT)
│   ├── architectures/                        # Neural network model definitions
│   │   ├── __init__.py
│   │   ├── CrossQ_GLBATT.py                 # Main model: Stabilized CrossQ + GLBATT
│   │   ├── CrossQ_GLBATT_NoFusion.py        # Ablation: Without Multi-View Decision Fusion
│   │   ├── CrossQ_GLBATT_NoStriding.py      # Ablation: Contiguous sequence (no Tiered Striding)
│   │   ├── GLBATT_NoGlobal.py               # Ablation: Local-only attention window
│   │   └── GLBATT_Summary.py                # Standard TD3 + GLBATT (Unstabilized baseline)
│   └── gtrxl_torch/                         # GTrXL Transformer backbone implementation
│       ├── __init__.py
│       └── gtrxl_torch.py
├── environments/                            # Custom benchmarking environments
│   ├── __init__.py
│   ├── complex_envs.py                      # Blind corridor and differential trap environments
│   ├── differential_lidar_env.py            # Differential drive robot simulator
│   ├── holonomic_lidar_env.py               # Standard holonomic navigation environment
│   ├── holonomic_lidar_env_moving.py        # Navigation with dynamic/moving obstacles (primary POMDP testbed)
│   ├── holonomic_360_lidar_env.py           # Full 360° LiDAR navigation environment
│   └── mujoco_pomdp_wrapper.py              # POMDP joint-masking wrapper for MuJoCo locomotion
├── scripts/                                 # Reproducibility training scripts
│   │
│   ├── # ── GLBATT (Proposed Method) ─────────────────────────────────────────
│   ├── train_gym_GLBATT.py                  # Unstabilized GLBATT on Gym benchmarks
│   ├── train_gym_CrossQ_GLBATT.py           # Stabilized CrossQ-GLBATT on Gym benchmarks
│   ├── train_mujoco_pomdp_GLBATT.py         # Unstabilized GLBATT on MuJoCo POMDP
│   ├── train_mujoco_pomdp_CrossQ_GLBATT.py  # Stabilized CrossQ-GLBATT on MuJoCo POMDP
│   ├── train_GLBATT_pygame.py               # Unstabilized GLBATT on PyGame navigation
│   ├── train_CrossQ_GLBATT_pygame.py        # Stabilized CrossQ-GLBATT on PyGame navigation
│   │
│   ├── # ── Ablations ────────────────────────────────────────────────────────
│   ├── train_mujoco_pomdp_CrossQ_GLBATT_NoFusion.py    # Ablation: No Decision Fusion
│   ├── train_mujoco_pomdp_CrossQ_GLBATT_NoStriding.py  # Ablation: No Tiered Striding
│   │
│   ├── # ── MLP Baselines (Gymnasium) ────────────────────────────────────────
│   ├── train_gym_TD3.py                     # TD3 baseline on Gym benchmarks
│   ├── train_gym_SAC.py                     # SAC baseline on Gym benchmarks
│   ├── train_gym_CrossQ.py                  # CrossQ baseline on Gym benchmarks
│   ├── train_gym_RecurrentPPO.py            # RecurrentPPO (LSTM) on Gym benchmarks
│   │
│   ├── # ── MLP Baselines (MuJoCo POMDP) ────────────────────────────────────
│   ├── train_mujoco_pomdp_TD3.py            # TD3 baseline on MuJoCo POMDP
│   ├── train_mujoco_pomdp_SAC.py            # SAC baseline on MuJoCo POMDP
│   ├── train_mujoco_pomdp_CrossQ.py         # CrossQ baseline on MuJoCo POMDP
│   ├── train_mujoco_pomdp_RecurrentPPO.py   # RecurrentPPO baseline on MuJoCo POMDP
│   │
│   └── # ── MLP Baselines (PyGame Navigation) ───────────────────────────────
│       ├── train_pygame_TD3.py              # TD3 baseline on PyGame navigation
│       ├── train_pygame_SAC.py              # SAC baseline on PyGame navigation
│       └── train_pygame_CrossQ.py           # CrossQ baseline on PyGame navigation
│
└── README.md
```

---

## Installation

Set up a virtual environment with Python 3.10 or later:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core deep learning packages
pip install torch numpy gymnasium tensorboard

# Install benchmark-specific dependencies
pip install pygame mujoco

# Install on-policy baseline dependencies
pip install stable-baselines3 sb3-contrib

# For CrossQ baselines: install the JAX-based SBX library
# (clone CrossQ-main into the repo root and install)
git clone https://github.com/perrin-isir/xpag CrossQ-main
pip install -e CrossQ-main/
pip install jax flax optax
```

---

## Running Training Benchmarks

All scripts run with default configurations matched to the parameters reported in the paper. Scripts are run **from the `scripts/` directory** (or from the repo root using `python3 scripts/<script>.py`).

### 1. Classical Gymnasium Benchmarks (Continuous Control)

Train GLBATT or baselines on `BipedalWalker-v3` and `LunarLanderContinuous-v3`:

```bash
# ── GLBATT (Proposed) ───────────────────────────────────────────────
# Stabilized CrossQ + GLBATT (main result)
python3 scripts/train_gym_CrossQ_GLBATT.py --env BipedalWalker-v3 --seed 1236

# Unstabilized GLBATT (TD3-style polyak target; baseline variant)
python3 scripts/train_gym_GLBATT.py --env BipedalWalker-v3 --seed 1236

# ── MLP Baselines ────────────────────────────────────────────────────
python3 scripts/train_gym_TD3.py --env BipedalWalker-v3 --seed 1236
python3 scripts/train_gym_SAC.py --env BipedalWalker-v3 --seed 1236
python3 scripts/train_gym_CrossQ.py --env BipedalWalker-v3 --seed 1236
python3 scripts/train_gym_RecurrentPPO.py --env BipedalWalker-v3 --seed 1236
```

### 2. MuJoCo POMDP Benchmarks (Locomotion with Occluded Joint States)

Train on `Hopper-v4` with the distal foot joint fully masked to simulate sensor failure:

```bash
# ── GLBATT (Proposed) ───────────────────────────────────────────────
python3 scripts/train_mujoco_pomdp_CrossQ_GLBATT.py --env Hopper-v4 --seed 1236
python3 scripts/train_mujoco_pomdp_GLBATT.py --env Hopper-v4 --seed 1236

# ── Ablations ───────────────────────────────────────────────────────
python3 scripts/train_mujoco_pomdp_CrossQ_GLBATT_NoFusion.py --env Hopper-v4 --seed 1236
python3 scripts/train_mujoco_pomdp_CrossQ_GLBATT_NoStriding.py --env Hopper-v4 --seed 1236

# ── MLP Baselines ────────────────────────────────────────────────────
python3 scripts/train_mujoco_pomdp_TD3.py --env Hopper-v4 --seed 1236
python3 scripts/train_mujoco_pomdp_SAC.py --env Hopper-v4 --seed 1236
python3 scripts/train_mujoco_pomdp_CrossQ.py --env Hopper-v4 --seed 1236
python3 scripts/train_mujoco_pomdp_RecurrentPPO.py --env Hopper-v4 --seed 1236
```

### 3. PyGame 2D Navigation Benchmarks (LiDAR-based Partial Observability)

Train in the custom 2D navigation environments. The `--env` flag selects between:
- `moving` — Holonomic robot with 6 dynamic moving obstacles (primary POMDP testbed)
- `corridor` — Blind corridor requiring long-horizon memory to navigate

```bash
# ── GLBATT (Proposed) ───────────────────────────────────────────────
# Moving obstacles environment (primary POMDP testbed)
python3 scripts/train_CrossQ_GLBATT_pygame.py --env moving --seed 1236

# Blind corridor environment
python3 scripts/train_CrossQ_GLBATT_pygame.py --env corridor --seed 1236

# ── MLP Baselines ────────────────────────────────────────────────────
python3 scripts/train_pygame_TD3.py --env moving --seed 1236
python3 scripts/train_pygame_SAC.py --env moving --seed 1236
python3 scripts/train_pygame_CrossQ.py --env moving --seed 1236
```

---

## Monitoring Training Progress

All scripts log training metrics via PyTorch TensorBoard:

```bash
tensorboard --logdir=runs/
```

Then open `http://localhost:6006` in your browser. Logged metrics include:
- `Reward/Episode` — Episode cumulative reward
- `Stats/SuccessRate` — Rolling 100-episode success rate (navigation tasks)
- `Loss/Critic` — TD3/SAC critic loss
- `Loss/Safety_Grounding` — Belief Abstraction auxiliary loss
- `Attention/GlobalUsage` and `Attention/LocalUsage` — Tiered memory attention allocation

---

## Baselines Summary

| Algorithm | Type | Description |
|-----------|------|-------------|
| **CrossQ-GLBATT** | Proposed | Full method: Tiered Striding + Decision Fusion + Belief Abstraction + 2N-Batch Stabilization |
| **Unstab. GLBATT** | Proposed (variant) | GLBATT without 2N-batch stabilization (standard TD3 polyak targets) |
| **TD3** | MLP Baseline | Twin Delayed DDPG with standard MLP actor-critic |
| **SAC** | MLP Baseline | Soft Actor-Critic with entropy regularization |
| **CrossQ** | MLP Baseline | CrossQ with 2N-batch normalization, MLP backbone only |
| **RecurrentPPO** | On-policy Baseline | PPO with LSTM backbone via SB3-contrib |

> **Note on CrossQ baselines**: CrossQ scripts require the [SBX](https://github.com/perrin-isir/xpag) JAX-based library installed from `CrossQ-main/`. All other scripts use pure PyTorch and require no additional setup beyond the packages listed above.
