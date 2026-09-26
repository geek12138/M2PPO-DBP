# M²PPO-DBP

Official implementation of the paper:

> Jinshuo Yang, Zhaoqilin Yang, Wenjie Zhou, Xin Wang, Youliang Tian.
> "M²PPO-DBP: Local Mean-Field Multi-Agent Proximal Policy Optimization with density-based punishment for spatial public goods games."
> Chaos, Solitons and Fractals, 211 (2026) 118851.

## Repository structure

| File | Description |
| --- | --- |
| `M2PPO_DBP.py` | M2PPO-DBP: LMF actor + centralized critic + density punishment |
| `LMFPPO.py` | LMF-PPO baseline (local mean-field) |
| `MAPPO.py` | MAPPO baseline (centralized critic) |
| `Fermi.py` | Fermi update rule baseline |
| `QL_Feimi.py` | Q-learning (+ optional Fermi) baseline |
| `main_M2PPO_DBP.py` | Entry point for M2PPO-DBP |
| `main_LMFPPO.py` | Entry point for LMF-PPO |
| `main_MAPPO.py` | Entry point for MAPPO |
| `main_Fermi.py` | Entry point for Fermi |
| `main_QL.py` | Entry point for Q-learning |
| `scripts/` | Shell scripts with example run commands |

---

## Installation

Requires Python 3.8+ and PyTorch (CUDA recommended for L = 200 lattices).

```bash
pip install torch numpy matplotlib tqdm
```

---

## Usage

### M²PPO-DBP (main method)

```bash
python main_M2PPO_DBP.py -epochs 1000 -runs 1 \
    -L_num 200 -alpha 1e-3 -gamma 0.99 \
    -clip_epsilon 0.2 -question 2 -ppo_epochs 1 \
    -batch_size 1 -gae_lambda 0.95 -seed 1 \
    -delta 0.5 -rho 0.01 -p_punish 0.5 -device cuda
```

### Baselines

```bash
# LMF-PPO
python main_LMFPPO.py -epochs 1000 -runs 1 -L_num 200 -alpha 1e-3 \
    -gamma 0.99 -clip_epsilon 0.2 -question 2 -ppo_epochs 1

# MAPPO
python main_MAPPO.py -epochs 10000 -runs 1 -L_num 200 -alpha 1e-2 -question 1

# Fermi update rule
python main_Fermi.py -epochs 10000 -runs 1 -L_num 200 -question 2 -seed 41

# Q-learning
python main_QL.py -epochs 10000 -runs 1 -L_num 200 -alpha 0.8 \
    -gamma 0.8 -epsilon 0.3 -question 2 -is_QL -seed 41
```

### Command-line arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-epochs` | 1000 | Number of training iterations |
| `-runs` | 1 | Independent runs per r value |
| `-L_num` | 200 | Lattice size L x L |
| `-alpha` | 1e-3 | Learning rate |
| `-gamma` | 0.99 | Discount factor |
| `-clip_epsilon` | 0.2 | PPO clip range |
| `-question` | 2 | Initialization (1 = random 50%, 2 = half-half, 3 = all-defector, 4 = checkerboard) |
| `-ppo_epochs` | 1 | PPO update epochs |
| `-batch_size` | 1 | Mini-batch size |
| `-gae_lambda` | 0.95 | GAE parameter |
| `-delta` | 0.5 | Value-loss coefficient |
| `-rho` | 0.01 | Entropy regularization coefficient |
| `-p_punish` | 0.5 | Density-based punishment strength p (M2PPO-DBP only) |
| `-device` | cuda | Compute device: cuda / cpu / mps |
| `-seed` | 1 | Random seed |

### Default hyperparameters (paper, Table 1)

| Parameter | Value | Description |
| --- | --- | --- |
| L | 200 | Lattice size |
| K | 5 | Public goods groups per agent |
| T | 1000 | Training epochs |
| alpha | 1e-3 | Learning rate |
| gamma | 0.99 | Discount factor |
| lambda | 0.95 | GAE parameter |
| epsilon | 0.2 | PPO clip range |
| delta | 0.5 | Value loss coefficient |
| rho | 0.02 | Entropy regularization coefficient |
| p | 0.5 | Density-based punishment strength |
| hidden dim | 32 | Hidden layer dimension |
| Seed | 1 | Random seed |

---

## Output

Each run creates a timestamped directory under `data/`, containing:

- `Density_C/`, `Density_D/` - cooperator / defector fractions over time
- `Value_C/`, `Value_D/`, `Total_Value/` - average payoffs over time
- `strategy_evolution_r*.pdf/.jpg` - cooperation / defection curves
- `C_D_r.pdf/.jpg` - final cooperation fraction versus r
- `shot_pic/r=<r>/two_type/` - spatial snapshots at t = 0, 1, 10, 100, 1000
- `checkpoint/` - saved model checkpoints

---

## Citation

If you use this code, please cite:

```bibtex
@article{yang2026m2ppodbp,
  title   = {M2PPO-DBP: Local Mean-Field Multi-Agent Proximal Policy Optimization
             with density-based punishment for spatial public goods games},
  author  = {Yang, Jinshuo and Yang, Zhaoqilin and Zhou, Wenjie and Wang, Xin and Tian, Youliang},
  journal = {Chaos, Solitons and Fractals},
  year    = {2026},
  volume  = {211},
  pages   = {118851},
  doi     = {10.1016/j.chaos.2026.118851}
}
```

---

## License

Released for academic and research purposes.

## Contact
- Zhaoqilin Yang - zqlyang@gzu.edu.cn
