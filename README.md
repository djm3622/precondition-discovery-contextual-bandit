# Preconditioner Discovery via Contextual Bandit Reinforcement Learning

This repository contains the implementation for **Preconditioner Discovery via Contextual Bandit Reinforcement Learning**, a method that combines reinforcement learning techniques with contextual bandits to optimize the selection of preconditioners for ill-conditioned systems. This approach is specifically suited for SPD systems arising from 2d diffusion equations, and similar equations.

## Abstract
Traditional preconditioners such as Jacobi, Incomplete LU, and Algebraic Multigrid methods offer problem-specific advantages but rely heavily on hyperparameter tuning. Recent advances have explored using deep neural networks to learn preconditioners, though challenges such as non-differentiable objectives and costly training procedures remain. This work introduces a reinforcement learning approach for learning preconditioners, specifically, a contextual bandit formulation. The proposed framework utilizes an actor-critic model, where the actor generates the Cholesky decomposition of preconditioners, and the critic evaluates them based on reward-based feedback. To further guide the training, we design a dual-objective loss function combining the critic and condition number minimization. We contribute a generalizable preconditioner learning method, dynamic sparsity exploration, and cosine schedulers for improved training stability. We compare our approach to traditional and neural preconditioners, demonstrating improved flexibility.


---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/djm3622/precondition-discovery-contextual-bandit.git
   cd precondition-discovery-contextual-bandit
   ```
---

## Usage

### Pre-Training Actor

```python
foo
```

### Pre-Training Critic
```python
foo
```

### Complete Algorithm
```python
foo
```

---

## Project Structure

```
.
├── Data/
│   ├── dataset.py
│   ├── generator.py
│   ├── critic_dataset.py
│   └── utility.py
├── JScripts/
│   ├── actor_critic.ipynd
│   ├── hyperparameter_tuning.ipynb
│   ├── pretrain_critic.ipynb
│   ├── actor_critic-experimental.ipynb
│   ├── actor_critic-experimental-dupe1.ipynb
│   ├── actor_critic-experimental-dupe2.ipynb
│   └── script_testing.ipynb
├── Evaluation/
│   └── utility.py
├── Model/
│   ├── ac_training.py
│   ├── actors.py
│   ├── critics.py
│   ├── scipy.py
│   ├── tuning.py
│   └── utility.py
├── Objectives/
│   ├── losses.py
│   └── rewards.py
└── README.md
```

## Contributions

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Specify the license (e.g., MIT, Apache 2.0)

## Contact

- Project Maintainer: [Your Name]
- Project Link: https://github.com/djm3622/precondition-discovery-contextual-bandit

## Acknowledgments

- List any references, papers, or resources that inspired this project
- Mention any funding sources or institutional support
