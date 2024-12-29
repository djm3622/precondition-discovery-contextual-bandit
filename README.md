# Preconditioner Discovery via Contextual Bandit Reinforcement Learning

This repository contains the implementation for **Preconditioner Discovery via Contextual Bandit Reinforcement Learning**, a method that applies reinforcement learning techniques to numerical methods to better select preconditioners for ill-conditioned systems. This approach is specifically suited for SPD systems arising from 2d diffusion equations, and similar equations.

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
For an in-depth example, please refer to <a href='https://github.com/djm3622/precondition-discovery-contextual-bandit/blob/main/JScripts/script_testing.ipynb'>/JScripts/script_testing.ipynb<a>.
```python
...
model_params = {
    'n':Config.mat_size,
    'hidden':Config.hidden,
    'batch_size':Config.batch_size,
    'sparse_tol':Config.sparse_tol,
    'diagonal_bias':Config.diagonal_bias
}
model = actors.CholeskyFCN(**model_params).to(device)

loss_params = {
    'l1': Config.l1,
    'inv': Config.inv,
    'dev': Config.dev,
    'cond': Config.cond,
    'batch_size': Config.batch_size,
    'size': Config.mat_size
}
criterion = losses.CondEyeDistance(**loss_params)

def step(batch, model, criterion, device, size, batch_size):
    A, b = batch
    A, b = A.to(device), b.to(device)
    output = model(A.view(batch_size, size*size))
    return criterion(A, output.view(batch_size, size, size))

training_params = {
    'epoches': Config.epoches,
    'criterion': criterion,
    'step': step,
    'train_loader': train_dataloader,
    'valid_loader': valid_dataloader,
    'model': model,
    'lr': Config.lr,
    'size': Config.mat_size,
    'batch_size': Config.batch_size,
    'device': device,
    'verbose': Config.verbose,
    'file_path': Config.file_path
}
train_log, valid_log = model_utility.shared_training_loop(**training_params)
...
```

### Pre-Training Critic
To generate the pretraining data from the hyperparameter tuning of multiple actors please refer to <a href='https://github.com/djm3622/precondition-discovery-contextual-bandit/blob/main/JScripts/hyperparameter_tuning.ipynb'>/JScripts/hyperparameter_tuning.ipynb<a>
For an in-depth example, please refer to <a href='https://github.com/djm3622/precondition-discovery-contextual-bandit/blob/main/JScripts/pretrain_critic.ipynb'>/JScripts/pretrain_critic.ipynb<a>.
```python
...
model_params = {
    'n': Config.n, 
    'down': Config.down, 
    'batch_size': Config.batch_size, 
    'sigmoid_scale': Config.sigmoid
}
critic = critics.SmallerSingleCritic(**model_params).to(device)

reward_func = reward.batched_reward(25, False, False)
criterion = losses.CriticSingleRewardLoss(1, 1, 1, reward_func)

def step(batch, model, criterion, device, size, batch_size):
    A, M, b = critic_dataset.postprocess(*batch)
    A, M, b = A.to(device).float(), M.to(device).float(), b.to(device).float()
    inp = torch.concat([A.view(batch_size, size**2), M.view(batch_size, size**2)], dim=1)
    
    out = model(inp)
    
    loss = criterion(out, A, M, b)
    return loss

training_params = {
    'epoches': Config.epoches,
    'criterion': criterion,
    'step': step,
    'train_loader': train_dl,
    'valid_loader': valid_dl,
    'model': critic,
    'lr': Config.lr,
    'size': Config.mat_size,
    'batch_size': Config.batch_size,
    'device': device,
    'verbose': Config.verbose,
    'file_path': Config.file_path,
    'accumulation_steps': 1
}
train_log, valid_log = model_utility.shared_training_loop(**training_params)
...
```

### Complete Algorithm
For an in-depth example, please refer to <a href='https://github.com/djm3622/precondition-discovery-contextual-bandit/blob/main/JScripts/actor_critic-experimental.ipynb'>/JScripts/actor_critic-experimental.ipynb<a>.
```python
...
critic_params = {
    'n': Config.n, 
    'down': Config.down, 
    'batch_size': Config.batch_size, 
    'sigmoid_scale': Config.sigmoid
}

critic = critics.SmallerSingleCritic(**critic_params).to(device)

reward_func = reward.batched_reward(Config.n, False, Config.switch_up)
if Config.multi:
    critic_crit = losses.CriticMultiRewardLoss(Config.sparse, Config.niter, Config.res, reward_func)
else:
    critic_crit = losses.CriticSingleRewardLoss(Config.sparse, Config.niter, Config.res, reward_func)

actor_params = {
    'n':Config.n,
    'hidden':Config.hidden,
    'batch_size':Config.batch_size,
    'sparse_tol':Config.sparse_tol,
    'diagonal_bias':Config.diagonal_bias
}
actor = actors.LuFCN(**actor_params).to(device)

loss_params = {
    'l1': Config.l1,
    'inv': Config.inv,
    'dev': Config.dev,
    'cond': Config.cond,
    'batch_size': Config.batch_size,
    'size': Config.n
}
actor_crit = losses.CondEyeDistance(**loss_params)

train_log, valid_log = [], []

def step(...):
   ....

ddpg_params = {
    'epoches': 1250,
    'actor': actor,
    'critic': critic,
    'train_set': train_dataloader,
    'valid_set': valid_dataloader,
    'device': device,
    'train_log': train_log,
    'valid_log': valid_log,
    'critic_crit': critic_crit,
    'actor_crit': actor_crit,
    'actor_lr': 1e-6,
    'critic_lr': 1e-6,
    'memory': 10000, 
    'wait': 32,
    'step': step,
    'verbose': False,
    'buffer': buffer,
    'condition_scheduler': condition_scheduler,
    'noise_scheduler': noise_scheduler
}
buffer, condition_scheduler, noise_scheduler, actor_optim, critic_optim = ac_training.train_experimental(**ddpg_params)
...
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
