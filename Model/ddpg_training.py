from tqdm.notebook import tqdm
import torch.optim as optim
import torch
import numpy as np
from collections import deque


def enqueue(buffer, state, action):
    buffer.append((state, action))


# TODO finish training loop for ddpg
def train(epoches, actor, critic, train_set, valid_set, device, train_log, 
          valid_log, critic_crit, actor_crit, actor_lr, critic_lr, memory, wait, step, verbose):
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='min', factor=0.1, patience=5, threshold=1e-2)
    critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode='min', factor=0.1, patience=5, threshold=1e-2)
    
    buffer = deque(maxlen=memory)
    
    best_val_loss = float('inf')
    
    for epoch in range(epoches):
        
        train_loss = 0
        actor.train()
        critic.train()
        t_loader = tqdm(train_set, desc=f'Actor-Train', leave=False, mininterval=2.0) if verbose else train_set
        for batch_idx, batch in enumerate(t_loader):
            actor_update, critic_update = step(batch_idx, actor, critic, batch, critic_crit, actor_crit, buffer, wait, critic_optimizer, actor_optimizer, verbose)
            
            if actor_update is not None:
                if verbose:
                    t_loader.set_postfix(actor_updates=actor_update, critic_update=critic_update)
                
                
                
            
    
    