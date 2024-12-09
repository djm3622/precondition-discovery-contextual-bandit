from tqdm.notebook import tqdm
import torch.optim as optim
import torch
import numpy as np
from collections import deque


def enqueue(buffer, state, action):
    buffer.append((state, action))


# TODO finish training loop for ddpg
def train(epoches, actor, critic, train_set, valid_set, device, train_log, 
          valid_log, critic_crit, actor_crit, actor_lr, critic_lr, memory, wait, step, verbose, buffer):
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='min', factor=0.1, patience=5, threshold=1e-2)
    critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode='min', factor=0.1, patience=5, threshold=1e-2)
    
    if buffer is None:
        buffer = deque(maxlen=memory)
    
    best_val_loss = float('inf')
    
    for epoch in range(epoches):
        
        actor_loss, critic_loss = 0, 0
        actor.train()
        critic.train()
        t_loader = tqdm(train_set, desc=f'Actor-Train', leave=False, mininterval=2.0) if verbose else train_set
        for batch_idx, batch in enumerate(t_loader):
            actor_update, critic_update = step(batch_idx, actor, critic, batch, critic_crit, actor_crit, buffer, wait, critic_optimizer, actor_optimizer, verbose)
            
            if verbose:
                t_loader.set_postfix(actor_updates=actor_update, critic_update=critic_update)
                
            actor_loss += actor_update
            critic_loss += critic_update
            
        print(f'Epoch {epoch}/{epoches}: Actor={actor_loss/len(t_loader) * wait}, Critic={critic_loss/len(t_loader) * wait}')
        train_log.append([actor_loss/len(t_loader) * wait, critic_loss/len(t_loader) * wait])
        
    return buffer


def train_experimental(epoches, actor, critic, train_set, valid_set, device, train_log, 
          valid_log, critic_crit, actor_crit, actor_lr, critic_lr, memory, wait, step, 
          verbose, buffer, condition_scheduler, noise_scheduler, opts=None):
    
    if opts is None:
        actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    else:
        actor_optimizer, critic_optimizer = opts
    
    if buffer is None:
        buffer = deque(maxlen=memory)
        
    for epoch in range(epoches):
        
        actor_loss, critic_loss = 0, 0
        actor.train()
        critic.train()
        t_loader = tqdm(train_set, desc=f'Actor-Train', leave=False, mininterval=2.0) if verbose else train_set
        
        condition_weight = condition_scheduler.get_value()
        exploration_prob = noise_scheduler.get_value()
        
        for batch_idx, batch in enumerate(t_loader):
            actor_update, critic_update = step(epoch, batch_idx, actor, critic, batch, critic_crit, 
                            actor_crit, buffer, wait, critic_optimizer, actor_optimizer, verbose, 
                            condition_weight, exploration_prob)
            
            if verbose:
                t_loader.set_postfix(actor_updates=actor_update, critic_update=critic_update)
                
            actor_loss += actor_update
            critic_loss += critic_update
            
        print(f'Epoch {epoch}/{epoches}: Actor={actor_loss/len(t_loader) * wait}, Critic={critic_loss/len(t_loader) * wait}')
        train_log.append([actor_loss/len(t_loader) * wait, critic_loss/len(t_loader) * wait])
        
        condition_scheduler.step_forward()
        noise_scheduler.step_forward()
                
    return buffer, condition_scheduler, noise_scheduler, actor_optimizer, critic_optimizer
                
            
                    
            
                
                
                
            
    
    