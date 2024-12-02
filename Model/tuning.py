# PARAMS:
# LOSS: l1, inv, dev, cond 
# MODEL: lu, chol, sym, ind
# TRAINING: lr, batch_size

# BUFFER
# store every INPUT/OUTPUT pair in /BUFFER
# these will be used to pretrain the critic

# EVAL
# each resulting combination should be scored by a its evaluation of near spartiy (np.allclose)
# and preconditioning over a few batches of data. Save these results.
# also save a few images from each set using the eval_utility.inspect_instance to see their patter

# NOTES:
# make sure to wrap a each case in a try catch incase it errors out

# TEST
# testing initally with small epoches, ensure each one is producing different results

from Model import linear
from Objectives import losses
from Data import dataset, generator
from Data import utility as data_utility
import itertools
import numpy as np
import torch
import torch.optim as optim
import math


def hyperparam_tuning(loss_params, models, training_params, epoches, device, buffer_path, size, t_len, save_freq):
    l1s, invs, devs, conds = loss_params  # Unpack the different loss parameter options
    lr, batch_size = training_params  # Learning rate and batch size


    param_combinations = itertools.product(models, l1s, invs, devs, conds, lr, batch_size)
    param_log = {}

    for model_p, l1, inv, dev, cond, learning_rate, b_size in param_combinations:
        try:
            loader = initialize_dataloader(t_len, b_size, size)
            model = initialize_model(model_p[0], model_p[1], device, b_size)  
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            criterion = initialize_loss(l1, inv, dev, cond, b_size, size)
            
            path = f'{buffer_path}/model_{model_p[0]}_l1_{l1}_inv_{inv}_dev_{dev}_cond_{cond}_lr_{learning_rate}_bsize_{b_size}'
            train_log = tuning_loop(epoches, criterion, loader, model, optimizer, b_size, size, path, device, save_freq)
            
            param_log[(model_p[0], l1, inv, dev, cond, learning_rate, b_size)] = train_log
            
        except Exception as e:
            print(f"Error with model {model_p[0]}, l1: {l1}, inv: {inv}, dev: {dev}, cond: {cond}, lr: {learning_rate}, batch size: {b_size}")
            print(e)
            param_log[(model_p[0], l1, inv, dev, cond, learning_rate, b_size)] = None
            
    return param_log

    
def initialize_model(model_type, model_params, device, batch_size):
    model = None
    
    if model_type == 'ind':
        model = linear.NonsymmetricIdFCN(**model_params)
    elif model_type == 'sym':
        model = linear.SymmetricIdFCN(**model_params)
    elif model_type == 'lu':
        model = linear.LuFCN(**model_params)
    else:
        model = linear.CholeskyFCN(**model_params)
        
    model.batch_size = batch_size
    
    return model.to(device)

def initialize_loss(l1, inv, dev, cond, b_size, size):
    loss_params = {
        'l1': l1,
        'inv': inv,
        'dev': dev,
        'cond': cond,
        'batch_size': b_size,
        'size': size
    }
    return losses.CondEyeDistance(**loss_params)

def initialize_dataloader(t_len, b_size, size):
    dataset_params = {
        'generator': generator.generate_2d_diffusion_spd,
        'norm_feat': data_utility.normalize_features,
        'epoch_len': t_len,
        'batch_size': b_size,
        'size': int(math.sqrt(size))
    }
    dset = dataset.SystemDataset(**dataset_params)

    loader_params = {
        'dataset': dset,
        'batch_size': b_size,
        'shuffle': False
    }
    return dataset.get_dataloader(**loader_params)


def save_batches(inp, out, batch_path):
    torch.save({'input': inp, 'output': out}, batch_path)
        
        
def tuning_loop(epoches, criterion, train_loader, model, optimizer, batch_size, size, path, device, save_freq):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-2)
    train_log = np.zeros(epoches)
    
    for epoch in range(epoches):
        
        train_loss = 0
        model.train()
        for b_idx, batch in enumerate(train_loader):
            A, b = batch
            A, b = A.to(device), b.to(device)
            output = model(A.view(batch_size, size*size))
            loss = criterion(A, output.view(batch_size, size, size))
            
            
            if torch.isnan(output).any():
                raise Exception("Nan value encountered.")
                
            if save_freq is not None and b_idx % save_freq == 0:
                batch_path = f'{path}_epoch_{epoch}_{b_idx}'
                inp, out = A.view(batch_size, size, size).cpu(), output.view(batch_size, size, size).detach().cpu()
                save_batches(inp, out, batch_path)
                
            train_loss += loss.item()

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
                
        train_loss = train_loss/len(train_loader)
        
        print(f'Case: {path}, Epoch: {epoch}, Train: {train_loss}')
        train_log[epoch] = train_loss
        
        scheduler.step(train_loss)
        
    return train_log


