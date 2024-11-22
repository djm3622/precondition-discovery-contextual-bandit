from tqdm.notebook import tqdm
import torch.optim as optim
import torch
import numpy as np
import os
import re
from collections import defaultdict


def load_model(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))
    
    
def save_model(model, state_dict_path):
    torch.save(model.state_dict(), state_dict_path)

    
def check_point(current_val_loss, best_val_loss, model, save_path):
    if current_val_loss <= best_val_loss:
        save_model(model, save_path)
        return current_val_loss
    return best_val_loss
    
    
# TODO update to take in a step functions
# use the one from koopman vit as reference
def shared_training_loop(epoches, criterion, train_loader, valid_loader, model, lr, size, batch_size, device, verbose, file_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-2)

    train_log, valid_log = np.zeros(epoches), np.zeros(epoches)
    best_val_loss = float('inf')
    
    for epoch in range(epoches):
        
        train_loss = 0
        model.train()
        t_loader = tqdm(train_loader, desc=f'Train', leave=False, mininterval=2.0) if verbose else train_loader
        for batch in t_loader:
            A, b = batch
            A, b = A.to(device), b.to(device)
            output = model(A.view(batch_size, size*size))
            loss = criterion(A, output.view(batch_size, size, size))

            if verbose:
                t_loader.set_postfix(train_loss=loss.item())
                
            train_loss += loss.item()

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
        valid_loss = 0
        model.eval()
        v_loader = tqdm(valid_loader, desc=f'Valid', leave=False, mininterval=2.0) if verbose else valid_loader
        for batch in v_loader:
            with torch.no_grad():
                A, b = batch
                A, b = A.to(device), b.to(device)
                output = model(A.view(batch_size, size*size))
                loss = criterion(A, output.view(batch_size, size, size))

                if verbose:
                    v_loader.set_postfix(valid_loss=loss.item())
                    
                valid_loss += loss.item()
                
        train_loss = train_loss/len(t_loader)
        valid_loss = valid_loss/len(v_loader)
        
        print(f'Epoch: {epoch}, Train: {train_loss}, Valid: {valid_loss}')
        train_log[epoch], valid_log[epoch] = train_loss, valid_loss
        
        scheduler.step(valid_loss)
        best_val_loss = check_point(valid_loss, best_val_loss, model, file_path)
        
    return train_log, valid_log


def group_files_by_parameters(file_list):

    regex = r"model_(?P<model>[a-z]+)_l1_(?P<l1>[0-9e\.-]+)_inv_(?P<inv>[0-9e\.-]+)_dev_(?P<dev>[0-9e\.-]+)_cond_(?P<cond>[0-9e\.-]+)_lr_(?P<lr>[0-9e\.-]+)_bsize_(?P<bsize>[0-9]+)"
    
    grouped_files = defaultdict(list)

    for file in file_list:
        match = re.search(regex, file)
        if match:
            key = (
                match.group("model"),
                match.group("l1"),
                match.group("inv"),
                match.group("dev"),
                match.group("cond"),
                match.group("lr"),
                match.group("bsize"),
            )
            grouped_files[key].append(file)

    return dict(grouped_files)


def get_torch_files(directory):
    torch_files = [file for file in os.listdir(directory)]
    return torch_files


def group_files_by_single_parameters(file_list):
    regex = r"model_(?P<model>[a-z]+)_l1_(?P<l1>[0-9e\.-]+)_inv_(?P<inv>[0-9e\.-]+)_dev_(?P<dev>[0-9e\.-]+)_cond_(?P<cond>[0-9e\.-]+)_lr_(?P<lr>[0-9e\.-]+)_bsize_(?P<bsize>[0-9]+)"
    
    grouped_by_param = defaultdict(lambda: defaultdict(list))
    
    for file in file_list:
        match = re.search(regex, file)
        if match:
            params = match.groupdict()
            
            for param, value in params.items():
                grouped_by_param[param][value].append(file)
    
    return grouped_by_param


def calc_loss(filenames, root_dir, loss_function, timeit):
    total_loss = 0.0
    total_batches = 0
    
    mmin = {'name': None, 'value': float('inf')}
    mmax = {'name': None, 'value': float('-inf')}
    
    loader = tqdm(filenames, desc=f'Time', leave=False) if timeit else filenames

    for file in loader:
        # Load the Torch tensor file
        data = torch.load(os.path.join(root_dir, file), weights_only=True)
        
        inp = data['input']
        output = data['output']
        
        # Ensure inp and output have the same shape
        if inp.shape != output.shape:
            raise ValueError(f"Shape mismatch in file {file}: inp shape {inp.shape}, output shape {output.shape}.")
        
        # Calculate the loss for the current file
        loss = loss_function(inp, output)
        
        if loss.item() == float('inf') or loss.item() == float('-inf') or loss.item() == float('nan'):
            print(f'Bad value for loss encountered! {loss.item()}')
        
        if loss < mmin['value']:
            mmin['name'], mmin['value'] = file, loss
        if loss > mmax['value']:
            mmax['name'], mmax['value'] = file, loss
        
        total_loss += loss.item() * inp.size(0)  # Scale by batch size
        total_batches += inp.size(0)

    # Compute the average loss
    average_loss = total_loss / total_batches if total_batches > 0 else float('nan')
    return average_loss, mmin, mmax