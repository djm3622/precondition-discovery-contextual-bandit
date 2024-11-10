from tqdm.notebook import tqdm
import torch.optim as optim
import torch
import numpy as np
import os


def load_model(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))
    
    
def save_model(model, state_dict_path):
    torch.save(model.state_dict(), state_dict_path)

    
def check_point(current_val_loss, best_val_loss, model, save_path):
    if current_val_loss <= best_val_loss:
        save_model(model, save_path)
        return current_val_loss
    return best_val_loss
    
    
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


