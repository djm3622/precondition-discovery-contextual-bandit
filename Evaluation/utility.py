import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from Model import utility as model_utility
from Data import utility as data_utility


def save_logs(train_log, valid_log, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    np.savez(log_path, train_log=train_log, valid_log=valid_log)
    print(f"Logs saved to: {log_path}")
    
    
def read_logs(log_path):
    # Load the .npz file
    logs = np.load(log_path)
    
    # Extract the arrays
    train_log = logs['train_log']
    valid_log = logs['valid_log']
    print(f"Logs read from: {log_path}")
    
    return train_log, valid_log

    
def plot_training(train_log, valid_log, title='EX', file_path='ex.png'):
    plt.plot(train_log, label="Train")
    plt.plot(valid_log, label="Valid")
    plt.title(f'{title}: Train/Valid Log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(file_path)
    plt.show()


def inspect_instance(A, model, ind, size, batch_size, device, criterion, file_path='EX.png'):
    model.eval()
    with torch.no_grad():
        out = model(A.view(batch_size, size*size).to(device)).view(batch_size, size, size).cpu()
        print(f'Loss: {criterion(A, out)}')

    inner = out @ A
    
    acond = torch.mean(torch.norm(A, p="fro", dim=(1, 2)) * torch.norm(torch.linalg.inv(A), p="fro", dim=(1, 2)))
    icond = torch.mean(torch.norm(inner, p="fro", dim=(1, 2)) * torch.norm(torch.linalg.inv(inner), p="fro", dim=(1, 2)))
    
    print(f'Inner DET: {torch.mean(torch.linalg.det(inner))}')
    print(f'Output min: {out.min()}')

    # Create figures with colorbars
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    im1 = ax1.matshow(out[ind])
    ax1.set_title(f'Output')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])

    im2 = ax2.matshow(A[ind])
    ax2.set_title(f'Input Matrix A: {acond}')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    im3 = ax3.matshow(inner[ind])
    ax3.set_title(f'Inner Product: {icond}')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()


def evaluate_parameters(single_params, parameter, criterion, directory, device, timeit, size=(12, 8)):
    print(f'========[Evaluating \"{parameter}\"]========')
        
    for param in list(single_params[parameter].keys()):
        print(f'\n++++++++[Running over \"{param}\"]++++++++')
        p = single_params[parameter][param]
        print(f'SIZE: {len(p)}')
        
        calc_loss_params = {
            'filenames': p, 
            'root_dir': directory, 
            'loss_function': criterion,
            'timeit': timeit
        }

        avg, minn, maxx = model_utility.calc_loss(**calc_loss_params)
                      
        data_utility.solver_results(minn, directory, device, 'Minimum')
        data_utility.solver_results(maxx, directory, device, 'Maximum')
        data_utility.calc_results(avg, minn, maxx, 0, directory, size)

        

def bootstrap_train_log(file_path):
    actor_values = []
    critic_values = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Actor=" in line and "Critic=" in line:
                # Extract the actor and critic values
                parts = line.strip().split(", ")
                actor_value = float(parts[0].split("Actor=")[-1])
                critic_value = float(parts[1].split("Critic=")[-1])
                
                actor_values.append(actor_value)
                critic_values.append(critic_value)
    
    return actor_values, critic_values

def plot_ac_training(actor_log, critic_log, title='Cosine Condition/Exploration Scheduler (No Cond)', file_path='cos_train_chol_nopre_nocond_1.png'):
    plt.plot(actor_log, label="Actor")
    plt.plot(critic_log, label="Critic")
    plt.title(f'{title}: Train Log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(file_path)
    plt.show()