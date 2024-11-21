import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from Model import scipy


def view_matrix(batch_mat, ind, ax):
    ax.matshow(batch_mat[ind])

    
def check_singularity(batch_mat, ind):
    mat = batch_mat[ind]
    det_mat = np.linalg.det(mat)
    print(f'Determinent: {det_mat}')

    if np.isclose(det_mat, 0):
        print("Matrix is singular.")
        return False
        
    rank_mat = np.linalg.matrix_rank(mat)
    if rank_mat < mat.shape[0]:
        print("Matrix is singular.")
        return False
    
    print("Matrix is non-singular.")
        
    return True


def check_conditioning(batch_mat, ind):
    mat = batch_mat[ind]
    _, S, _ = torch.linalg.svd(mat)
    cond = S.max() / S.min()
    print(f'Condition: {cond}')


def normalize_features(graph_data):
    x_mean = graph_data.mean()
    x_std = graph_data.std()
    out = (graph_data - x_mean) / (x_std)
    return out.float()


def set_device(device_pref, ind_dev=0):
    device = None
    
    if device_pref == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{ind_dev}')
        print('Now using GPU.')
    else:
        device = torch.device('cpu')
        if device_pref == 'cuda':
            print('GPU not available, defaulting to CPU.')
        else:
            print('Now using CPU.')
    
    return device


def is_spd(batch_mat, ind):
    A = batch_mat[ind]
    
    # Check if A is square
    if A.shape[0] != A.shape[1]:
        return False
    
    # Check if A is symmetric
    if not torch.allclose(A, A.T):
        return False
    
    # Check if all eigenvalues are positive
    eigenvalues = torch.linalg.eigvals(A).real  # Get the real parts of the eigenvalues
    if torch.any(eigenvalues <= 0):
        return False
    
    return True


def seed(seed):
    torch.manual_seed(seed)
    
    
def plot_notable(inst, ax1, ax2, ax3, ind, directory):
    mat = torch.load(os.path.join(directory, inst['name']), weights_only=True)
    inp, output = mat['input'], mat['output']
    
    inner = output @ inp 
    _, S, _ = torch.linalg.svd(inner)
    cond = S.max() / S.min()
    
    view_matrix(inp, ind, ax1)
    view_matrix(output, ind, ax2)
    view_matrix(output @ inp, ind, ax3)
    
    ax3.set_title(f'COND: {cond}')

    
def calc_results(avg, minn, maxx, ind, directory, size):
    fig, axs = plt.subplots(2, 3, figsize=size)
    plot_notable(minn, axs[0][0], axs[0][1], axs[0][2], ind, directory)
    print(f'Minimum: {minn["value"]}')

    plot_notable(maxx, axs[1][0], axs[1][1], axs[1][2], ind, directory)
    print(f'Maximum: {maxx["value"]}')

    print(f'Average: {avg}')
    

def solver_results(inst, directory, device, extra):
    instance = torch.load(os.path.join(directory, inst['name']), weights_only=True)
    iput, output = instance['input'], instance['output']

    _, info = scipy.run_solver(iput, output, device)
    print(f'Mean residuals ({extra}): {torch.mean(info["last_residual"])}')